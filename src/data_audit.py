"""Read-only audit for the IronFlow competition data.

This script never rewrites source files.  It inventories CSV/XLSX files, checks
schemas, timestamps, missingness, duplicates, train/test overlap, suspicious
target leakage, and cross-table timestamp alignment.  Outputs are evidence
artifacts for review; they are not cleaned data and must not be used as model
inputs directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
TIME_NAMES = ("timestamp", "datetime", "date", "time", "时间", "日期")
TARGET_NAMES = ("generator_1", "generator_all")
SUSPICIOUS_TERMS = (
    "generator",
    "load",
    "power",
    "发电",
    "负荷",
    "电量",
    "功率",
    "target",
    "label",
    "y_",
)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy/path values into strict JSON-compatible values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_strict(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            return frame, encoding
        except Exception as exc:  # pragma: no cover - diagnostics are retained
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Unable to read {path}: {' | '.join(errors)}")


def find_time_column(columns: list[str]) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in columns}
    for name in TIME_NAMES:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for column in columns:
        text = str(column).lower()
        if any(token in text for token in TIME_NAMES):
            return str(column)
    return None


def parse_time(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None:
        return None
    # format='mixed' is available in pandas 2.x and avoids silently coercing
    # mixed date representations into a single inferred format.
    try:
        return pd.to_datetime(frame[column], errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(frame[column], errors="coerce")


def numeric_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue
        result[str(column)] = {
            "min": json_safe(non_null.min()),
            "q01": json_safe(non_null.quantile(0.01)),
            "median": json_safe(non_null.median()),
            "q99": json_safe(non_null.quantile(0.99)),
            "max": json_safe(non_null.max()),
            "mean": json_safe(non_null.mean()),
            "std": json_safe(non_null.std()),
            "negative_count": int((non_null < 0).sum()),
            "unique_numeric": int(non_null.nunique(dropna=True)),
        }
    return result


def audit_csv(path: Path, root: Path) -> dict[str, Any]:
    frame, encoding = read_csv_strict(path)
    columns = [str(c) for c in frame.columns]
    time_column = find_time_column(columns)
    times = parse_time(frame, time_column)

    missing = frame.isna().sum()
    inf_count = 0
    non_numeric: dict[str, int] = {}
    constant: list[str] = []
    all_empty: list[str] = []
    for column in columns:
        series = frame[column]
        all_empty_flag = bool(series.isna().all() or series.astype("string").str.strip().eq("").all())
        if all_empty_flag:
            all_empty.append(column)
        if series.nunique(dropna=True) <= 1:
            constant.append(column)
        numeric = pd.to_numeric(series, errors="coerce")
        inf_count += int(numeric.isin([float("inf"), float("-inf")]).sum())
        non_numeric_count = int((series.notna() & numeric.isna()).sum())
        if non_numeric_count:
            non_numeric[column] = non_numeric_count

    time_audit: dict[str, Any] = {"column": time_column}
    if times is not None:
        valid = times.dropna()
        deltas = valid.sort_values().diff().dropna().dt.total_seconds()
        expected = 900.0
        time_audit.update(
            {
                "invalid_count": int(times.isna().sum()),
                "duplicate_count": int(times.duplicated(keep=False).sum()),
                "unique_count": int(valid.nunique()),
                "monotonic_in_input": bool(times.dropna().is_monotonic_increasing),
                "min": json_safe(valid.min()) if not valid.empty else None,
                "max": json_safe(valid.max()) if not valid.empty else None,
                "delta_seconds": {
                    "mode": json_safe(deltas.mode().iloc[0]) if not deltas.mode().empty else None,
                    "min": json_safe(deltas.min()) if not deltas.empty else None,
                    "max": json_safe(deltas.max()) if not deltas.empty else None,
                    "non_15min_count": int((deltas != expected).sum()),
                    "gap_over_15min_count": int((deltas > expected).sum()),
                    "negative_count": int((deltas < 0).sum()),
                },
            }
        )
        full_grid = pd.date_range(valid.min(), valid.max(), freq="15min") if not valid.empty else pd.DatetimeIndex([])
        observed = set(valid)
        missing_grid = [item for item in full_grid if item not in observed]
        time_audit["expected_15min_grid_count"] = int(len(full_grid))
        time_audit["missing_from_expected_15min_grid_count"] = int(len(missing_grid))
        time_audit["first_grid_missing"] = [json_safe(item) for item in missing_grid[:20]]
        gaps = []
        for left, right, delta in zip(valid.sort_values().iloc[:-1], valid.sort_values().iloc[1:], deltas):
            if delta > expected:
                gaps.append({"from": json_safe(left), "to": json_safe(right), "seconds": json_safe(delta)})
        time_audit["first_gaps"] = gaps[:20]

    suspicious = []
    for column in columns:
        lowered = column.lower()
        terms = [term for term in SUSPICIOUS_TERMS if term in lowered]
        if terms:
            suspicious.append({"column": column, "matched_terms": terms})

    train_or_test = "test" if path.parent.name.lower() == "test" or path.name.lower().startswith("pre_test") else "train"
    target_presence = {}
    if train_or_test == "test":
        for target in TARGET_NAMES:
            if target in frame.columns:
                target_presence[target] = {
                    "non_null": int(frame[target].notna().sum()),
                    "missing": int(frame[target].isna().sum()),
                    "min": json_safe(pd.to_numeric(frame[target], errors="coerce").min()),
                    "max": json_safe(pd.to_numeric(frame[target], errors="coerce").max()),
                }

    return {
        "file": path.relative_to(root).as_posix(),
        "partition": train_or_test,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "encoding": encoding,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": columns,
        "dtypes": {column: str(frame[column].dtype) for column in columns},
        "missing": {column: int(missing[column]) for column in columns if missing[column]},
        "missing_total": int(frame.isna().sum().sum()),
        "missing_rate": float(frame.isna().sum().sum() / max(frame.size, 1)),
        "infinite_numeric_count": inf_count,
        "non_numeric_cells": non_numeric,
        "constant_columns": constant,
        "all_empty_columns": all_empty,
        "duplicate_row_count": int(frame.duplicated(keep=False).sum()),
        "numeric_summary": numeric_summary(frame),
        "time": time_audit,
        "suspicious_columns": suspicious,
        "test_target_presence": target_presence,
    }


def audit_workbook(path: Path, root: Path) -> dict[str, Any]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    sheets: list[dict[str, Any]] = []
    for name in workbook.sheetnames:
        sheet = workbook[name]
        rows = list(sheet.iter_rows(values_only=True))
        nonempty_rows = [row for row in rows if any(value is not None for value in row)]
        preview = [[json_safe(value) for value in row] for row in nonempty_rows[:100]]
        formulas = sum(
            1
            for row in sheet.iter_rows()
            for cell in row
            if isinstance(cell.value, str) and cell.value.startswith("=")
        )
        sheets.append(
            {
                "name": name,
                "max_row": int(sheet.max_row),
                "max_column": int(sheet.max_column),
                "nonempty_row_count": len(nonempty_rows),
                "formula_count": formulas,
                "preview": preview,
            }
        )
    return {
        "file": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "sheets": sheets,
    }


def stem_key(relative: str) -> str:
    name = Path(relative).stem.lower()
    return re.sub(r"^pre_test_", "", name) if name.startswith("pre_test_") else re.sub(r"^pre_", "", name)


def cross_table_audit(root: Path, csv_results: list[dict[str, Any]]) -> dict[str, Any]:
    tables: dict[str, dict[str, Any]] = {}
    for result in csv_results:
        path = root / result["file"]
        frame, _ = read_csv_strict(path)
        time_column = result["time"].get("column")
        times = parse_time(frame, time_column)
        tables[result["file"]] = {"frame": frame, "times": times, "partition": result["partition"]}

    alignment: dict[str, Any] = {}
    for partition in ("train", "test"):
        members = {key: value for key, value in tables.items() if value["partition"] == partition}
        sets = {key: set(value["times"].dropna()) for key, value in members.items() if value["times"] is not None}
        if sets:
            union = set().union(*sets.values())
            intersection = set.intersection(*sets.values()) if len(sets) > 1 else set(next(iter(sets.values())))
            alignment[partition] = {
                "files": sorted(sets),
                "union_count": len(union),
                "intersection_count": len(intersection),
                "exclusive_counts": {key: len(values - intersection) for key, values in sets.items()},
                "exclusive_samples": {key: [json_safe(item) for item in sorted(values - intersection)[:20]] for key, values in sets.items()},
                "pairwise": [],
            }
            names = sorted(sets)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    overlap = sets[left] & sets[right]
                    alignment[partition]["pairwise"].append(
                        {"left": left, "right": right, "overlap_count": len(overlap), "first_overlap": json_safe(min(overlap)) if overlap else None}
                    )

    train_times = set().union(*[set(v["times"].dropna()) for v in tables.values() if v["partition"] == "train" and v["times"] is not None])
    test_times = set().union(*[set(v["times"].dropna()) for v in tables.values() if v["partition"] == "test" and v["times"] is not None])
    overlap = train_times & test_times
    schema_comparison = []
    exact_duplicate_rows = []
    train_by_key = {stem_key(key): value for key, value in tables.items() if value["partition"] == "train"}
    test_by_key = {stem_key(key): value for key, value in tables.items() if value["partition"] == "test"}
    for key in sorted(set(train_by_key) & set(test_by_key)):
        train_columns = [str(c) for c in train_by_key[key]["frame"].columns]
        test_columns = [str(c) for c in test_by_key[key]["frame"].columns]
        schema_comparison.append({
            "family": key,
            "same_column_order": train_columns == test_columns,
            "train_only_columns": sorted(set(train_columns) - set(test_columns)),
            "test_only_columns": sorted(set(test_columns) - set(train_columns)),
        })
        train_frame = train_by_key[key]["frame"]
        test_frame = test_by_key[key]["frame"]
        time_column = find_time_column(train_columns)
        if time_column and time_column in test_frame.columns:
            train_indexed = train_frame.copy()
            test_indexed = test_frame.copy()
            train_indexed[time_column] = parse_time(train_indexed, time_column)
            test_indexed[time_column] = parse_time(test_indexed, time_column)
            common = sorted(set(train_indexed[time_column].dropna()) & set(test_indexed[time_column].dropna()))
            equal_count = 0
            samples = []
            compare_columns = [column for column in train_columns if column != time_column and column in test_columns]
            for timestamp in common:
                left_rows = train_indexed[train_indexed[time_column] == timestamp]
                right_rows = test_indexed[test_indexed[time_column] == timestamp]
                if len(left_rows) != 1 or len(right_rows) != 1:
                    continue
                left = left_rows.iloc[0][compare_columns].astype("string").fillna("<NA>").tolist()
                right = right_rows.iloc[0][compare_columns].astype("string").fillna("<NA>").tolist()
                if left == right:
                    equal_count += 1
                    if len(samples) < 20:
                        samples.append(json_safe(timestamp))
            exact_duplicate_rows.append({"family": key, "count": equal_count, "samples": samples})
    return {
        "within_partition": alignment,
        "train_test_timestamp_overlap_count": len(overlap),
        "train_test_first_overlap": json_safe(min(overlap)) if overlap else None,
        "train_test_last_overlap": json_safe(max(overlap)) if overlap else None,
        "train_last_timestamp": json_safe(max(train_times)) if train_times else None,
        "test_first_timestamp": json_safe(min(test_times)) if test_times else None,
        "schema_comparison": schema_comparison,
        "train_test_exact_duplicate_rows": exact_duplicate_rows,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# IronFlow 数据审计报告（自动生成）",
        "",
        f"- 审计时间（UTC）：{report['audit_time_utc']}",
        f"- 审计状态：**{report['overall_status']}**",
        f"- 文件数：{len(report['files'])}",
        "- 原始文件：只读访问，未执行插值、删除、填充、重采样或覆盖。",
        "",
        "## 1. 文件指纹与结构",
        "",
        "| 文件 | 分区 | 行数 | 列数 | SHA-256（前 16 位） | 时间列 | 缺失单元格 | 重复行 |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for item in report["files"]:
        lines.append(
            f"| `{item['file']}` | {item.get('partition', 'xlsx')} | {item.get('rows', item.get('sheets', [{}])[0].get('nonempty_row_count', ''))} | {item.get('columns', '')} | `{item['sha256'][:16]}` | {item.get('time', {}).get('column', '')} | {item.get('missing_total', '')} | {item.get('duplicate_row_count', '')} |"
        )
    lines.extend(["", "## 2. 阻塞项与警告", ""])
    if not report["findings"]:
        lines.append("未发现自动规则标记的问题。")
    else:
        for finding in report["findings"]:
            lines.append(f"- **{finding['severity']}** `{finding['code']}`：{finding['message']}")

    lines.extend(["", "## 3. 时间与跨表对齐", ""])
    cross = report["cross_table"]
    lines.append(f"- 训练/测试时间戳交集：**{cross['train_test_timestamp_overlap_count']}**")
    lines.append(f"- 训练最后时间：`{cross['train_last_timestamp']}`")
    lines.append(f"- 测试最早时间：`{cross['test_first_timestamp']}`")
    for partition, detail in cross["within_partition"].items():
        lines.append(f"- `{partition}` 分区跨表并集 {detail['union_count']}，交集 {detail['intersection_count']}。")
    lines.extend(["", "## 4. 测试标签可见性", ""])
    for item in report["files"]:
        for target, detail in item.get("test_target_presence", {}).items():
            lines.append(f"- `{item['file']}` 包含 `{target}`：非空 {detail['non_null']}，缺失 {detail['missing']}。该列必须从预测输入契约中排除。")
    lines.extend(["", "## 5. Excel 工作簿摘要", ""])
    for item in report["workbooks"]:
        lines.append(f"### `{item['file']}`")
        for sheet in item["sheets"]:
            lines.append(f"- 工作表 `{sheet['name']}`：{sheet['max_row']}×{sheet['max_column']}，非空行 {sheet['nonempty_row_count']}，公式 {sheet['formula_count']}。")
            if sheet["preview"]:
                lines.append("  - 预览：" + "；".join(" | ".join("" if v is None else str(v) for v in row) for row in sheet["preview"][:4]))
    lines.extend(["", "## 6. 审计结论", "", "当前审计结果只用于冻结数据事实和建模边界。按官方滚动预测答复，起点 t 的已提供观测（包括当前发电量）可以使用，但任何 t 之后的目标或观测都必须禁止。缺失、常量列和边界上下文需要在协议中明确处理后，才允许进入 baseline。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="IronFlow read-only data audit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "outputs" / "data_audit_v0").resolve()
    out.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(list((root / "data").glob("*.csv")) + list((root / "test").glob("*.csv")))
    workbook_paths = sorted((root / "data").glob("*.xlsx"))
    csv_results = [audit_csv(path, root) for path in csv_paths]
    workbook_results = [audit_workbook(path, root) for path in workbook_paths]
    cross = cross_table_audit(root, csv_results)

    findings: list[dict[str, str]] = []
    def add(severity: str, code: str, message: str) -> None:
        findings.append({"severity": severity, "code": code, "message": message})

    for item in csv_results:
        name = item["file"]
        time = item["time"]
        if time.get("column") is None:
            add("BLOCKER", "NO_TIMESTAMP", f"{name} 没有可识别的时间列。")
        elif time.get("invalid_count", 0):
            add("BLOCKER", "INVALID_TIMESTAMP", f"{name} 有 {time['invalid_count']} 个无法解析的时间戳。")
        if time.get("duplicate_count", 0):
            add("BLOCKER", "DUPLICATE_TIMESTAMP", f"{name} 有 {time['duplicate_count']} 行处于重复时间戳。")
        if time.get("delta_seconds", {}).get("gap_over_15min_count", 0):
            add("WARN", "TIME_GAP", f"{name} 存在 {time['delta_seconds']['gap_over_15min_count']} 个超过 15 分钟的时间间隔。")
        if time.get("missing_from_expected_15min_grid_count", 0):
            add("WARN", "GRID_MISSING", f"{name} 相对完整 15 分钟网格缺少 {time['missing_from_expected_15min_grid_count']} 个时间点。")
        if item["all_empty_columns"]:
            add("WARN", "ALL_EMPTY_COLUMN", f"{name} 存在全空字段：{', '.join(item['all_empty_columns'])}。")
        non_time_constant = [column for column in item["constant_columns"] if column != item["time"].get("column")]
        if non_time_constant:
            add("WARN", "CONSTANT_COLUMN", f"{name} 存在非时间常量字段：{', '.join(non_time_constant)}。")
        if item["missing_total"]:
            add("WARN", "MISSING_VALUE", f"{name} 有 {item['missing_total']} 个缺失单元格，必须在折内拟合处理规则。")
        if item["infinite_numeric_count"]:
            add("BLOCKER", "INFINITE_VALUE", f"{name} 存在 {item['infinite_numeric_count']} 个无穷数值。")
        if item["partition"] == "test" and item["test_target_presence"]:
            add("WARN", "CURRENT_ORIGIN_TARGET_OBSERVATION", f"{name} 含有当前时刻目标观测：{', '.join(item['test_target_presence'])}；按官方滚动答复仅允许在对应起点 t 使用，不得读取 t 之后的目标。")

    if cross["train_test_timestamp_overlap_count"]:
        add("WARN", "BOUNDARY_CONTEXT_OVERLAP", f"训练与测试时间戳有 {cross['train_test_timestamp_overlap_count']} 个交集；按官方滚动答复可作为起点 t 的边界上下文，但必须禁止使用 t 之后数据。")
    for duplicate in cross["train_test_exact_duplicate_rows"]:
        if duplicate["count"]:
            add("WARN", "BOUNDARY_EXACT_DUPLICATE", f"{duplicate['family']} 训练/测试在共同时间戳上有 {duplicate['count']} 行全部字段相同；将其视为边界上下文，不作为独立泛化证据。")

    status = "BLOCKED" if any(f["severity"] == "BLOCKER" for f in findings) else ("WARN" if findings else "PASS")
    report = {
        "audit_version": "v0",
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.as_posix(),
        "overall_status": status,
        "files": csv_results + workbook_results,
        "workbooks": workbook_results,
        "cross_table": cross,
        "findings": findings,
        "method": {
            "csv_encodings_tried": list(CSV_ENCODINGS),
            "expected_interval_seconds": 900,
            "targets_for_leakage_check": list(TARGET_NAMES),
            "source_mutation": False,
        },
    }
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    (out / "audit_report.md").write_text(render_report(report), encoding="utf-8")
    summary = pd.DataFrame(
        [
            {
                "file": item["file"],
                "partition": item.get("partition", "xlsx"),
                "rows": item.get("rows"),
                "columns": item.get("columns"),
                "missing_total": item.get("missing_total"),
                "duplicate_row_count": item.get("duplicate_row_count"),
                "time_column": item.get("time", {}).get("column"),
                "time_min": item.get("time", {}).get("min"),
                "time_max": item.get("time", {}).get("max"),
                "sha256": item["sha256"],
            }
            for item in csv_results
        ]
    )
    summary.to_csv(out / "csv_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"status": status, "out": out.as_posix(), "findings": findings}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
