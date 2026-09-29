"""Create the lightweight, read-only round-2 raw-data audit.

The command reads official copies under ``data/raw`` and writes only small
manifests/tables plus a Markdown report.  It never changes source CSV files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.io_utils import read_csv_strict, sha256, write_json


FILE_PAIRS = {
    "gas": ("Semi_gas.csv", "Semi_test_gas.csv"),
    "gas_holder": ("Semi_gas_holder.csv", "Semi_test_gas_holder.csv"),
    "gas_user": ("Semi_gas_user.csv", "Semi_test_gas_user.csv"),
    "load": ("Semi_load.csv", "Semi_test_load.csv"),
}
TARGETS = ("generator_1", "generator_all")


def safe_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def profile_file(path: Path, relative_path: str, partition: str, family: str) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    frame, encoding = read_csv_strict(path)
    if "datetime" not in frame:
        raise AssertionError(f"{relative_path}: missing datetime")
    times = pd.to_datetime(frame["datetime"], errors="coerce")
    valid_times = times.dropna()
    if valid_times.empty:
        raise AssertionError(f"{relative_path}: no valid datetime")
    grid = pd.date_range(valid_times.min(), valid_times.max(), freq="1min")
    missing_timestamps = grid.difference(pd.DatetimeIndex(valid_times))
    deltas = valid_times.diff().dropna()
    full_duplicates = int(frame.duplicated(keep=False).sum())
    datetime_duplicates = int(valid_times.duplicated(keep=False).sum())

    summary = {
        "partition": partition,
        "family": family,
        "file": relative_path,
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
        "encoding": encoding,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "datetime_min": valid_times.min().isoformat(),
        "datetime_max": valid_times.max().isoformat(),
        "invalid_datetime_count": int(times.isna().sum()),
        "datetime_duplicate_rows": datetime_duplicates,
        "full_duplicate_rows": full_duplicates,
        "expected_minute_grid_rows": int(len(grid)),
        "missing_timestamp_count": int(len(missing_timestamps)),
        "missing_timestamps": "|".join(ts.isoformat() for ts in missing_timestamps),
        "non_one_minute_deltas": int((deltas != pd.Timedelta(minutes=1)).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
    }

    column_rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if column == "datetime":
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        non_null = int(numeric.notna().sum())
        row: dict[str, Any] = {
            "partition": partition,
            "family": family,
            "file": relative_path,
            "column": str(column),
            "source_dtype": str(frame[column].dtype),
            "rows": int(len(frame)),
            "non_null": non_null,
            "missing": int(numeric.isna().sum()),
            "missing_rate": float(numeric.isna().mean()),
            "infinite_count": int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()),
            "unique_non_null": int(numeric.nunique(dropna=True)),
            "all_empty": bool(non_null == 0),
            "constant_non_null": bool(non_null > 0 and numeric.nunique(dropna=True) <= 1),
            "first_non_null_time": None,
            "last_non_null_time": None,
            "zero_count": int((finite == 0).sum()),
            "negative_count": int((finite < 0).sum()),
            "abs_over_1e9_count": int((finite.abs() > 1e9).sum()),
            "min": None,
            "p01": None,
            "q1": None,
            "median": None,
            "q3": None,
            "p99": None,
            "max": None,
        }
        if non_null:
            non_null_times = valid_times[numeric.notna() & times.notna()]
            if not non_null_times.empty:
                row["first_non_null_time"] = non_null_times.min().isoformat()
                row["last_non_null_time"] = non_null_times.max().isoformat()
        if not finite.empty:
            quantiles = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            row.update(
                {
                    "min": safe_float(finite.min()),
                    "p01": safe_float(quantiles.loc[0.01]),
                    "q1": safe_float(quantiles.loc[0.25]),
                    "median": safe_float(quantiles.loc[0.5]),
                    "q3": safe_float(quantiles.loc[0.75]),
                    "p99": safe_float(quantiles.loc[0.99]),
                    "max": safe_float(finite.max()),
                }
            )
        column_rows.append(row)
    frame = frame.copy()
    frame["datetime"] = times
    return summary, column_rows, frame


def distribution_shift_rows(
    family: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, Any]]:
    recent_start = train["datetime"].max() - pd.Timedelta(days=14)
    recent = train[train["datetime"] >= recent_start]
    rows: list[dict[str, Any]] = []
    for column in sorted((set(train.columns) & set(test.columns)) - {"datetime"}):
        x = pd.to_numeric(recent[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        y = pd.to_numeric(test[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 100 or len(y) < 100:
            continue
        mean_train = float(x.mean())
        mean_test = float(y.mean())
        std_train = float(x.std())
        z_shift = abs(mean_test - mean_train) / std_train if std_train > 0 else np.nan
        rows.append(
            {
                "family": family,
                "column": column,
                "recent_train_start": recent_start.isoformat(),
                "recent_train_mean": mean_train,
                "recent_train_std": std_train,
                "test_mean": mean_test,
                "absolute_mean_shift": abs(mean_test - mean_train),
                "standardized_mean_shift": safe_float(z_shift) if np.isfinite(z_shift) else None,
            }
        )
    return rows


def markdown_report(
    file_summary: pd.DataFrame,
    columns: pd.DataFrame,
    shifts: pd.DataFrame,
    schema_checks: list[dict[str, Any]],
) -> str:
    lines = [
        "# 复赛原始数据审计报告 v3",
        "",
        "本报告由只读审计脚本从 `data/raw/` 生成。它不修改原始数据，也不代表官方得分。",
        "",
        "## 1. 文件概况",
        "",
        "| 分区 | 文件族 | 行数 | 列数 | 时间范围 | 缺失时间戳 | 重复时间戳 | 缺失单元格 |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in file_summary[file_summary["partition"].isin(["train", "test"])].iterrows():
        lines.append(
            f"| {row['partition']} | {row['family']} | {row['rows']} | {row['columns']} | "
            f"{row['datetime_min']} 至 {row['datetime_max']} | {row['missing_timestamp_count']} | "
            f"{row['datetime_duplicate_rows']} | {row['missing_cells']} |"
        )

    lines.extend(["", "## 2. Schema 一致性", "", "| 文件族 | 训练/测试列顺序一致 | 训练列数 | 测试列数 |", "|---|---|---:|---:|"])
    for item in schema_checks:
        lines.append(
            f"| {item['family']} | {'是' if item['columns_equal'] else '否'} | "
            f"{item['train_columns']} | {item['test_columns']} |"
        )

    lines.extend(["", "## 3. 时间缺口", ""])
    gap_rows = file_summary[file_summary["missing_timestamp_count"] > 0]
    if gap_rows.empty:
        lines.append("未发现分钟时间戳缺口。")
    else:
        for _, row in gap_rows.iterrows():
            lines.append(f"- `{row['file']}`：{row['missing_timestamps']}")

    lines.extend(["", "## 4. 全空与部分投运字段", ""])
    all_empty = columns[columns["all_empty"]]
    partial = columns[(columns["partition"] == "train") & (~columns["all_empty"]) & (columns["missing_rate"] >= 0.05)]
    lines.append("全空字段：")
    for _, row in all_empty.iterrows():
        lines.append(f"- `{row['file']}` / `{row['column']}`")
    lines.append("")
    lines.append("训练缺失率不低于 5% 但并非全空的字段：")
    for _, row in partial.iterrows():
        lines.append(
            f"- `{row['column']}`：缺失率 {row['missing_rate']:.2%}，首次有效 {row['first_non_null_time']}"
        )

    lines.extend(["", "## 5. 明显数值异常候选", ""])
    extreme = columns[(columns["abs_over_1e9_count"] > 0) | (columns["negative_count"] > 0)]
    if extreme.empty:
        lines.append("未发现负值或绝对值超过 1e9 的字段。")
    else:
        lines.extend(["| 分区 | 字段 | 负值数 | 绝对值>1e9 | 最小值 | 最大值 |", "|---|---|---:|---:|---:|---:|"])
        for _, row in extreme.iterrows():
            lines.append(
                f"| {row['partition']} | `{row['column']}` | {row['negative_count']} | "
                f"{row['abs_over_1e9_count']} | {row['min']} | {row['max']} |"
            )

    target_rows = columns[(columns["partition"] == "test") & (columns["column"].isin(TARGETS))]
    lines.extend(["", "## 6. 测试目标可用性", ""])
    for _, row in target_rows.iterrows():
        lines.append(f"- `{row['column']}`：非空 {row['non_null']}，缺失 {row['missing']}。")
    lines.append("")
    lines.append("结论：测试期目标历史不可直接使用，正式特征合同必须能在两个目标全空时运行。")

    lines.extend(["", "## 7. 训练末 14 天与测试期均值漂移", ""])
    if shifts.empty:
        lines.append("没有可计算的连续数值字段。")
    else:
        top = shifts.sort_values("standardized_mean_shift", ascending=False).head(15)
        lines.extend(["| 字段 | 标准化均值偏移 | 近期训练均值 | 测试均值 |", "|---|---:|---:|---:|"])
        for _, row in top.iterrows():
            lines.append(
                f"| `{row['column']}` | {row['standardized_mean_shift']:.3f} | "
                f"{row['recent_train_mean']:.3f} | {row['test_mean']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## 8. Gate 结论",
            "",
            "- 原始数据指纹、文件结构和字段统计已生成。",
            "- 原始 CSV 未被修改。",
            "- 时间缺口、全空列、部分投运字段和明显异常仍需由 v3 清洗合同处理。",
            "- 本报告完成数据审计证据，不代表清洗和标签 Gate 已通过。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    raw_root = root / "data" / "raw"
    manifest_dir = root / "manifests" / "round2_v3"
    docs_dir = root / "docs"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    schema_checks: list[dict[str, Any]] = []
    shifts: list[dict[str, Any]] = []

    for family, (train_name, test_name) in FILE_PAIRS.items():
        for partition, filename in (("train", train_name), ("test", test_name)):
            path = raw_root / partition / filename
            if not path.exists():
                raise FileNotFoundError(path)
            relative = path.relative_to(root).as_posix()
            summary, profiles, frame = profile_file(path, relative, partition, family)
            file_rows.append(summary)
            column_rows.extend(profiles)
            frames[(partition, family)] = frame
        train = frames[("train", family)]
        test = frames[("test", family)]
        schema_checks.append(
            {
                "family": family,
                "columns_equal": list(train.columns) == list(test.columns),
                "train_columns": len(train.columns),
                "test_columns": len(test.columns),
                "train_column_names": list(train.columns),
                "test_column_names": list(test.columns),
            }
        )
        shifts.extend(distribution_shift_rows(family, train, test))

    fingerprint_rows = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".zip"}:
            fingerprint_rows.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256(path),
                }
            )

    file_summary = pd.DataFrame(file_rows).sort_values(["partition", "family"])
    column_summary = pd.DataFrame(column_rows).sort_values(["partition", "family", "column"])
    shift_summary = pd.DataFrame(shifts).sort_values(
        "standardized_mean_shift", ascending=False, na_position="last"
    )

    file_summary.to_csv(manifest_dir / "raw_file_summary.csv", index=False, encoding="utf-8")
    column_summary.to_csv(manifest_dir / "column_profile.csv", index=False, encoding="utf-8")
    shift_summary.to_csv(manifest_dir / "distribution_shift.csv", index=False, encoding="utf-8")
    write_json(
        manifest_dir / "data_fingerprints.json",
        {
            "manifest_version": "round2_v3_raw_1",
            "raw_data_read_only": True,
            "files": fingerprint_rows,
        },
    )
    audit = {
        "audit_version": "round2_v3_raw_1",
        "source_mutation": False,
        "files": file_rows,
        "schema_checks": schema_checks,
        "test_target_non_null": {
            target: int(frames[("test", "load")][target].notna().sum()) for target in TARGETS
        },
        "fingerprint_file_count": len(fingerprint_rows),
        "gate_pass": False,
        "gate_note": "raw audit complete; cleaning, label and split gates remain open",
    }
    write_json(manifest_dir / "raw_data_audit.json", audit)
    report = markdown_report(file_summary, column_summary, shift_summary, schema_checks)
    (docs_dir / "10_DATA_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    print(
        f"PASS files={len(file_summary)} columns={len(column_summary)} "
        f"fingerprints={len(fingerprint_rows)} report=docs/10_DATA_AUDIT_REPORT.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
