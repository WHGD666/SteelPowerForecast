"""Statistical, read-only audit aligned with the prelim quality checks.

The organizer has not published the exact outlier thresholds.  This script
therefore reports transparent IQR and robust-MAD candidates without changing
data or claiming an official score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_audit import read_csv_strict, sha256


def profile_frame(frame: pd.DataFrame, relative_file: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if str(column).lower() in {"datetime", "timestamp", "time"}:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median = float(values.median())
        mad = float((values - median).abs().median())
        if mad > 0:
            robust_z = (values - median).abs() / (1.4826 * mad)
            robust_count = int((robust_z > 6).sum())
        else:
            robust_count = 0
        iqr_count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "file": relative_file,
                "column": str(column),
                "non_null": int(values.size),
                "missing": int(frame[column].isna().sum()),
                "zero_count": int((values == 0).sum()),
                "negative_count": int((values < 0).sum()),
                "min": float(values.min()),
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": float(values.max()),
                "iqr_lower": lower,
                "iqr_upper": upper,
                "iqr_outlier_count": iqr_count,
                "mad": mad,
                "robust_mad_outlier_count": robust_count,
                "candidate_outlier": bool(iqr_count or robust_count),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="IronFlow prelim data-quality audit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None, help="optional generated input.csv to check feat_ naming")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "outputs" / "data_quality_audit_v1").resolve()
    out.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(list((root / "data").glob("*.csv")) + list((root / "test").glob("*.csv")))
    profiles: list[dict[str, object]] = []
    file_summary: list[dict[str, object]] = []
    for path in csv_paths:
        frame, encoding = read_csv_strict(path)
        relative = path.relative_to(root).as_posix()
        profiles.extend(profile_frame(frame, relative))
        numeric_columns = [column for column in frame.columns if column != "datetime" and pd.api.types.is_numeric_dtype(pd.to_numeric(frame[column], errors="coerce"))]
        all_empty = [str(column) for column in frame.columns if frame[column].isna().all()]
        constant = [str(column) for column in frame.columns if column != "datetime" and frame[column].nunique(dropna=True) <= 1]
        file_summary.append(
            {
                "file": relative,
                "partition": "test" if path.parent.name == "test" else "train",
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "encoding": encoding,
                "sha256": sha256(path),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated(keep=False).sum()),
                "all_empty_columns": all_empty,
                "constant_non_time_columns": constant,
                "numeric_columns": len(numeric_columns),
            }
        )

    profile_frame_df = pd.DataFrame(profiles)
    profile_frame_df.to_csv(out / "column_quality_profile.csv", index=False, encoding="utf-8-sig")
    candidate_rows = profile_frame_df[profile_frame_df["candidate_outlier"]].copy() if not profile_frame_df.empty else profile_frame_df

    feat_check: dict[str, object] = {"checked": False, "bad_columns": [], "columns": []}
    if args.input is not None:
        input_frame = read_csv_strict(args.input.resolve())[0]
        extra = [str(column) for column in input_frame.columns if column != "datetime"]
        raw_columns = set().union(*[set(pd.read_csv(root / "data" / f"Pre_{family}.csv", nrows=0).columns) for family in ("gas", "gas_holder", "gas_user", "load")])
        bad = [column for column in extra if column not in raw_columns and not column.startswith("feat_")]
        feat_check = {"checked": True, "bad_columns": bad, "columns": list(input_frame.columns)}

    report = {
        "audit_version": "quality-v1",
        "official_score_claim": False,
        "method": {
            "missing": "count and rate by file/column",
            "duplicates": "exact full-row duplicates",
            "outliers": "1.5*IQR fence OR robust MAD z > 6; candidate only",
            "invalid_columns": "all-empty and non-time constant columns",
            "feature_prefix": "extra input columns must start with feat_",
        },
        "files": file_summary,
        "outlier_candidate_columns": candidate_rows[["file", "column", "iqr_outlier_count", "robust_mad_outlier_count"]].to_dict("records") if not candidate_rows.empty else [],
        "outlier_candidate_count": int(len(candidate_rows)),
        "feat_prefix_check": feat_check,
        "source_mutation": False,
    }
    (out / "quality_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# IronFlow 数据质量审计 v1",
        "",
        "本报告不代表官方得分。组委会未公开异常值阈值，因此仅报告透明的 IQR/MAD 候选，不自动删除任何样本。",
        "",
        "## 文件摘要",
        "",
        "| 文件 | 分区 | 缺失单元格 | 重复整行 | 全空列 | 常量非时间列 | SHA-256 前 16 位 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for item in file_summary:
        lines.append(f"| `{item['file']}` | {item['partition']} | {item['missing_cells']} | {item['duplicate_rows']} | {', '.join(item['all_empty_columns']) or '-'} | {', '.join(item['constant_non_time_columns']) or '-'} | `{str(item['sha256'])[:16]}` |")
    lines.extend(["", "## 统计异常候选", "", f"IQR/MAD 联合规则标记列数：**{len(candidate_rows)}**。这些列需要结合工业工况、停机状态和时间段复核，不能直接按异常值删除。", ""])
    if not candidate_rows.empty:
        lines.append("| 文件 | 字段 | IQR 候选数 | MAD 候选数 |")
        lines.append("|---|---|---:|---:|")
        for _, row in candidate_rows.iterrows():
            lines.append(f"| `{row['file']}` | `{row['column']}` | {int(row['iqr_outlier_count'])} | {int(row['robust_mad_outlier_count'])} |")
    else:
        lines.append("没有列被透明规则标记为候选异常。")
    lines.extend(["", "## 放行解释", "", "重复整行、负值和无穷值需要硬性拦截；缺失、全空列、常量列和统计极值需要形成可复现处理决策。最终特征工程字段必须使用 `feat_` 前缀。", ""])
    (out / "quality_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": out.as_posix(), "candidate_outlier_columns": len(candidate_rows), "official_score_claim": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
