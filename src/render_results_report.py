"""Render the tracked IronFlow baseline report from machine-readable registries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=Path("manifests/baseline_results_v1.csv"),
    )
    parser.add_argument(
        "--submission-registry",
        type=Path,
        default=Path("manifests/official_submissions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/baseline_result_v1.md"))
    args = parser.parse_args()
    root = args.root.resolve()
    baseline_path = root / args.baseline_registry
    submission_path = root / args.submission_registry
    output_path = root / args.output
    baselines = read_rows(baseline_path)
    submissions = read_rows(submission_path)
    if not baselines or not submissions:
        raise AssertionError("result registries must not be empty")

    latest = submissions[-1]
    source_run = latest["source_development_run_id"]
    source_baselines = [row for row in baselines if row["run_id"] == source_run]
    if not source_baselines:
        raise AssertionError(f"no development rows found for source run {source_run}")
    source_baselines.sort(key=lambda row: float(row["mape"]))
    selected = source_baselines[0]
    if selected["model"] != latest["selected_model"]:
        raise AssertionError("official submission model differs from best recorded development MAPE")

    lines = [
        "# IronFlow Baseline v1 结果记录",
        "",
        "> 本报告由 `src/render_results_report.py` 从两个受版本控制的 CSV 注册表生成。",
        "> 官方分数与内部验证指标属于不同口径，不得直接等同。",
        "",
        "## 1. 官方初赛探针提交",
        "",
        "| 项目 | 记录 |",
        "| --- | --- |",
        f"| 平台队伍编号 | `{latest['platform_team_id']}` |",
        f"| 队名 | {latest['team_name']} |",
        f"| 提交时间 | {latest['submitted_at_local']} ({latest['timezone']}) |",
        f"| 出分时间 | {latest['score_published_at_local']} ({latest['timezone']}) |",
        f"| 官方总分 | **{float(latest['official_score']):.4f}** |",
        f"| 当时排名快照 | **{latest['leaderboard_rank_snapshot']}** |",
        f"| 方案 | `{latest['submission_version']}` / `{latest['selected_model']}` |",
        f"| 来源开发 run | `{source_run}` |",
        f"| 来源代码提交 | `{latest['source_code_commit']}` |",
        f"| 本地提交包 | `{latest['local_artifact_filename']}` |",
        f"| 提交包 SHA256 | `{latest['local_artifact_sha256']}` |",
        "",
        "该分数是比赛平台返回的初赛隐藏测试综合总分。初赛由数据质量和短周期预测两部分组成，平台未返回分项，因此不能由 83.5246 反推出隐藏测试 MAPE。排名 74 仅表示出分时刻的排行榜快照，后续会随其他队伍提交变化。",
        "",
        "## 2. 内部开发基线",
        "",
        "固定切分包含 3 个滚动开发折、576 个预测起点；每个模型汇总 9,216 个目标-步长预测。封存留出集未使用。",
        "",
        "| 模型 | MAPE | 1-MAPE | MAE | RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in source_baselines:
        lines.append(
            "| "
            + row["model"]
            + f" | {float(row['mape']):.6f} | {float(row['score_1_mape']):.6f}"
            + f" | {float(row['mae']):.6f} | {float(row['rmse']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"按预先冻结的选择规则，`{selected['model']}` 以最低 pooled development MAPE 入选首份探针提交。LightGBM 的 MAE/RMSE 较低，但主指标 MAPE 略差，因此本轮没有用 LightGBM 生成提交。",
            "",
            "## 3. 可复现性与边界",
            "",
            f"- 开发 run：`{source_run}`。",
            f"- 切分指纹：`{selected['split_sha256']}`。",
            f"- 准备后训练表指纹：`{selected['prepared_train_sha256']}`。",
            f"- 提交文件规模：{latest['input_rows']} 行 input、{latest['input_columns']} 列输入、{latest['prediction_columns']} 列短周期结果。",
            "- 提交包只包含 `input.csv` 与 `s_result.csv`，原始赛事数据和本地派生产物不进入公开 Git。",
            "- 官方成绩证据来自用户提供的平台排行榜记录；仓库不保存登录态、截图或任何账号凭据。",
            "- 内部封存留出集仍保持未消费状态；本轮只完成 baseline 建立和外部初赛探针提交。",
            "",
            "## 4. 记录来源",
            "",
            "- `manifests/baseline_results_v1.csv`：内部开发指标的受控注册表。",
            "- `manifests/official_submissions.csv`：官方提交与排行榜快照注册表。",
            "- 生成命令：`python src/render_results_report.py`。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
