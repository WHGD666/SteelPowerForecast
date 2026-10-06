# 2026-10-01 学弟 v16 候选交接快照

本目录保存 2026-10-01 收到的复赛复现材料，只作为不可修改的外部证据和源码快照。目录名中的 `v16_candidate` 表示交接手册提出的候选，不代表当前约 61 分 ZIP 已被证明来自 v16。

## 证据身份

| 文件 | SHA-256 |
|---|---|
| `evidence/AI复现手册_60分保底.md` | `e03d304c1def1bca8bcf37cf1f97d5e519491cae4f955ba914e3eb3ccbcc421c` |
| `evidence/复赛代码_给学长.zip` | `cc9067dd1b7da565dbf3f17677b68f185c8baf70f618dca97a8109e92478d728` |
| `source_snapshot/src/build_submission_semi.py` | `91d62dac938d084b863b7f1c1f0bbdb32fb93669d0eafc481e84ba6ed15cac8f` |
| `source_snapshot/src/build_submission_semi_v15_sepwindow.py` | `77a513f13172a41fd693aeb8644b2ddde426086b487efd07422326c9f31be64c` |
| `source_snapshot/configs/submission_semi_composed.yaml` | `b2275abec284e5a23f5e014f704c3242c641a876a2b4f42b79123c708b2f9b57` |
| `source_snapshot/configs/cleaning_v2.yaml` | `9a782cfe4fb8afbe6706825787e16c39d96ba6f18afbefea263e06d8eeb3b0f4` |
| `source_snapshot/requirements.txt` | `1a67e8e7d370c230640fcc56ec312a2d4de43d0e54b46716c3ee48b8e4d78213` |

## 相对 2026-09-29 快照的变化

逐文件哈希比较结果：34 个文件完全相同；新增 4 个文件；修改 1 个文件。

新增：

- `AI复现手册_60分保底.md`
- `src/build_submission_semi_v15_sepwindow.py`
- `src/build_submission_semi_variants.py`
- `src/ramp_lag_diagnosis.py`

唯一修改的旧文件是 `src/build_submission_semi.py`，变化仅为增加 `--train-start` 参数并在构建模式中过滤训练起始日期。

## 使用规则

- 不在 `source_snapshot/` 内直接运行、调试或修改。
- 使用 `src.round2_v3.prepare_external_v16_reproduction` 复制到本地忽略的工作区。
- 第一轮复现必须保持原标签、特征、参数和后处理逻辑不变。
- 复现验收以冻结提交包内两个 CSV 的 SHA-256 为准；ZIP 容器哈希可能因成员时间戳而变化。

最小复现已排除“61 分包等于交接源码原样 v15/v16”。完整证据和后续输入要求见 `docs/14_EXTERNAL_61_REPRODUCTION_CONTRACT.md`。
