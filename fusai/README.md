# IronFlow 复赛 v3 工作区

本目录是 IronFlow 复赛的全新受控实验主线。它与初赛代码、学弟复赛旧代码和历史输出隔离，但仍位于同一个 `SteelPowerForecast` 大仓库中。

当前目标不是马上堆模型，而是先建立一条能够解释、比较、复现和审计的冲分链路。官方历史最高分 **52.4085** 作为外部历史基准保留；在完整复现链补齐以前，它不视为本目录中的可复现基线。

## 当前状态

- 项目阶段：**Stage 4 — 平台校准后受控改进与历史机制复现**。
- 当前日期：2026-09-29。
- 官方门槛：短期准确率不低于 90%，长期准确率不低于 85%，总分不低于 60。
- 项目目标：形成合规、可复现、稳定超过 60 分并冲击 65+ 的方案。
- 官方数据、规则、答疑和学弟历史资料已完成隔离快照与哈希核验。
- 原始数据只读审计、正式标签、验证切分和本地指标合同已完成。
- v3 baseline、层次目标实验和首次平台校准均已完成。
- 首次 v3 平台分为 **48.9241**，低于历史 legacy v9 的 **52.4085**；该结果已拒绝为高分候选，只保留为可复现校准锚点。
- 平台仅返回总分，不能推断短期、长期或分目标表现。

## 不可违反的原则

1. 官方原始数据只读，不在原文件上填补、修正、覆盖或追加字段。
2. 预测起点为 `t` 时，只允许使用 `t` 及以前已经观测到的信息。
3. 测试期 `t` 之后的煤气发生量、用户耗气、机组耗气、气柜状态均为禁止未来信息。
4. t+15 的目标按 `[t, t+14分钟]` 共 15 条一分钟负荷的均值理解；后续步长依次顺延。
5. 短周期和长周期可以使用不同模型，必须分别验证和记录。
6. 所有正式实验使用不可覆盖的 `run_id`；失败实验同样保留。
7. 每次正式比较只能在相同协议、相同标签、相同切分和相同指标下进行。
8. 平台分数是外部测试证据，不是训练标签；禁止通过平台反推测试真值。
9. Oracle、真实未来路径和泄漏诊断只能标记为 `diagnostic`，不得包装成可提交模型。
10. 提交包必须能由对应代码、配置和模型重新生成，并保存哈希。

## 目录说明

| 目录 | 用途 | 是否允许人工覆盖 |
|---|---|---|
| `references/` | 官方规则、答疑和来源说明 | 否 |
| `legacy/` | 学弟旧代码、旧实验和历史证据快照 | 否 |
| `data/raw/` | 官方原始数据，只读 | 否 |
| `data/prepared/` | 由固定配置生成的处理数据 | 不覆盖，按版本重建 |
| `configs/round2_v3/` | v3 协议、清洗、切分和实验配置 | 版本化修改 |
| `src/common/` | 经测试验证的公共组件 | 代码审查后修改 |
| `src/round2_v3/` | 复赛 v3 正式实现 | 代码审查后修改 |
| `tests/round2_v3/` | 标签、信息边界、切分和提交合同测试 | 随协议维护 |
| `manifests/round2_v3/` | 数据指纹、切分清单和协议指纹 | 不覆盖，生成新版本 |
| `experiments/round2_v3/` | 实验登记簿和人类可读日志 | 只追加 |
| `outputs/` | 本地运行产物 | 每个 run 独立、不可覆盖 |
| `submissions/round2_v3/` | 提交登记和冻结包 | 每个 submission 独立 |
| `templates/` | run 和 submission manifest 模板 | 版本化维护 |

## 文档阅读顺序

1. `docs/00_PROJECT_STATUS.md`
2. `docs/01_OFFICIAL_TASK_CONTRACT.md`
3. `docs/02_DATA_GOVERNANCE_AND_AUDIT.md`
4. `docs/03_VALIDATION_PROTOCOL.md`
5. `docs/04_EXPERIMENT_PROTOCOL.md`
6. `docs/05_HIGH_SCORE_STRATEGY.md`
7. `docs/06_RISK_REGISTER.md`
8. `docs/07_SUBMISSION_PROTOCOL.md`
9. `docs/08_LEGACY_EVIDENCE.md`
10. `docs/09_DECISION_LOG.md`
11. `docs/10_DATA_AUDIT_REPORT.md`
12. `docs/11_BASELINE_V1_REPORT.md`
13. `docs/12_PLATFORM_CALIBRATION_POSTMORTEM.md`

## 训练前门禁（已完成）

在开始模型训练前，以下事项必须全部完成：

- [x] 官方资料和数据复制到对应目录并完成 SHA-256 指纹。
- [x] 原始数据结构、缺失、异常、部分投运字段和分布漂移审计完成。
- [x] 标签区间映射通过边界样例测试，并生成正式 manifest。
- [x] 允许字段、禁止字段和诊断字段清单冻结。
- [x] 时间切分清单生成并通过不重叠检查。
- [x] 验证模拟能够证明每个起点只使用 `t` 及以前的信息。
- [x] 评价指标实现通过人工小样例测试。
- [x] 提交列名、顺序、行数和时间连续性合同冻结。

门禁未通过前，不得把任何模型结果称为 v3 baseline。

## 下一步

当前不重复提交 baseline 微调版本，优先恢复可复现的 legacy v9 等价锚点并识别其有效机制：

1. 对齐旧版目标历史反馈、递归路径、煤气平衡、训练窗口和后处理。
2. 在冻结切分上逐项替换，重点观察 `generator_1`、晚九月和气柜投运折。
3. 对长周期 15/30/60 分钟 origin 抽样做同协议消融。
4. 候选通过新门禁后，平台提交只改变短文件或长文件中的一个，保证总分变化可归因。

完整教训、禁止推断和新提交门禁见 `docs/12_PLATFORM_CALIBRATION_POSTMORTEM.md`。

已准备好全量因果对齐命令，但由于会读写数十 MB 数据，按当前协作约定由人工在
`fusai/` 目录执行：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.prepare_causal_base
```

该命令不训练模型，不使用 GPU，不修改 raw 文件；它只生成新的版本目录和 manifest，
且如果目录已存在会拒绝覆盖。

因果 base 与 origin features 均已生成并验收。正式 baseline 命令为：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.preflight_baseline
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_baseline --config configs\round2_v3\baseline_v1.yaml
```

该命令会训练 5 折 × 2 周期 × 2 目标，共 20 个 LightGBM，并同步计算日历弱基线；
保存 OOF、逐折模型、指标、事件日志和不可覆盖的 run manifest。它是 CPU 密集任务，
因此由人工执行。

首个 baseline 已完成。当前受控层次目标实验命令：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.preflight_hierarchical
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_hierarchical --config configs\round2_v3\hierarchical_v1.yaml
```

该实验只训练 5 折 × 2 周期共 10 个 `large_units` 模型，`generator_all` 直接复用控制组 OOF；
不使用平台数据，不生成提交文件。

层次实验已因 OOF 退化而拒绝。首次平台校准包按以下顺序生成（已完成，除复现检查外不应重复提交）：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.build_test_features_with_history
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.build_submission_baseline --config configs\round2_v3\submission_baseline_v1.yaml
```

第一条命令确保 10 月首个起点的 lag/rolling 连续使用 9 月历史；第二条命令全量训练
4 个模型、生成 960 行短/长文件、执行物理与 schema 检查并打包 ZIP。
