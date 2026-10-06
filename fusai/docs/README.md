# docs 索引

复赛全部文档的导航索引。**写实验报告请从"收官总结"三份开始**，再按需下钻到编号
报告；治理与协议类文件定义了整个项目的行为边界。所有结论性表述均可追溯到
`experiments/round2_v3/registry.csv`（32 个本地 run）、
`submissions/round2_v3/registry.csv`（10 次提交）与 `docs/09_DECISION_LOG.md`
（D-001～D-080）。

## 收官总结（报告写作的事实基准）

| 文档 | 内容 |
|---|---|
| `70_COMPREHENSIVE_EXPERIMENT_LEDGER.md` | 全量实验总账：平台提交 10 次、本地 run 32 个、15 条关闭路线的一页式总表 |
| `71_TECHNICAL_SOLUTION_SUMMARY.md` | 技术方案总结：问题→数据→信息边界→特征→验证→模型→门禁→工程→结论 |
| `72_FINAL_SCORE_RECORD.md` | 最终成绩记录：初赛 83.52、复赛时间线、v29b 63.4621 分项与归因 |
| `73_V29B_BEST_SUBMISSION_DOSSIER.md` | 最佳版本专门档案：v29b 成绩单、配方四区块、血统全史、哈希证据链、平台消融全景与缺口归因 |

## 治理与协议（00-09）

| 文档 | 内容 |
|---|---|
| `00_PROJECT_STATUS.md` | 收官状态、阶段门禁终版（G0-G10）、资产与边界 |
| `01_OFFICIAL_TASK_CONTRACT.md` | 官方任务合同：目标定义、评分、门槛 |
| `02_DATA_GOVERNANCE_AND_AUDIT.md` | 数据治理规范：只读、指纹、分区 |
| `03_VALIDATION_PROTOCOL.md` | 验证协议：信息边界、切分、指标 |
| `04_EXPERIMENT_PROTOCOL.md` | 实验管理协议：角色、单因素、run_id、失败留存 |
| `05_HIGH_SCORE_STRATEGY.md` | 冲分策略（历史文件） |
| `06_RISK_REGISTER.md` | 风险登记册 |
| `07_SUBMISSION_PROTOCOL.md` | 提交协议：单区块替换、哈希、噪声认知 |
| `08_LEGACY_EVIDENCE.md` | 学弟历史证据索引 |
| `09_DECISION_LOG.md` | 决策日志 D-001～D-080（项目决策的权威流水） |

## 审计、基线与外部基准（10-20）

| 文档 | 内容 |
|---|---|
| `10_DATA_AUDIT_REPORT.md` | 原始数据审计：结构、缺失、异常、漂移 |
| `11_BASELINE_V1_REPORT.md` | v3 baseline 报告（平台校准前） |
| `12_PLATFORM_CALIBRATION_POSTMORTEM.md` | 首次平台校准 48.92 复盘：本地 OOF 为何失真 |
| `13_EXTERNAL_61_CANDIDATE_REVIEW.md` | 61 分外部包审验 |
| `14_EXTERNAL_61_REPRODUCTION_CONTRACT.md` | 61 分复现合同 |
| `15_LATEST_OPTIMAL_BUNDLE_REVIEW.md` | 最优整合包（v28 血统）审验 |
| `16_64_65_PLATFORM_LADDER.md` | 64/65 分平台阶梯规划（历史） |
| `17_TOP3_AND_50_TRIALS_REVIEW.md` | 最高三包 + 近 50 版试错全档案审阅（v29b 63.4621 的权威审验） |
| `18_BREAK_65_RESEARCH_PLAN.md` | 冲 65 研究计划（历史） |
| `19_STEP_EVENT_AUDIT_REPORT.md` | 台阶事件与因果前兆审计（13 个 episode） |
| `20_EVENT_SIGNAL_ABLATION_REPORT.md` | process-only 特征组消融报告 |

## 实验协议与报告（21-60，按族分组）

动态燃料族：`21`/`22` 协议与报告（21 天版）、`23` 平台复盘（62.81）、`24`/`25` 异常
加权、`26`/`27` onset hazard、`28`/`29` v28 标签对齐、`30`/`31` short-g1 恢复字段、
`52`/`53` short 动态燃料、`54`/`55` ±5MW 限幅、`56`/`57` 燃料趋势、`58`/`59` 专家
门控、`60` short 限幅平台探针（62.72）。

共享 horizon 与指标族：`32`-`35` long-g1 shared v1/v2、`36`/`37` long-gall shared +
平台探针（63.45）、`38` long-g1 平台探针（63.10）、`39`/`40` 因果重叠一致性、
`41`/`42` test-similarity 校准（AUC 0.95 但方向全错）、`43`/`44` MAPE 目标、
`45`/`46` MAPE fuel、`47` 电价审计、`48`/`49` 节假日审计、`50`/`51` 节假日探针
（63.30）。

每个编号对为"预登记协议 → 结果报告"；平台探针报告均含 ZIP/CSV 哈希与归因边界。
单次实验的机器记录见两个 registry.csv。

## 收官阶段实验（61-69）

| 文档 | 内容 |
|---|---|
| `61`/`62` | 历史相似工况轨迹：协议 / 报告（+0.058，拒绝；其 OOF 成为后续控制组） |
| `63`/`64` | 24h 窗直接 GRU：协议 / 报告（独立 -1.91，拒绝） |
| `65`/`66` | holder_2 正向边界诊断：协议 / 报告（正向 4/4 判负，发现反向信号） |
| `67`/`68` | 反向信号新鲜块确认：协议 / 报告（残差门禁判负、机制 8/8 成立，路线永久关闭） |
| `69_OCTOBER_REGIME_BRIEFING.md` | 十月工况情报简报（高气柜制度、BF3 硬外推；可对外） |

## 项目外部结构

- `../experiments/round2_v3/`：EXPERIMENT_LOG.md（55 条人读日志）+ registry.csv（32 run）
- `../submissions/round2_v3/`：SUBMISSION_LOG.md（哈希级提交日志）+ registry.csv
- `../configs/round2_v3/`：baseline / experiments / diagnostics 全部冻结配置
- `../src/round2_v3/`：runner 与纯函数库；`../tests/round2_v3/`：118 项 pytest
- `../outputs/`：全部 run 工件（manifest 含输入/源码 SHA-256 指纹，不可覆盖）
- `../references/`、`../legacy/`：官方资料与学弟代码只读快照
