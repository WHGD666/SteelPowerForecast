# IronFlow

IronFlow 是 AIC AI+钢铁产业命题赛的项目根目录，任务方向为煤气发电量预测与发电优化。

## 当前阶段

当前已完成 Phase 4 Baseline 建立与第一份初赛探针提交：只读审计、因果清洗 v1、预测信息集、滚动切分、短周期输出契约、三组基线比较和提交包校验均已形成可复现记录。尚未开始受控调参、深度模型或调度优化，内部封存留出集仍未使用。

当前原则：

- 原始比赛数据仅保存在本地，不上传到公开 GitHub。
- 预测必须遵守时间因果性，只使用预测起点及之前可获得的信息。
- 在任务合同、字段口径、时间切分和指标实现冻结前，不比较模型得分。
- 失败或被拒绝的正式实验也必须保留记录，不能覆盖历史结果。

## 任务摘要

- 预测目标：`generator_1`、`generator_all`。
- 数据粒度：当前文件为 15 分钟。
- 初赛短周期：未来 2 小时，即 8 个步长。
- 复赛长周期：未来 24 小时，即 96 个步长。
- 预测指标：官方定义的 1-MAPE，零值处理和聚合规则待评分细则确认。
- 半决赛：在预测资源边界基础上，增加气柜、机组和电价约束下的发电优化。

## 目录说明

```text
.
├── configs/       任务与实验协议草案
├── docs/          赛题、数据、业务逻辑和环境说明
├── experiments/   实验运行目录（本地生成，不入库）
├── manifests/     数据指纹和协议指纹（不保存原始数据）
├── outputs/       预测、指标、模型和报告（本地生成，不入库）
├── src/           后续实现代码
├── data/          本地比赛训练数据，已由 Git 忽略
└── test/          本地比赛测试数据，已由 Git 忽略
```

## 阶段门禁

1. 项目初始化：目录、数据策略和文档就绪。
2. 数据审计：字段、时间、缺失、重复、异常和泄漏路径可解释。
3. 协议冻结：目标、允许字段、滚动切分、指标和输出格式固定。
4. Baseline：持久性、周期基线和树模型在相同时间折上比较。
5. 受控优化：一次只改变一个主要因素，保留完整运行记录。
6. 候选冻结与封存评估：最终候选确定后才使用封闭留出集。
7. 预测与优化联合：形成可复现的离线提交包。

## 当前审计结论

审计结果为 **WARN / 协议 v1 已冻结**：训练/测试在 `2025-05-01 00:00:00` 存在四表边界重复，测试 `Pre_test_load.csv` 含当前时刻目标观测。根据官方答复，起点 `t` 的已提供字段（包括当前实际发电量）允许使用，但 `t` 之后的目标和观测禁止使用。详见 [数据审计报告](docs/data_audit_report_v0.md)、[质量审计](docs/data_quality_audit_v1.md) 和 [任务合同 v1](docs/task_contract_v1.md)。

契约校验命令：

```powershell
D:\anaconda\envs\vocs\python.exe src/validate_contract.py --root .
```

数据准备命令：

```powershell
D:\anaconda\envs\vocs\python.exe src/prepare_dataset.py --root . --config configs/cleaning_v1.yaml
D:\anaconda\envs\vocs\python.exe -m unittest tests/test_preparation_contract.py -v
```

派生数据仅保存在本地 `outputs/prepared_v1/`，不会覆盖或上传原始数据。清洗规则见 [cleaning_policy_v1.md](docs/cleaning_policy_v1.md)，执行证据见 [preparation_report_v1.md](docs/preparation_report_v1.md)。

## Baseline 与官方探针记录

当前里程碑结果见 [Baseline v1 结果记录](docs/baseline_result_v1.md)。机器可读的唯一来源为 [内部开发结果注册表](manifests/baseline_results_v1.csv) 和 [官方提交注册表](manifests/official_submissions.csv)。官方总分与内部验证 `1-MAPE` 的定义不同，不做直接等同。

复现基础流程：

```powershell
D:\anaconda\envs\vocs\python.exe src/build_splits.py
D:\anaconda\envs\vocs\python.exe -m pytest -q
D:\anaconda\envs\vocs\python.exe src/run_baseline.py --config configs/baseline_v1.yaml
D:\anaconda\envs\vocs\python.exe src/build_submission.py --config configs/submission_persistence_v1.yaml
D:\anaconda\envs\vocs\python.exe src/render_results_report.py
```
