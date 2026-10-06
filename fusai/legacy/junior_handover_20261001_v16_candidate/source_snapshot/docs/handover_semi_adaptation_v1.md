# IronFlow 复赛适配工作交接文档 v1

> 文档目的：将 IronFlow 项目从"初赛阶段"交接到"复赛阶段"，供接手 AI/工程师在**不破坏既有冻结协议**的前提下继续工作。
> 文档读者：接手本项目的 AI 代理或工程师。
> 文档状态：基于 2026-09-25 的项目实际扫描结果整理。
> 首要要求：**开始任何编码前，先通读本文档第 3 节"冻结纪律"，再读取第 6 节列出的核心参照文件。**

---

## 1. 项目概览

| 项目 | 内容 |
| --- | --- |
| 项目代号 | IronFlow |
| 赛事 | AIC AI+钢铁产业命题赛 |
| 任务方向 | 煤气发电量预测与发电优化 |
| 平台队伍编号 | `AIC-2026-84933645` |
| 队名 | 这模型调的有力量 |
| 项目根目录 | `D:\zuoye\sai\SteelPowerForecast` |
| 预测目标 | `generator_1`、`generator_all` |
| 数据粒度 | 15 分钟 |
| 主指标 | 官方 `1-MAPE`（零值/缺失/聚合规则官方未完全公开） |

### 1.1 赛事三阶段结构

1. **初赛（已完成）**：短周期预测，从起点 `t` 预测 `t+15` 至 `t+120`，共 8 步（2 小时）。
2. **复赛（当前进入）**：长周期预测，未来 24 小时，即 96 个步长。
3. **半决赛（未来）**：在预测资源边界基础上，增加气柜、机组和电价约束下的发电调度优化。

依据：`README.md:20-23`。

---

## 2. 当前进度与状态

### 2.1 已完成里程碑

- 数据只读审计、因果清洗 v1、预测信息集、滚动切分、短周期输出契约已形成可复现记录。
- 基线建立：persistence、seasonal_day、lightgbm_direct 在同一时间折上比较。
- 第一轮冲分（v2 特征工程）与第二轮冲分（v7/v8/v9/v10 实验）已完成。
- 第二轮组合提交包已生成并通过契约门禁，**等待用户上传官方平台**。

### 2.2 官方提交记录

| 提交 | 方案 | 官方分 | 排名快照 | 状态 |
| --- | --- | ---: | ---: | --- |
| 2026-09-18 10:38 | persistence 探针 | 83.5246 | 74 | 已出分 |
| 2026-09-20 00:30 | v5 组合 | 86.1107 | 74 | 已出分 |
| 2026-09-20 09:29 | round2 组合 | 待回填 | 待回填 | **待用户上传** |

依据：`manifests/official_submissions.csv`。

### 2.3 第二轮内部最优（提交决策依据）

- 分目标 argmin 组合：`generator_1` 用 v7 加性残差 LightGBM（分目标 dev MAPE 0.060464）；`generator_all` 用 v8 比值 LightGBM（分目标 dev MAPE 0.048335）。
- 组合口径 pooled MAPE **0.054400**（1-MAPE 0.945600），优于 v5 的 0.055971。
- 提交包：`outputs/submissions/20260920T092901Z_round2_composed/这模型调的有力量_gas_predict_prelim.zip`，SHA256 `96069ff4b6b255b969a2d6b7a80b6ff303e745995df1428fb35e6757037bc97e`。
- 详细记录见 `docs/score_push_round2_result_v1.md` 与 `docs/score_push_round2_plan_v1.md`。

### 2.4 唯一未闭环事项

第二轮提交包尚未上传官方平台，`official_score` 与 `submitted_at_local` 待回填。**接手后应先向用户确认该提交是否已上传、官方分是否已出。**

---

## 3. 冻结纪律（最高优先级，违反即失败）

以下约束在初赛阶段已冻结，**复赛适配时不得静默修改历史 v1 文件**；如需变更，必须新建 v2 文件并保留 v1。

1. **时间因果性**：预测起点 `t` 仅可使用 `<= t` 的信息；禁止任何 `> t` 的观测、目标、耗气或用户消耗。
2. **proxy-sensitive 三字段未来值禁用**：`generator_use_blast_furnace_gas`、`generator_use_coke_gas`、`generator_use_converter_gas` 当前时刻值可用，未来值禁用。
3. **特征命名**：所有新特征必须使用 `feat_` 前缀。
4. **因果前向填充**：仅允许向后看，`limit_steps <= 4`。
5. **封存留出集**：`holdout_01` 全程封存，候选冻结前不得消费；相关 run 的 `sealed_holdout_evaluated` 必须为 `False`。
6. **切分指纹不变**：初赛冻结切分指纹为 `14c740902ea974d5de1d0836cbcae11c0c49b077c81487e407c69c4d0798d28e`。**注意：复赛长周期需要新切分，应新建 `splits_v2`，不得改动 `splits_v1`。**
7. **失败实验也留痕**：失败 run 必须保留 `failure.json` 与 registry 失败行，不得删除或覆盖。
8. **run_id 不可覆盖**：一旦创建的 run_id 不得复用或覆盖。

依据：`configs/protocol_v1.yaml`、`docs/task_contract_v1.md`、`configs/cleaning_v1.yaml`、`configs/splits_v1.yaml`。

---

## 4. 复赛与初赛的核心差异

| 维度 | 初赛（当前） | 复赛 | 依据 |
| --- | --- | --- | --- |
| 数据文件前缀 | `Pre_*` | **`Semi_*`** | `data/data_dictionary.xlsx` → "初赛复赛"表 |
| 预测周期 | 短周期 8 步（2 小时） | **长周期 96 步（24 小时）** | `README.md:20-21` |
| 提交命名 | `teamname_gas_predict_prelim.zip` | 复赛命名（**待官方确认**） | `docs/task_contract_v1.md:42` |
| 阶段标识 | `prelim` | `semi`（建议值） | 各 config `competition_stage` |
| 输出列数 | `s_result.csv` 17 列 | 预计 193 列（datetime + 96×2） | 由 8 步推得 |
| 半决赛额外 | — | 气柜/机组/电价约束的发电调度优化 | `README.md:23` |

数据字典原文（"初赛复赛"工作表）：
- 初赛数据文件前缀：`Pre`
- 复赛数据文件前缀：`Semi`
- 初赛与复赛的字段说明一致（gas / gas_holder / gas_user / load 四类 + 江苏电价峰谷时段图）。

---

## 5. 复赛适配清单

### A. 数据层（⚠️ 当前硬阻塞）

- **复赛数据 `Semi_*.csv` 尚未出现在项目中**。`data/` 仅有 `Pre_*.csv`，`test/` 仅有 `Pre_test_*.csv`。
- 接手第一步：向用户确认复赛数据是否已下发，索取数据文件并确认：
  1. 训练期与测试期时间范围；
  2. 是否仍为 15 分钟粒度；
  3. 测试集 `Semi_test_load.csv` 是否仍含起点当前时刻目标观测；
  4. 边界上下文重复行是否仍存在。

### B. 配置层（`configs/`）

| 文件:行 | 需改动 |
| --- | --- |
| `configs/splits_v1.yaml:4` | `maximum_horizon_steps: 8` → 96；**应新建 `splits_v2.yaml`**，v1 冻结 |
| `configs/cleaning_v1.yaml:9` | `boundary_context_timestamp: '2025-05-01 00:00:00'` 等按复赛数据新建 `cleaning_v2.yaml` |
| `configs/baseline_v1.yaml:12` | `horizons_steps: [1..8]` → `[1..96]` |
| `configs/protocol_v1.yaml:4,62` | `stage: prelim_short_forecast`、`prelim_archive`；`long_steps: 96` 已预留可直接启用 |
| 各 `experiment_*.yaml` | `horizons_steps` / `horizons_minutes` 扩至 96 |
| 各 `submission_*.yaml` | `competition_stage`、`archive_suffix`、`horizons_*` |

### C. 脚本层（`src/`，硬编码点）

| 文件:行 | 硬编码内容 | 建议 |
| --- | --- | --- |
| `src/prepare_dataset.py:47` | `Pre_{family}.csv` / `Pre_test_{family}.csv` | 将数据前缀参数化（从 config 读取 `data_prefix`） |
| `src/validate_contract.py:70,71,76,82,101,124,152` | `Pre_*` 文件名 + `192` 起点断言（`:82`） | 前缀与起点数参数化 |
| `src/data_quality_audit.py:109` | `Pre_{family}.csv` | 前缀参数化 |

参数化良好的脚本（horizon 已从 config 读取，无需改代码）：
- `src/build_splits.py`（`maximum_horizon_steps` 来自 config）
- `src/build_submission*.py`（`horizons_steps` / `horizons_minutes` 来自 config）
- `src/run_baseline.py`、`src/run_experiment_v2.py`、`src/run_experiment_v7.py`、`src/run_ensemble_eval.py`

> **性能警告**：`one_model_per_target_and_horizon` 策略下，96 步 × 2 目标 = **192 个 LightGBM 模型**，训练成本约放大 12 倍（相对 8 步）。接手时需先评估长周期建模策略（是否需要共享模型、直接多步、或分 horizon 分组）。

### D. 清单/注册表（`manifests/`、`experiments/`）

| 文件 | 说明 |
| --- | --- |
| `manifests/splits_v1.csv` + `manifests/splits_v1.sha256` | 基于 8 步冻结；复赛需新建 `splits_v2.csv` + `.sha256` |
| `manifests/baseline_results_v1.csv` | 初赛结果，保留为历史 |
| `manifests/official_submissions.csv` | 含 `competition_stage` 列；复赛需新增 semi 行 |
| `experiments/registry.csv` | 13 列结构（无 stage 列）；复赛 run 需在新切分上重跑并追加 |

registry 13 列结构：`run_id, created_at_utc, git_commit, git_dirty, config_sha256, prepared_train_sha256, split_sha256, best_development_model, best_development_mape, best_development_score_1_mape, duration_seconds, artifact_directory, sealed_holdout_evaluated`。

### E. 文档层（`docs/`）

| 文件 | 需改动 |
| --- | --- |
| `README.md:5-7,20-23,39-47` | 当前阶段、任务摘要、阶段门禁 |
| `docs/task_contract_v1.md` | 初赛短周期契约；**需新建 `task_contract_v2.md`**（复赛长周期契约），v1 不得静默修改 |
| `docs/environment.md:28` | OR-Tools 待调度阶段安装（半决赛用） |
| `docs/baseline_result_v1.md`、`docs/data_audit_report_v0.md`、`docs/data_quality_audit_v1.md` | 初赛口径，保留为历史 |

### F. 环境层

- 半决赛调度优化需安装 **OR-Tools**（`docs/environment.md:28` 已预告）。
- 正式提交前需在离线 Linux 环境重新验证运行。

---

## 6. 关键文件地图（接手必读）

### 6.1 契约与协议（只读参照，不得静默修改）

- `configs/protocol_v1.yaml` —— 冻结协议（信息集、目标、特征、提交格式）
- `configs/protocol_v0.yaml` —— 协议草案（含 long_steps: 96 预留）
- `docs/task_contract_v1.md` —— 任务合同 v1（信息集 + 初赛输出契约）
- `docs/task_contract_v0.md` —— 任务合同 v0（含初赛评分构成说明）
- `configs/cleaning_v1.yaml` —— 因果清洗规则
- `configs/splits_v1.yaml` —— 滚动切分规则
- `docs/cleaning_policy_v1.md` —— 清洗策略说明

### 6.2 核心脚本（可复用）

- `src/prepare_dataset.py` —— 数据准备（含数据前缀硬编码）
- `src/build_splits.py` —— 切分构建与指纹
- `src/run_baseline.py` —— 基线 runner
- `src/run_experiment_v2.py` —— v2 特征工程 runner
- `src/run_experiment_v7.py` —— v7/v8/v9 统一 runner（三模式目标变换 + v9 特征组 + 失败留痕）
- `src/run_ensemble_eval.py` —— v10 融合评估器
- `src/build_submission.py` / `build_submission_v5.py` / `build_submission_round2.py` —— 提交构造器
- `src/validate_contract.py` —— 契约门禁
- `src/render_results_report.py` —— 结果渲染

### 6.3 结果文档

- `docs/score_push_round2_plan_v1.md` —— 第二轮计划
- `docs/score_push_round2_result_v1.md` —— 第二轮结果（含口径隔离声明）
- `docs/baseline_plan_v1.md` / `docs/baseline_result_v1.md` —— 第一轮基线与结果

### 6.4 测试

- `tests/` —— 契约测试（当前 `pytest -q` 为 13 passed）

---

## 7. 环境与可复现命令

### 7.1 Python 解释器（⚠️ 路径不一致，务必核实）

- 实际使用：`E:/anaconda/envs/vocs/python.exe`
- 文档记录：`docs/environment.md` 与 `README.md` 写的是 `D:\anaconda\envs\vocs\python.exe`
- **接手时请先核实解释器真实路径**，以能跑通 `pytest` 的路径为准。

### 7.2 关键版本

Python 3.10.x、NumPy 2.1.3、Pandas 2.2.3、scikit-learn 1.5.2、LightGBM 4.7.0、openpyxl 3.1.5。GPU：RTX 4060 Laptop（8GB）。

### 7.3 常用命令（全部在项目根执行）

契约校验：

    <python> src/validate_contract.py --root .

数据准备与切分：

    <python> src/prepare_dataset.py --root . --config configs/cleaning_v1.yaml
    <python> src/build_splits.py

测试：

    <python> -m pytest -q

基线：

    <python> src/run_baseline.py --config configs/baseline_v1.yaml

实验（v7/v8/v9 同 runner）：

    <python> src/run_experiment_v7.py --config configs/experiment_v7_resid_additive.yaml

融合：

    <python> src/run_ensemble_eval.py --config configs/ensemble_v10_blend.yaml

提交构造：

    <python> src/build_submission_round2.py --config configs/submission_round2_composed.yaml

---

## 8. 待办与阻塞项

| 优先级 | 事项 | 状态 |
| --- | --- | --- |
| 阻塞 | 取得复赛数据 `Semi_*.csv` 并确认时间范围 | 待用户提供 |
| 高 | 确认第二轮提交是否已上传、官方分是否已出并回填 | 待用户确认 |
| 高 | 新建 `splits_v2.yaml`（`maximum_horizon_steps: 96`）与对应 `splits_v2.csv`/`.sha256` | 未开始 |
| 高 | 新建 `cleaning_v2.yaml` 适配复赛数据 | 未开始 |
| 高 | 数据前缀参数化（`prepare_dataset.py`、`validate_contract.py`、`data_quality_audit.py`） | 未开始 |
| 中 | 新建 `task_contract_v2.md`（复赛长周期契约） | 未开始 |
| 中 | 评估 96 步长周期建模策略（192 模型成本问题） | 未开始 |
| 中 | 新建复赛计划与结果文档 | 未开始 |
| 低 | 半决赛调度优化环境准备（OR-Tools） | 未开始 |

---

## 9. 风险与注意事项

1. **不得静默修改 v1**：所有初赛冻结文件（protocol_v1、cleaning_v1、splits_v1、task_contract_v1 等）必须保留；复赛变更一律新建 v2。
2. **切分指纹会变**：复赛长周期必然产生新切分指纹，这是预期行为，但必须在结果文档中明确声明，不能与初赛指纹混用比较。
3. **官方分与内部指标口径隔离**：官方总分与内部 `1-MAPE` 定义不同，禁止等同或换算；提交决策只依据内部组合口径 pooled development MAPE。
4. **proxy-sensitive 与因果性在长周期下更易被违反**：96 步跨度更大，务必复检特征构造是否只用到 `<= t` 信息。
5. **训练成本放大**：96 步 × 2 目标 = 192 模型，需先做小规模验证再全量运行，避免超时。
6. **失败留痕纪律**：复赛实验同样要求失败也留痕，run_id 不可覆盖。
7. **原始数据只读**：`data/`、`test/` 下原始文件不得修改，所有派生数据仅写入 `outputs/`。
8. **历史实验不要清理**：`outputs/experiment_runs/` 下失败 run 目录、`experiments/registry.csv` 失败行均须保留。

---

## 10. 术语与口径速查

| 术语 | 含义 |
| --- | --- |
| 起点 / origin | 滚动预测的发起时刻 `t` |
| 短周期 | 初赛 8 步（t+15 ~ t+120 分钟） |
| 长周期 | 复赛 96 步（t+15 ~ t+1440 分钟，即 24 小时） |
| pooled MAPE | 跨折、跨目标、跨步长汇总的 MAPE |
| 1-MAPE | 内部开发分（与官方分口径不同） |
| proxy-sensitive | 与目标业务近因强、未来值须禁用的耗气三字段 |
| 封存留出集 | `holdout_01`，候选冻结前不得消费 |
| run_id | 形如 `20260920T043055Z_da27404d_5aa2b13f`，全局唯一不可覆盖 |
| 切分指纹 | `manifests/splits_v1.sha256` 记录的 SHA256 |

---

## 11. 交接完成检查清单

接手 AI/工程师在开始复赛编码前，应逐项确认：

- [ ] 已通读本文档第 3 节冻结纪律。
- [ ] 已确认 Python 解释器真实路径并可跑通 `pytest -q`。
- [ ] 已确认第二轮提交的官方上传与出分状态。
- [ ] 已取得复赛数据 `Semi_*.csv` 并确认时间范围。
- [ ] 已阅读 `docs/task_contract_v1.md`、`configs/protocol_v1.yaml`、`configs/splits_v1.yaml`。
- [ ] 已确认复赛变更采用"新建 v2"而非"修改 v1"的方式。
- [ ] 已评估 96 步长周期建模策略与训练成本。