# Phase C：Horizon-specific 动态燃料 long-g1 实验协议

## 1. 为什么转向这条路线

Phase B 已证明，直接给共享 LightGBM 增加事件过程特征最多只带来 0.3795 个百分点的 long-g1 OOF 改善，且跨事件不稳定，无法承担从 63.4621 冲到 65 的主要增益。

动态燃料路线研究的是另一个问题：v29b 一类旧方案主要根据预测起点的燃料状态形成静态锚点，但 24 小时内燃料—发电关系会随 horizon 演化。此前只读诊断曾得到 long-g1 约 +4.07 个百分点，但该结果没有形成正式代码、冻结配置和不可覆盖 run，因此只能视为高价值假设，不能直接作为候选证据。

本实验的目标是用当前正式标签、五个冻结时序折和精确 baseline 控制组，验证该增益是否能够复现。

## 2. 唯一主要变化

只改变 long `generator_1` 的预测表示：

- 控制组：冻结 baseline v1 的 long-g1 OOF，逐值复用，不重新训练树模型。
- 动态模型：对 96 个 horizon 分别拟合一个线性模型。
- 唯一晋级候选：`70% dynamic_fuel_ols + 30% control`。
- short 和全部 `generator_all` 保持不变。
- 首轮不加入 Phase B 的高炉生产特征，也不搜索融合权重。

独立输出 `dynamic_fuel_ols` 仅用于判断机制，不具备晋级资格。这样可以避免看到 OOF 后在多个权重中挑最优值。

## 3. 因果输入

每个预测起点只使用三路发电煤气耗用的起点及历史值：

- 当前值；
- 1、4、8、16、96 个 15 分钟步长的 lag，即 15、60、120、240、1440 分钟；
- 4、16、96 个步长的历史均值，即 1、4、24 小时均值；
- 对应目标时刻的小时、分钟、星期、周末、月份和月内日期日历项。

共 27 个燃料历史特征，加 9 个目标时刻日历特征，模型输入共 36 个。特征 registry 的最大来源偏移为 0，真实目标历史和未来过程量均未进入模型。

## 4. 折内训练边界

每个验证折、每个 horizon 独立执行：

1. 根据 horizon 计算标签已经完整成熟的最晚训练起点。
2. 从该起点向前取恰好 21 天、每 15 分钟一个起点的近期窗口。
3. 缺失值中位数、标准化参数和 OLS 系数全部只在该折该 horizon 的训练数据内拟合。
4. 对验证起点预测后，与冻结 baseline OOF 按折、起点、目标区间和 horizon 严格对齐。
5. 真值必须与冻结控制组逐值一致，否则运行失败。

该设计共有 5 折 × 96 horizon = 480 个轻量线性模型。它是 CPU 任务，不使用 GPU，但仍由人工执行正式命令。

## 5. 晋级门禁

唯一可晋级变体 `blend_dynamic70_control30` 必须同时满足：

- long-g1 总体准确率至少提高 0.8 个百分点；
- 普通日起点退化不超过 0.2 个百分点；
- 最近三个折的平均增益不为负；
- 最近三个折中最差单折不得低于 -0.2 个百分点；
- 超过半数可评估独立 episode 得到改善；
- 近端和远端 horizon 均通过安全检查。

即使全部通过，也只表示可以进入“与冻结 v29b 单区块集成”的下一阶段，不会自动生成提交包。

## 6. 运行与产物

轻量预检已通过：

```text
PASS preflight folds=5 horizons=96 models=480 fuel_features=27 model_features=36 max_source_offset=0
```

正式运行命令：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_dynamic_fuel_g1 --config configs\round2_v3\experiments\dynamic_fuel_g1_v1.yaml
```

正式 run 会保存：

- `oof_predictions.csv`：控制、standalone OLS 和冻结 70/30 候选；
- `coefficients.csv`：每折每 horizon 的折内 imputer、scaler 和 OLS 参数；
- `dynamic_fuel_feature_registry.csv`：因果来源偏移；
- `metrics.csv`、`variant_comparison.csv`、`gate_results.csv`；
- `run_manifest.json`、`events.jsonl` 和人类可读 `report.md`。

输出目录不可覆盖。失败 run 也会记录到实验 registry。

## 7. 解释边界

- 本实验的控制组是可复现 baseline OOF，不是 v29b OOF；本地提升不能直接等价为 v29b 平台提升。
- 如果复现此前约 +4 pct 的方向，下一步仍需把动态模型按完全相同逻辑全量训练，并只替换 v29b 的 long-g1 区块进行安全审验。
- 如果总体改善小于 0.8 pct 或近期单折失败，则停止该路线，不通过改窗口、改权重、挑 horizon 的方式在同一 OOF 上追分。
- 平台提交只在正式本地门禁通过、v29b 其余三个区块逐值冻结、提交差异可归因后讨论。

