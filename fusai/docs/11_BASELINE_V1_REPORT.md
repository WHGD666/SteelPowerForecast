# v3 Baseline V1 评估报告

## 1. 运行身份

- run_id：`20260929T124936Z_r2v3_baseline_46e02afe85`
- 状态：completed
- 角色：competition baseline；已完成一次外部校准，不再作为高分提交候选
- 协议：`round2_v3`
- 验证：5 个冻结 model-selection walk-forward 折
- 模型：target-time 日历中位数弱基线 + shared-horizon LightGBM
- 运行时间：358.08 秒
- OOF 行数：559,104；其中 LightGBM 279,552 行
- holdout：未评估；Sep 28–29 未进入模型选择结果

预测、指标和 summary 的 SHA-256 均与 run manifest 一致；预测键无重复，实际值和预测值均有限。

## 2. 总体 OOF 结果

| 模型 | 周期 | MAPE | 1-MAPE | MAE |
|---|---|---:|---:|---:|
| 日历中位数 | 短期 | 19.60% | 80.40% | 40.17 |
| LightGBM | 短期 | **11.67%** | **88.33%** | 21.51 |
| 日历中位数 | 长期 | 18.51% | 81.49% | 37.87 |
| LightGBM | 长期 | **15.47%** | **84.53%** | 30.96 |

LightGBM 对短期有明确增益；长期只有约 3.04 个百分点的绝对准确率增益，说明远期仅使用 origin 历史状态的上限较明显。

## 3. 分目标结果

| 周期 | 目标 | MAPE | 1-MAPE | 判断 |
|---|---|---:|---:|---|
| 短期 | generator_1 | 15.02% | 84.98% | 当前首要瓶颈 |
| 短期 | generator_all | 8.33% | 91.67% | 已越过 90% 工程线 |
| 长期 | generator_1 | 15.54% | 84.46% | 接近 85%，仍不稳定 |
| 长期 | generator_all | 15.40% | 84.60% | 接近 85%，远期退化明显 |

短期总体未达到 90%，主要由 `generator_1` 造成。长期两个目标都接近但尚未稳定越过 85%。

## 4. 时间折稳定性

| 折 | short g1 | short gall | long g1 | long gall |
|---|---:|---:|---:|---:|
| July regular | 72.95% | 82.77% | 76.57% | 64.45% |
| August regular | 90.31% | 94.03% | 90.68% | 92.48% |
| BF3 active | 91.38% | 95.42% | 88.74% | 93.97% |
| Late September | 79.49% | 93.92% | 81.92% | 90.81% |
| Holder1 active | 88.95% | 92.96% | 83.55% | 83.36% |

主要风险不是平滑的 horizon 衰减，而是工况和训练跨度迁移：

- 七月折 `generator_1` 明显低估，`generator_all` 明显高估。
- 九月中下旬 `generator_1` 转为高估。
- 第一折训练历史最短，树模型对超出训练分布的关系缺乏外推能力。
- 晚九月新投运字段和低负荷工况会改变特征含义。

因此不得通过删除第一折来制造更高均值。下一模型应增强目标结构和跨工况稳定性。

## 5. Horizon 诊断

| 周期/区间 | g1 准确率 | gall 准确率 |
|---|---:|---:|
| short 15–30 min | 85.38% | 93.31% |
| short 45–60 min | 85.16% | 92.28% |
| short 75–120 min | 84.69% | 90.54% |
| long 0–2 h | 84.77% | 86.29% |
| long 2–6 h | 84.56% | 85.05% |
| long 6–12 h | 84.33% | 84.32% |
| long 12–24 h | 84.44% | 84.31% |

`generator_1` 从近端开始就偏弱，单纯增加未来路径不一定能解决；`generator_all` 则呈现清晰的随 horizon 退化，更适合后续尝试未来煤气路径或日历/工况融合。

## 6. 物理与融合诊断

LightGBM OOF 中以下违规均为 0：

- 非正预测。
- `generator_1 > 200`。
- `generator_all > 440`。
- `generator_1 > generator_all`。

同一 OOF 上的非交叉验证 oracle 混合只为 `generator_1` 带来约 0.14–0.20 个百分点提升，且对 `generator_all` 最优权重为 100% LightGBM。因此简单日历融合不是当前主要突破口。

## 7. 平台校准结果

- submission_id：`20260929T212923_calibration_baseline_v1_91cd219d33`
- 提交时间：2026-09-29 21:44:10
- 平台总分：**48.9241**
- 历史 legacy v9：**52.4085**
- 差值：**-3.4844**
- 平台分项：未返回

提交前的 schema、时间覆盖、物理边界、模型哈希和 ZIP 成员检查全部通过。因此该结果
主要被视为泛化/模型问题，而不是格式问题。由于平台没有短期、长期或分目标明细，不能
把分差归因到任一目标，也不能从总分反推测试真值。

## 8. 决策

该 run 被接受为第一个可复现 v3 控制组，但拒绝为高分候选；不得重复提交同族微调版本。

下一项受控实验优先验证目标层次分解：

```text
large_units = generator_all - generator_1
derived_generator_1 = predicted_generator_all - predicted_large_units
```

该实验只改变目标表示，复用相同标签、切分、特征、参数和 `generator_all` OOF，可直接判断层次结构是否改善当前最弱的 `generator_1`。实验已完成并退化，故被拒绝。

下一阶段先恢复 legacy v9 等价流程，识别目标反馈、煤气平衡、训练窗口和长短期策略中
真正有效的部分；完整复盘及新提交门禁见 `docs/12_PLATFORM_CALIBRATION_POSTMORTEM.md`。
