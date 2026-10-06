# v16/v29b short-g1 恢复字段隔离实验协议

## 1. 假设

v29b 的 short-g1 来自 v16_final，官方准确率 89.47%，但其树仍使用旧清洗列。后续 v28 恢复了
`blast_furnace_3`、`air_heater_3` 并补全高炉平衡，历史交底明确指出这项变化从未在 short-g1
单独验证。本实验只检验这一项。

## 2. 控制与候选

- 控制：v16 legacy feature engine + September point-label tree + September fuel OLS。
- 候选：v28 restored feature engine + 完全相同的 tree/fuel 训练口径。
- 两者均采用 v16 的 `20% tree + 80% fuel`，8 个短周期 horizon 不改变权重。
- 冻结 short-gall、long-g1、long-gall；不接入标签均值、hazard、异常加权或平台反馈。

说明：旧引擎确实排除 `blast_furnace_3` 与 `air_heater_3`，旧高炉平衡还遗漏
`air_heater_5`；`converter_user1` 本来已作为原始协变量存在，因此不把它虚报为本实验新增字段。

## 3. 验证

- 折：`wf_03_bf3_active`、`wf_04_late_september`、`wf_05_holder1_active`。
- horizon：15–120 分钟，共 8 步。
- primary：官方区间均值 MAPE；auxiliary：legacy point MAPE。
- 模型数：3 × 8 × 2 = 48 个 CPU LightGBM。
- 不生成提交包。

组件门禁：官方 short-g1 至少 +0.15 pct，最差折不低于 -0.05 pct，legacy point 不退化，
任一 horizon 损失不超过 0.10 pct。只有达到 +0.50 pct 才具备单区块平台提交资格。

## 4. 命令

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_short_g1_restored_columns --config configs\round2_v3\experiments\short_g1_restored_columns_v1.yaml
```
