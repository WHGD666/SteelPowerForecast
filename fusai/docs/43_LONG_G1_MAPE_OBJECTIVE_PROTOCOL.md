# long-g1 MAPE 对齐训练协议

## 假设

v28 long `generator_1` 的树以 MAE 训练，而官方指标为 MAPE。对训练行使用
`1 / max(abs(y), 20 MW)` 的样本权重，可在保持同一 L1 树模型的同时，使经验损失严格对应带安全下限的
MAPE，并减少高负荷样本对树分裂的过度支配。

## 唯一变化

候选只给 v28 `generator_1` 树训练传入均值归一为 1 的 inverse-target 权重。以下全部冻结：

- 九月树训练窗口；
- legacy point `t+h` 训练标签；
- v28 restored-column 因果特征；
- 300 棵树及全部 LightGBM 参数；
- 九月 fuel OLS；
- 15/30 分钟 fuel 权重 0.8，其余 horizon 0.6；
- 三个近期 walk-forward 折和官方区间均值评价口径。

权重只依赖各折训练标签，不进入特征，不读取验证或测试目标。20 MW 下限预先冻结；当前 g1 正常范围远高于该值，
其作用仅是防御异常接近零的标签。禁止运行后改下限或改成幂次权重。

## 控制与计算预算

控制直接复用已冻结 anomaly-weight run 中的 `control_weight1` OOF，SHA-256 为
`672dc4ab36f1e2352826f44b3f2411a3d199b40078d5e3c856a0fddd36aeb917`。
候选训练 3 折 × 96 horizon = 288 个 CPU LightGBM；不生成提交包。

## 晋级门禁

- pooled official long-g1 至少 +0.30 pct 才算有效组件；
- 至少 +0.80 pct 才允许考虑单区块平台提交；
- 三折不得有负增益；
- ordinary 工况损失不超过 0.10 pct；
- near 与 far 任一段损失不超过 0.10 pct；
- short 投影损失不超过 0.10 pct；
- 可评估事件必须多数改善。

任一关键门禁失败即关闭本路线，不扫描权重指数、标签下限或训练窗口。

