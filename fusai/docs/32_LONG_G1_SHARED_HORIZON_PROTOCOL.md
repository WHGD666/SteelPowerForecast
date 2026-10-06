# v28 long-g1 共享 horizon 树实验协议

## 1. 背景与假设

v29b 的 long-g1 来自 v28：九月训练窗内为 96 个 horizon 分别训练树。远端每棵树只有约数千条、
早期折甚至只有数百条可成熟样本，方差较大。此前 v3 baseline 使用过共享 horizon 结构，但同时改变了
标签、特征、训练窗和模型权重，不能据此判断该结构是否适合 v28。

本实验在 v28 内做纯净消融：以一个共享树替换 96 棵独立树，并加入 horizon 与目标时刻日历编码；
其余配方保持不变。

## 2. 冻结项

- 目标：仅 long `generator_1`。
- 原始特征：v28 restored columns/full balance。
- 训练窗：九月起至每折 `train_end`。
- 标签：legacy point at `t+h`，与 v28 一致。
- LightGBM 参数：与 v28 一致。
- fuel：每折九月三路耗气 OLS，15/30 分钟权重 0.8，其余权重 0.6。
- short-g1、short-gall、long-gall 不变。
- 不使用未来过程量、目标历史、平台反馈或事件事后标签。

唯一主要变化是：96 棵独立树变为 1 棵共享树。新增的 horizon/目标时刻日历列是共享模型识别预测位置
所必需的合法已知特征，不引入外部未来过程信息。

## 3. 验证与门禁

- 控制：冻结 anomaly-weight run 的 v28 ×1 OOF。
- 三个近期 walk-forward 折，96 horizon。
- primary：官方区间均值 MAPE；auxiliary：legacy point MAPE。
- 保存 fold、episode、ordinary 与 near/mid/far 指标。
- 规模：每折一个模型，共 3 个 CPU LightGBM；runner 不生成 ZIP。

晋级要求：long-g1 至少 +0.5 pct，最差折不低于 0，普通工况损失不超过 0.2 pct，
多数 episode 改善，near/far 损失均不超过 0.2 pct。达到 +0.8 pct 才具备单区块平台候选资格。

## 4. 运行命令

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_long_g1_shared_horizon --config configs\round2_v3\experiments\long_g1_shared_horizon_v1.yaml
```
