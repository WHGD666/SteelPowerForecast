# short-g1 一小时燃料趋势外推协议

## 假设

高自由度 dynamic OLS 在一个近期折发生严重制度失稳。新的候选不学习跨工况映射，只使用预测起点可见的三类发电耗气最近一小时变化，并通过每折九月 fuel OLS 系数换算为 g1 修正。

## 冻结公式

对每类发电耗气：

`forecast(h) = current + min(h/60, 1) × (current - lag_60min)`

然后：

`candidate = v16_control + clip(0.8 × (fuel_proxy_forecast - fuel_proxy_current), -5, +5 MW)`

固定 60 分钟滞后、60 分钟达到完整趋势、之后不再扩大、v16 fuel weight=0.8、对称 cap=5 MW。禁止扫描滞后、阻尼、权重或 cap。

## 合同与门禁

- 只改 short `generator_1`；short-gall 与全部 long 冻结；
- 3 个近期 walk-forward 折、8 个 horizon、官方区间均值 MAPE；
- fuel OLS 每折仅使用 9 月 1 日至训练截止数据；
- 不使用真实目标历史、未来过程量、测试目标或平台反馈拟合；
- pooled 至少 +0.50 pct；最差折/horizon ≥-0.10 pct；普通工况退化不超过 0.10 pct；至少 2 个事件且半数改善；
- 通过也不自动生成 ZIP，先审查完整结果。

该路线与 dynamic OLS 的区别是：它只外推可观测物理趋势，不学习可能随工况漂移的 36 维回归系数。
