# long-g1 共享 horizon v2：8–9 月树窗稳定性复验

## 1. 唯一变化

相对 v1 run `20261004T105317Z_long_g1_shared_aa987f06b5`，只把共享树训练起点从
`2025-09-01` 提前到 `2025-08-01`。九月 fuel OLS、point 标签、restored 特征、模型参数、
目标时刻特征和 0.8/0.6 融合全部冻结。

这是为验证 v1 最早折仅两天训练历史导致的样本饥饿，不是开放窗口搜索。不会继续尝试 7 月、全期或其他日期。

## 2. 门禁和停止条件

门禁与 v1 相同：pooled long-g1 至少 +0.5 pct、最差折不低于 0、普通工况损失不超过
0.2 pct、多数 episode 改善、near/far 不低于 -0.2 pct；+0.8 pct 才可成为单区块平台候选。

如果 v2 仍有负折或 pooled 未达到 +0.5 pct，停止共享树窗口研究，不再选择 v1/v2 混合比例或继续扫训练起点。

## 3. 命令

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_long_g1_shared_horizon --config configs\round2_v3\experiments\long_g1_shared_horizon_v2_augsep.yaml
```
