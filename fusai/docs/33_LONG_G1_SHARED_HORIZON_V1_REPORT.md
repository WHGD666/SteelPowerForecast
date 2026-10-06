# v28 long-g1 共享 horizon 树 v1 实验报告

## 1. 裁决

共享 horizon 树是目前少数呈现广泛正向结构的 long-g1 新路线，但 v1 尚不能提交。它在 pooled
official long-g1 上提升 0.3597 pct，四个 horizon bucket 全部提升、两个可评估 episode 全部改善；
但未达到 +0.5 pct 组件门槛，并在训练历史只有两天的 BF3-active 折退化 0.3840 pct。

## 2. 运行信息

- run_id：`20261004T105317Z_long_g1_shared_aa987f06b5`
- 角色：scientific OOF
- 控制：v28 的 96 棵独立 horizon 树 OOF
- 候选：每折一个共享 horizon 树，保留 v28 fuel 公式
- 规模：3 个模型；训练行数 13,776 / 161,232 / 216,528
- 耗时：14.985 秒
- holdout：未评估
- submission：未生成

## 3. 总体结果

| 版本 | MAPE | accuracy | 相对控制变化 |
|---|---:|---:|---:|
| v28 per-horizon control | 0.133052 | 86.6948% | 0.0000 pct |
| shared horizon tree | 0.129454 | 87.0546% | +0.3597 pct |

legacy point 辅助口径提升 0.3617 pct，与官方区间均值方向一致。

## 4. 稳定性

| 折 | 控制 accuracy | 候选 accuracy | 变化 |
|---|---:|---:|---:|
| BF3-active | 87.4808% | 87.0968% | -0.3840 pct |
| late September | 84.3724% | 85.4667% | +1.0943 pct |
| holder1-active | 87.4572% | 88.0709% | +0.6138 pct |

Horizon bucket 增益：near +0.3258、mid1 +0.5866、mid2 +0.2237、far +0.3578 pct。
普通工况损失 0.1541 pct，仍低于预登记 0.2 pct 安全线。pre-event、active-event、recovery
分别比控制改善约 0.8401、0.9459、0.7506 pct。episode 12/13 均改善。

## 5. 机制判断与下一步

最早折只允许使用 9 月 1–2 日，展开后 13,776 个成熟 horizon 样本；后两折分别有 16.1 万和
21.7 万。共享模型需要同时学习 horizon、目标时刻与工况交互，样本不足对它的影响比 96 个各自局部拟合
的控制更明显。该解释在运行前没有用于挑选候选，只用于定义一次预登记的稳定性复验。

下一版本唯一改变 tree training start：从 9 月 1 日扩展到 8 月 1 日。fuel OLS 继续只用九月，
标签、特征、参数、融合和验证全部不变。若 v2 仍存在负折或 pooled 未达到 +0.5 pct，停止共享树窗口研究。
