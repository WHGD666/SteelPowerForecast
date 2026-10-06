# v28 long-g1 共享 horizon v2 实验报告

## 1. 裁决

8–9 月树窗修复了 v1 的样本饥饿：pooled long-g1 提升 0.6336 pct，三个折和四个 horizon bucket
全部改善，普通工况也提升 0.3845 pct。它通过组件收益、跨折、普通工况和 near/far 门禁，成为当前最强的
新 long-g1 组合积木。

但它仍不单独提交：增益未达到预登记 +0.8 pct 单独候选线，且两个 episode 只改善一个，
`component_gate_passed=False`。后续不再扫描树窗或共享/独立树融合比例。

## 2. 运行信息

- run_id：`20261004T122856Z_long_g1_shared_85d58617c3`
- parent：`20261004T105317Z_long_g1_shared_aa987f06b5`
- 角色：scientific OOF
- 唯一变化：共享树训练起点 2025-09-01 → 2025-08-01
- fuel：仍只用九月，公式和 0.8/0.6 权重不变
- 规模：3 个模型；训练行数 299,472 / 446,928 / 502,224
- 耗时：24.106 秒
- holdout：未评估
- submission：未生成

## 3. 结果

| 版本 | MAPE | accuracy | 相对 v28 控制 |
|---|---:|---:|---:|
| v28 per-horizon control | 0.133052 | 86.6948% | 0.0000 pct |
| shared horizon Aug–Sep | 0.126716 | 87.3284% | +0.6336 pct |

legacy point 辅助口径提升 0.6362 pct。

| 折 | 控制 accuracy | 候选 accuracy | 变化 |
|---|---:|---:|---:|
| BF3-active | 87.4808% | 87.6719% | +0.1911 pct |
| late September | 84.3724% | 86.3843% | +2.0119 pct |
| holder1-active | 87.4572% | 87.6143% | +0.1571 pct |

Horizon bucket：near +0.5102、mid1 +0.6731、mid2 +0.5266、far +0.6944 pct。
普通、pre-event、active-event、recovery 分别变化 +0.3845、+0.2114、+0.1199、+1.7938 pct。

## 4. Episode 风险

- episode 12：+2.8225 pct，但仅 265 个评分单元。
- episode 13：-0.5877 pct，共 7,872 个评分单元。

episode 13 的退化主要位于 mid/far。其起点平均表现为较高 holder_2 和较低燃料比，但该模式是在看过该
episode 后发现；用同一 OOF 搜索阈值会造成事后门控。测试期 holder_2 整体又偏高，因此不据此制作规则。

## 5. 后续定位

- 冻结 v2 预测结构、8–9 月树窗和全部参数，不再调窗或融合。
- 标记为 `promising_component_not_standalone`，不是平台成绩。
- 暂不生成 ZIP；优先在 long-gall 做同样的结构消融。
- 若 long-gall 得到跨折稳定的独立收益，再把两个区块组成唯一候选并执行平台探针。
