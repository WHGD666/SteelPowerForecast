# long-g1 MAPE 对齐燃料回归报告

## 结论

inverse-target weighted median fuel proxy 产生小幅 pooled 正收益，但跨折不稳定且远低于晋级门槛。
该路线不生成提交包，也不扫描 quantile、alpha、分母下限或燃料窗口。

- run：`20261004T152405Z_long_g1_mape_fuel_20b0cc90f0`
- 计算：3 个燃料线性规划模型，0 个 LightGBM，3.734 秒
- long-g1：86.6948% → 86.8033%，**+0.1085 pct**
- short 投影：89.0483% → 88.9631%，**-0.0852 pct**
- legacy point long：+0.1083 pct

## 稳定性

| 视图 | 准确率变化 pct |
|---|---:|
| BF3-active | +0.0189 |
| late September | +0.6660 |
| holder1-active | -0.1735 |
| ordinary | +0.1018 |
| near 15–120 | -0.0852 |
| mid 135–360 | +0.0111 |
| mid 375–720 | +0.2362 |
| far 735–1440 | +0.1094 |

2/2 episode 改善，但 pooled 未到 +0.30，最差折为负，且 near/short 轻微退化。
`component_gate_passed=False`，`standalone_submission_eligible=False`。

## 解释

候选在三个折相对 OLS 的平均燃料改变量分别为 +0.115、-1.593、-3.512 MW。随着九月样本增加，
加权中位数回归倾向整体下修燃料锚点；该方向对 late-September 有利，但对 holder1-active 有害，
不能安全迁移到十月。燃料损失函数不是当前 1.54 分缺口的主因。

