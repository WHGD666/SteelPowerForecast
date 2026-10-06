# long-g1 MAPE 对齐树训练报告

## 结论

inverse-target weighted L1 没有改善 v28 long-g1，且在最早近期折和普通工况明显退化。
实验未通过组件门禁或单独提交门禁，按预登记停止条件关闭；不扫描 denominator floor、权重指数或树窗。

- run：`20261004T151651Z_long_g1_mape_00babc641a`
- 耗时：50.891 秒；288 个 LightGBM；52 个冻结特征
- long-g1：86.6948% → 86.5951%，**-0.0997 pct**
- short 投影：89.0483% → 88.9512%，**-0.0971 pct**
- legacy point long：-0.0980 pct

## 稳定性

| 视图 | 准确率变化 pct |
|---|---:|
| BF3-active | -0.4558 |
| late September | +0.1276 |
| holder1-active | +0.1049 |
| ordinary | -0.3258 |
| near 15–120 | -0.0971 |
| mid 135–360 | -0.0824 |
| mid 375–720 | -0.0362 |
| far 735–1440 | -0.1376 |

2/2 可评估 episode 改善，但不足以抵消普通工况和早期折退化。`component_gate_passed=False`，
`standalone_submission_eligible=False`。

## 解释与下一步

树只占 v28 最终 long-g1 的 20%–40%，而 inverse-target 权重改变了树分裂对低负荷样本的关注，
却破坏了 BF3-active 与 ordinary 工况。下一实验不继续调整树，而只验证占最终预测 60%–80% 的
九月 fuel proxy：把 OLS 换为固定的 inverse-target weighted median regression，其余预测逐值冻结。

