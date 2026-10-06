# long-g1 MAPE 对齐燃料回归协议

## 假设

v28 long-g1 的最终预测有 60%–80% 来自九月 fuel proxy，但该 proxy 用 OLS 最小化平方误差，和官方
MAPE 不一致。固定 `quantile=0.5, alpha=0`，并给训练行使用 `1/max(abs(y), 20 MW)` 权重，得到的
weighted median linear regression 直接最小化带安全下限的训练 MAPE。

## 唯一变化

只替换每折九月 `generator_1 ~ 三类发电耗气` 的燃料映射。候选由冻结控制 OOF 做精确增量重构：

`candidate = control + fuel_weight(h) × (mape_median_fuel - ols_fuel)`

因此树预测、特征、标签、LightGBM、切分和 horizon 融合全部逐值冻结。`fuel_weight` 保持部署代码的
真实语义：15/30 分钟为 0.8，其余为 0.6。

## 信息边界

- 每折只拟合 2025-09-01 至该折 train_end 的成熟训练行；
- 权重只依赖训练目标，不进入模型特征；
- 验证和测试目标均不参与拟合；
- 不读取预测起点之后的过程观测；
- 平台分数只用于确定控制组，不参与参数选择。

## 门禁

- pooled official long-g1 至少 +0.30 pct；+0.80 pct 才可考虑单区块提交；
- 三折均不得退化；ordinary 损失不超过 0.10 pct；
- near/far 任一段损失不超过 0.10 pct；short 投影损失不超过 0.10 pct；
- 多数可评估 episode 必须改善。

本实验只拟合 3 个小型线性规划模型，不训练 LightGBM、不生成提交包。若失败，不扫描 quantile、alpha、
分母下限或燃料窗口。

