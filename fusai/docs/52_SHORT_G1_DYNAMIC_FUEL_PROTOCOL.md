# short-g1 动态燃料隔离实验协议

## 背景

long-g1 动态燃料、共享 horizon 和国庆修正均未迁移到十月平台，禁止继续在 long 区块微调。但既有动态燃料 OOF 在 15–120 分钟 near bucket 相对通用控制提高约 3.48 个百分点，而此前平台探针始终冻结 short，尚未回答它能否改善 v29b 的 v16-lineage short-g1。

## 唯一变化

控制为 `20261004T102800Z_short_g1_restored_1fb7013867` 中逐值冻结的 `v16_legacy_columns` OOF。候选固定为：

`0.70 × horizon-specific causal dynamic-fuel OLS + 0.30 × v16 control`

只评估 15–120 分钟 `generator_1`。不改变 short-gall、任何 long 输出、切分、标签或官方 MAPE 口径；不搜索 blend 权重。

## 因果与验证合同

- 三个近期 walk-forward 折：BF3-active、late-September、holder1-active；
- 每折每 horizon 只用训练截止前已成熟标签；
- 动态输入仅含起点及之前的三类发电耗气 current/lag/rolling 与目标时刻日历；
- 最近 21 天训练窗、OLS、折内中位数填补和标准化完全冻结；
- 24 个线性模型，CPU，无测试目标、无平台反馈拟合。

## 晋级门禁

- pooled short-g1 至少 +1.0 pct；
- 最差单折和最差单 horizon 不低于 -0.20 pct；
- 普通工况退化不超过 0.20 pct；
- 至少两个可评估 episode，且至少半数改善；
- 即使门禁通过，也先审查结果，不自动生成提交包。

该实验是对尚未隔离测试的 short 机制进行科学验证，不是对失败 long 模型换区块盲投。
