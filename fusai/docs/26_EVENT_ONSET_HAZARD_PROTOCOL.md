# 事件 onset hazard 因果预警协议

## 1. 实验目的

此前过程特征、动态燃料和异常样本加权都证明“事件期间存在信号”，但没有证明模型能在事件开始前
做出可泛化预警。本实验先回答：只使用预测 origin 及历史过程量，能否对未参与训练的未来 episode
提前 2、6、12、24 小时预警？

本阶段只做二分类诊断，不改 v29b 预测，不估计发电修正幅度，也不生成提交包。

## 2. 标签与信息边界

对 origin `t` 和 horizon `h`，标签为：是否存在 episode start 满足 `t < start <= t+h`。
事件在 origin 已经开始时不计为 onset 正样本，避免把“已经发生”伪装成“提前预测”。

所有过程特征最大来源时间不超过 origin；不使用真实发电目标历史、未来过程观测或测试目标。
每折训练 origin 额外按 horizon 做标签成熟 embargo，保证训练标签窗口完全结束于验证起点之前。

## 3. 模型与对照

- control：5 个小时/星期日历特征。
- candidate：同一日历特征加 19 个预登记因果过程特征。
- 模型：固定 `C=0.1`、L2、class-balanced logistic regression。
- 阈值：每折仅根据训练概率的 top 10% 确定，不在验证 episode 上调阈值。
- 不搜索模型、C、阈值或特征子集。

## 4. Episode 时序折

按未来时间顺序分四折，每折留出两个完整 episode：

1. episode 06–07；
2. episode 08–09；
3. episode 10–11；
4. episode 12–13。

每折模型只能使用首次验证 episode 之前已经成熟的训练标签。这样共有 8 个真正未见 episode，
而不是把同一事件的相邻行随机分到训练和验证。

## 5. 通过条件

24 小时主任务必须同时满足：

- pooled average precision / prevalence ≥1.5；
- average precision 比 calendar-only 至少高 0.02；
- top-10% alert 的 precision lift ≥1.5、recall ≥50%；
- 至少预警 75% 的独立 episode；
- 至少 3/4 折的 AP 高于 calendar-only；
- 实际验证 alert 比例不超过 15%。

6 小时任务还必须达到 AP lift ≥1.5 且至少预警 50% episode。
只有全部通过，才进入“hazard × fold-local residual correction”阶段；否则关闭事件 hazard 路线。
