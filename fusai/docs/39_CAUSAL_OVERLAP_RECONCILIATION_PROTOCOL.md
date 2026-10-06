# 因果重叠预测一致性实验协议

## 假设

滚动预测中，当前起点 `t` 的 horizon `h` 与上一起点 `t-15` 的 horizon `h+15` 指向同一个目标时刻。
官方答疑明确允许把前一预测起点已经输出的结果作为后续输入。因此可构造：

```text
final(t,h) = 0.50 × base(t,h) + 0.50 × final(t-15,h+15)
```

递推在每个 fold 开头或时间断点重置；最远 horizon 没有对应旧预测，保持 base。整个过程不读取未来过程量、
真实负荷或当前时刻之后的数据。

## 控制与选择规则

- g1 控制：v28 per-horizon `control_weight1` OOF；
- gall 控制：v20 per-horizon OOF；
- primary：long-g1，`alpha_current=0.50`；
- 0.25/0.75 仅为敏感性，不得事后改选；
- primary 至少 +0.30 pct、三折非负、ordinary/near/far 安全、多数 episode 改善，才允许制作提交。

## 意义

该路线与此前失败的共享 horizon 模型不同：不重新拟合目标关系，而是使用赛事允许的历史预测反馈，减少
同一目标时刻在连续起点上的预测抖动。若 primary 不过门禁，立即关闭，不扫描更多 alpha。

