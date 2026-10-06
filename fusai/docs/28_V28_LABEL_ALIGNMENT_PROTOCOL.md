# v28/v29b g1 官方区间均值标签对齐协议

## 1. 背景

官方答疑要求 15 分钟重采样采用区间均值，但 v28/v29b 的 g1 上游源码把训练目标写成
`origin + horizon` 对应的 15 分钟网格瞬时点。高分证明旧配方有效，却不证明该标签错位已经最优。

本实验保持 v28 的字段、九月窗口、LightGBM、燃料列和 0.8/0.6 权重不变，只把训练标签改成
官方 `[t+h-15, t+h)` 区间均值。

## 2. 版本

- control：point tree + point fuel，直接复用异常加权 run 中的 ×1 OOF。
- tree-only ablation：interval tree + point fuel。
- fuel-only ablation：point tree + interval fuel。
- 唯一晋级候选：interval tree + interval fuel。

两个 ablation 仅解释收益来源，禁止事后替代预登记候选。

## 3. 公平比较

- 三个最近 walk-forward 折保持不变。
- interval tree 使用与 legacy control 完全相同的训练 origin，仅替换标签值。
- point tree 从冻结 control prediction 和 fold-local point fuel 精确反解，不重新调参。
- interval fuel 使用相同九月训练 origin，将同一 origin 的三路耗气映射到该 origin 开始的完整 15 分钟均值。
- 所有标签必须在 fold train_end 前完整成熟。
- 不使用目标历史特征、未来过程量或平台反馈。

## 4. 晋级门禁

primary candidate 必须同时满足：

- long-g1 至少 +0.5 pct 才能成为组合积木；
- long-g1 至少 +0.8 pct 才能单独进入平台候选；
- 三个近期折最差增益不低于 0；
- 普通工况损失不超过 0.2 pct；
- 多数独立 episode 改善；
- near 与 far 损失均不超过 0.2 pct。

runner 不生成 ZIP。只有门禁通过后，才允许将 v29b long-g1 单区块替换并打包。

## 5. 运行命令

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_v28_label_alignment --config configs\round2_v3\experiments\v28_label_alignment_v1.yaml
```

规模为 288 个 CPU LightGBM，预计约为上一轮异常加权实验的四分之一到二分之一。
