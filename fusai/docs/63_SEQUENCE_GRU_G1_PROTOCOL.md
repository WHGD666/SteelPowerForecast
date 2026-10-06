# Long G1 直接多步 GRU 实验协议

## 1. 实验定位

历史相似轨迹实验表明，单个起点的过程摘要不足以稳定决定未来 24 小时 `generator_1`
轨迹。本实验改用完整历史时间窗：让一个轻量单向 GRU 读取起点前 24 小时的因果过程演化，
一次直接输出未来 96 个官方 15 分钟区间均值。

本轮是 scientific OOF，不生成提交包。它与此前 LSTM/递归旧代码不是同一实验：不把模型预测
递归回填为目标历史，也不逐步自反馈；96 个 horizon 一次直接输出。

## 2. 冻结验证协议

继续使用历史相似轨迹实验的四个连续十日伪测试：

1. 2025-07-01 至 07-10；
2. 2025-08-01 至 08-10；
3. 2025-09-03 至 09-12；
4. 2025-09-19 至 09-28。

每个训练 origin 的 `t+1440` 完整区间必须在对应伪测试开始前结束。模型、特征缩放和目标
标准化均按块重新拟合；验证块目标在训练时不可见。已知不完整区间只在评分时排除，四个块仍
保持完整 960 个滚动起点。

控制组直接复用 run `20261005T080116Z_analog_trajectory_g1_4e4f828951` 中相同四块的
`shared_horizon_tree` OOF，加载前必须通过 manifest 和 SHA-256 校验。

## 3. 输入与信息边界

- 历史窗：96×15 分钟，即起点前 24 小时，包含起点当前状态；
- 过程特征：21 个预登记的生产、用户耗气、发电耗气、二号气柜和煤气平衡因果特征；
- 日历特征：每个历史步的小时 sin/cos、星期 sin/cos，共 4 个；
- 最终输入：96×25；
- 禁止目标：`generator_1`、`generator_all` 的真实历史；
- 禁止信息：预测起点之后的过程观测；
- 缺失填补和 median/IQR 缩放只用该折训练历史拟合。

## 4. 固定模型与训练

- 模型：两层单向 GRU；
- hidden size：96；
- dropout：0.10；
- LayerNorm：启用；
- 输出：一次输出 96 个 horizon；
- 参数量：100,800；
- 目标变换：每个 horizon 对正值取 log，再用训练侧均值和标准差归一化；
- 损失：标准化 log 目标上的 SmoothL1，beta=0.20；
- optimizer：AdamW，lr=0.001，weight decay=0.0001；
- epochs：30；batch size：256；gradient clip：1.0；
- seed：20261005；仅一个种子；
- 设备：CUDA，NVIDIA GeForce RTX 4060 Laptop GPU。

禁止扫描历史窗、hidden size、层数、dropout、epoch、损失函数、种子和学习率。

## 5. 候选与门禁

保存三个可比版本：

- `shared_horizon_tree`：冻结控制；
- `gru_direct`：诊断用独立 GRU；
- `blend_gru25_tree75`：唯一晋级候选。

固定融合为：

`0.25 × GRU + 0.75 × shared tree`

门禁：

- pooled long-g1 至少提升 0.8 个百分点；
- 最差十日块不退化；
- 至少 3/4 十日块改善；
- near 与 far 均不得低于控制 0.2 个百分点以上；
- 所有预测必须有限并位于 0.001–200 MW；
- 即使通过，本轮也不自动生成提交，先审查逐块、逐 horizon 和训练稳定性。

## 6. 已完成验证

- PyTorch 2.5.1、CUDA 11.8、RTX 4060 Laptop 可用；
- 6 项 GRU 专项测试通过；
- preflight 通过：4 块、最少 5474 个训练 origin、96×25 输入、96 输出；
- 模型参数量 100,800；
- 正确识别 96 个缺失验证标签单元；
- 控制 manifest、控制 OOF 哈希、行数与键均通过。

## 7. 正式运行命令

```powershell
cd D:\daima\aic\shiyan\fusai
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_sequence_gru_g1 `
  --config configs\round2_v3\experiments\sequence_gru_g1_v1.yaml
```

程序会训练 4 个 GPU GRU，保存每折模型、epoch loss、OOF、指标、门禁与不可覆盖 run manifest。

