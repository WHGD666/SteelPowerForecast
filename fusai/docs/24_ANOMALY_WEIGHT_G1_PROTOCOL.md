# g1 异常日样本加权隔离实验协议

## 1. 实验问题

平台最优 v29b 的 long `generator_1` 来自 v28。历史 v32–v34 给低燃料训练样本 ×3 权重时，
同时修改了 long-g1 燃料融合权重；v34b 的 62.9325 因而不能判断样本加权本身是否有效。

本实验只回答一个问题：

> 保持 v28 的字段、树窗口、LightGBM 参数、燃料回归以及实际部署的 0.8/0.6 近远融合完全不变，
> 仅对低燃料训练 origin 加权，是否能稳定改善 g1？

## 2. 唯一主要变化

异常定义沿用交接代码，但全部折内计算：

1. 在每个训练折中拟合 `generator_all ~ 高炉煤气耗量 + 转炉煤气耗量`；
2. 得到燃料隐含发电量，先做 16 步平滑，再除以 672 步历史基线；
3. 比值严格小于 0.8 的训练 origin 定义为低燃料样本；
4. control 权重为 1；唯一晋级候选权重为 3；2 和 5 仅用于稳定性灵敏度检查。

×3 已在看到本项目 OOF 结果前预登记。禁止从 ×2/×5 中按最好结果重新挑选候选。

## 3. 冻结项

- 目标：只评估 `generator_1`；不改变 `generator_all`。
- 树特征：精确复刻 v28 恢复列特征，不加入目标历史。
- g1 树窗口：2025-09-01 起，且每个 horizon 的标签必须在 fold train_end 前成熟。
- 模型：v28 LightGBM 参数与随机种子保持不变。
- 燃料回归：九月窗口、三种发电耗气、无正则最小二乘。
- 融合：源码实际语义为 15/30 分钟燃料权重 0.8，45–1440 分钟权重 0.6。
- 禁止：预测起点后的过程量、未来真实目标、平台反馈拟合和随机切分。

## 4. 验证口径

使用冻结时序折：

- `wf_03_bf3_active`
- `wf_04_late_september`
- `wf_05_holder1_active`

主指标使用官方 15 分钟区间均值标签；另保留 legacy 瞬时点标签作为辅助方向检查。
前两个普通月折不纳入，因为 v28 的 g1 树固定从九月开始训练，在七八月无法形成同配方的因果回测。

## 5. 门禁

×3 必须同时满足：

- long-g1 区间均值准确率至少提升 0.15 pct；
- 三个近期折的最差增益不低于 0；
- 普通工况损失不超过 0.10 pct；
- 多数独立事件改善；
- near/far 损失均不超过 0.10 pct；
- legacy 瞬时点 long-g1 不退化；
- short-g1 损失不超过 0.10 pct。

通过以上条件仅允许把它作为后续组合积木。只有 long-g1 提升至少 0.8 pct 才允许单独进入提交候选阶段。
本 runner 永远不直接生成 ZIP。

## 6. 运行命令

轻量预检：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_anomaly_weight_g1 --config configs\round2_v3\experiments\anomaly_weight_g1_v1.yaml --preflight-only
```

正式实验：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_anomaly_weight_g1 --config configs\round2_v3\experiments\anomaly_weight_g1_v1.yaml
```

规模为 3 折 × 96 horizon × 4 权重，共 1,152 个 CPU LightGBM。完成后先查看
`variant_comparison.csv`、`gate_results.csv` 和 `report.md`，不得只看 pooled 指标。
