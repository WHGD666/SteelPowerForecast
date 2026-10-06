# v20 long-gall 共享 horizon 树实验协议

## 1. 目标

v29b long-g1 的共享树 v2 已获得 +0.6336 pct，但尚不足以单独覆盖 65 分缺口。本实验转向当前
87.54% 的 long `generator_all`，检验共享 horizon 是否能在不改变 v20 物理后处理的情况下提供独立收益。

## 2. 控制与唯一变化

控制在每个折内重建 v20：

- old-column/full-history 的 96 棵 point-label LightGBM；
- 九月两路发电耗气 OLS；
- 1–32 步采用 0.5 tree + 0.5 fuel；
- 33–96 步采用 0.225 tree + 0.075 origin-time climatology + 0.7 fuel；
- full-history fuel ratio 的 16/672 步因果门控，阈值 ±0.20、缩放 clip(0.80,1.20)。

候选只把 96 棵树替换为一个共享 horizon 树。旧列、全历史树窗、point 标签、模型参数、fuel、
climatology 和 gate 全部冻结。严禁使用 restored-column gall 血统。

## 3. 验证和门禁

- 三个近期 walk-forward 折，96 horizon；预计 291 个 CPU LightGBM。
- primary：官方区间均值；auxiliary：legacy point。
- 保存 fold、episode、ordinary、near/mid/far 指标。
- 不生成提交包。

组件门槛为 +0.30 pct；与已冻结 g1 积木组成候选要求至少 +0.40 pct。最差折不得为负，普通工况
损失不超过 0.20 pct，near/far 损失不超过 0.15 pct，多数 episode 改善。

## 4. 命令

代码、68 项 pytest 与只读 preflight 已通过。正式训练由人工执行：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.run_long_gall_shared_horizon --config configs\round2_v3\experiments\long_gall_shared_horizon_v1.yaml
```

预计训练 291 个 CPU LightGBM，不使用 GPU；按当前机器和既有实验速度，通常约 2–6 分钟。
runner 只生成 OOF、指标、manifest 和注册记录，不生成提交 ZIP。只有预登记门禁通过后，才允许进入
`g1 v2 + gall shared` 的组合候选构建。
