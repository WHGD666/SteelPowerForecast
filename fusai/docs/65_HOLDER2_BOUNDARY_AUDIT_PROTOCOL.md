# 二号气柜边界状态诊断协议

## 1. 实验定位

Phase B 过程特征实验中，普通 holder_2 水平与简单差分作为主信号已被拒绝（事件稳定性差、
增益仅噪声量级）。D-077 关闭 GRU 路线后，唯一尚未单独验证的部分是：**折内估计的气柜
边界压力**——水位接近历史边界、持续贴边、多尺度逼近速度与煤气平衡状态的组合——是否与
shared-tree 控制在远端 horizon 留下的 `generator_1` 残差存在稳定关系。

本轮是 **diagnostic**：不训练任何预测模型，不生成提交包，只测量预登记特征与冻结控制
残差的关系。它是后续模型实验的先决门禁，不是模型实验本身。

## 2. 冻结验证协议

继续使用四个连续十日伪测试块（与 run `20261005T080116Z_analog_trajectory_g1_4e4f828951`
完全同构）：

1. 2025-07-01 至 07-10；
2. 2025-08-01 至 08-10；
3. 2025-09-03 至 09-12；
4. 2025-09-19 至 09-28。

每块 960 个 15 分钟起点，共 3840 个 origin。控制组为该 run 的 `shared_horizon_tree`
OOF（368,640 行），加载前必须通过 manifest 状态与 SHA-256 校验。

结果变量（按 origin 聚合控制 OOF）：

- `far_signed_percentage_residual`：h735–1440 各 horizon `(actual − prediction)/|actual|`
  的均值，即控制组远端偏差方向；
- `near_signed_percentage_residual`：h015–120 同口径；
- `far_minus_near_actual_ratio`：真实负荷远端相对近端的变化比例。

## 3. 边界特征合同（全部折内拟合）

输入只使用 event feature registry 中 5 个预登记因果列：`holder_2__level`、
`holder_2__delta_1h/4h/12h`、`balance_proxy_total__level`（最大来源偏移 0，无目标历史）。

对每个块，只用块开始前（≥14 天）的训练历史拟合：

- 参考边界：level 的 5%/95% 分位；贴边区：10%/90% 分位；
- 速度缩放：三个 delta 的 IQR（下限 1e-6），裁剪 ±3 后归一；
- 平衡状态：balance proxy 的中位数/IQR 标准化，同样裁剪；
- 贴边持续：高/低区连续步数，上限 96 步（24 小时）；
- 压力分数（固定权重，禁止搜索）：

```
0.45 × 居中位置 + 0.25 × 多尺度速度 + 0.20 × 有符号贴边 + 0.10 × 平衡状态
```

- 高/低压力组阈值：训练历史压力分数的 25%/75% 分位。

行 `t` 的特征只依赖 `≤ t` 的行（未来扰动单测保护）。

## 4. 诊断门禁（预登记，不回改）

全部满足才允许进入模型实验：

- pooled Spearman(压力分数, 远端残差) ≥ 0.08；
- ≥ 75% 十日块该 Spearman 为正；
- pooled 高压组−低压组远端残差差 ≥ 1.0 个百分点；
- ≥ 75% 十日块该差值为正；
- 每块高低两组覆盖率 ≥ 5%。

即使全部通过，本轮也不生成提交；模型实验需另行预登记。任一不满足则当场关闭
holder_2 边界路线，不回扫分位、分桶、权重或阈值。

## 5. 已完成验证

- 全项目 pytest：`104 passed`（含 4 项边界专项测试：合同取值、因果性扰动、
  压力分数方向、Spearman 与汇总方向）；
- preflight：`PASS blocks=4 origins=3840 features=5 models=0 minimum_history_rows=5664`；
- 事件特征 manifest：`target_history_included=False`、最大来源偏移 0、注册表 5 列全部因果；
- 控制 run 状态 `completed`，OOF 哈希、行数（4×960×96）、键唯一性通过。

## 6. 正式运行命令

```powershell
cd D:\daima\aic\shiyan\fusai
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.audit_holder2_boundary `
  --config configs\round2_v3\diagnostics\holder2_boundary_audit_v1.yaml
```

纯 CPU、零模型训练，预计数十秒内完成。输出 5 个工件（逐 origin 诊断、各块边界合同、
关系汇总、特征相关、门禁结果）并追加实验登记簿；`--preflight-only` 可随时重跑复验。
