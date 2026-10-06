# v16/v29b short-g1 恢复字段实验报告

## 1. 裁决

恢复字段候选 pooled short-g1 提升 0.1024 pct，但没有达到预登记的 +0.15 pct 组件门槛，
并在 late-September 折退化 1.0467 pct。该变体不能替换 v29b short-g1，不生成提交包，
停止对其做无条件融合权重和 horizon 权重搜索。

## 2. 运行信息

- run_id：`20261004T102800Z_short_g1_restored_1fb7013867`
- 角色：scientific OOF
- 规模：3 个近期时序折 × 8 horizon × 2 个特征引擎，共 48 个 LightGBM
- 耗时：7.516 秒
- 控制：v16 legacy columns + 九月 point tree + 九月 fuel OLS + 20/80 融合
- 候选：只把 tree feature engine 切换为 v28 restored columns/full balance
- holdout：未评估
- submission：未生成

## 3. 总体与折间结果

| 版本 | MAPE | accuracy | 相对控制变化 |
|---|---:|---:|---:|
| v16 legacy columns | 0.115698 | 88.4302% | 0.0000 pct |
| restored columns | 0.114674 | 88.5326% | +0.1024 pct |

legacy point 辅助指标提升 0.0941 pct，方向一致，但幅度同样较小。

| 折 | 控制 accuracy | 候选 accuracy | 变化 |
|---|---:|---:|---:|
| BF3-active | 88.2048% | 88.9968% | +0.7920 pct |
| late September | 85.0803% | 84.0335% | -1.0467 pct |
| holder1-active | 90.8889% | 91.0677% | +0.1789 pct |

平均折增益为 -0.0253 pct；pooled 正收益来自不同折样本量和误差权重，不能掩盖中间折的系统性退化。

## 4. Horizon 与逐日诊断

15/30 分钟分别变化 -0.0644/-0.0026 pct；45–120 分钟均为正，变化约
+0.0868 至 +0.1832 pct。但 late-September 折在全部 8 个 horizon 上均退化，范围为
-1.2945 至 -0.5450 pct，因此只关闭近端 horizon 不能修复稳定性。

逐日收益为：9 月 3–5 日 +0.5041/+1.0485/+0.8234 pct，9 月 19–20 日
-0.8573/-1.2362 pct，9 月 25–27 日 +0.0452/+0.4881/+0.0034 pct。

只读诊断把起点过程量与每个起点的候选收益连接后，BF3 与 AH3 的线性相关仅约 0.058 和
-0.010；没有发现可直接预登记的稳定物理阈值。AH5 和高炉煤气耗量存在相关，但样本只有 8 个验证日，
在同一 OOF 上搜索阈值会构成事后选择，不允许据此制作门控提交。

## 5. 结论与停止条件

- `component_gate_passed=False`
- `standalone_submission_eligible=False`
- 不生成 ZIP，不消耗平台次数。
- 不继续搜索 restored/control 的 0.1–0.9 融合权重。
- 不按日期、折 ID 或验证误差做门控。
- 不把只关闭 15/30 分钟包装成新候选；它仍保留 late-September 全 horizon 退化。
- v29b short-g1 继续冻结为正式控制。

恢复字段确实包含信号，但当前数据不足以证明一个可外推的安全门控。下一项研究回到此前独立发现的
2h onset process 排序信号；它必须以 fold-local、只用起点及历史过程量的方式进入 short-g1，且不能使用
事件事后标签作为线上开关。
