# 事件 onset hazard 诊断报告

## 1. 裁决

24 小时事件 hazard 未通过预登记门禁，不进入 long-g1 条件修正阶段。该路线不能解释或弥补
v29b 距离 65 的长周期缺口。

2 小时 onset 排名信号显著，只保留为未来 short-g1 独立实验假设；不得把本次整体失败包装成通过。

## 2. 运行信息

- run_id：`20261004T093404Z_event_onset_hazard_ce66daea3f`
- 角色：diagnostic
- 规模：4 个 episode 时序折 ×4 horizon ×2 模型，共 32 个逻辑回归
- 独立验证 episode：8 个
- 耗时：4.331 秒
- submission：未生成

## 3. Pooled 结果

| horizon | 模型 | prevalence | AP | AP lift | precision | recall | alert fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2h | calendar | 1.28% | 7.86% | 6.12 | 6.41% | 50.00% | 10.01% |
| 2h | process | 1.28% | 34.74% | 27.06 | 6.58% | 98.44% | 19.22% |
| 6h | calendar | 3.85% | 30.90% | 8.02 | 20.27% | 55.21% | 10.49% |
| 6h | process | 3.85% | 27.61% | 7.17 | 15.47% | 66.15% | 16.47% |
| 12h | calendar | 7.70% | 36.52% | 4.74 | 32.59% | 45.57% | 10.77% |
| 12h | process | 7.70% | 15.51% | 2.01 | 15.77% | 41.93% | 20.49% |
| 24h | calendar | 15.41% | 11.46% | 0.74 | 15.64% | 10.29% | 10.13% |
| 24h | process | 15.41% | 13.41% | 0.87 | 14.41% | 29.43% | 31.46% |

## 4. 门禁失败原因

- 24h AP lift 0.87，低于 1.5，且 AP 本身低于 prevalence。
- 相对 calendar 的 AP 只增加 0.0195，低于 0.02。
- precision lift 0.94，说明 process 告警并未富集真实 24h onset。
- recall 29.43%，低于 50%。
- process 只在 2/4 折超过 calendar。
- 训练 top-10% 阈值迁移到验证后产生 31.46% 告警，严重超过 15% 安全上限。

虽然 24h 检测到 7/8 episode、6h 检测 8/8，但这是大量告警换来的，不具备可执行精度。

## 5. 保留信号与停止条件

2h process 的 AP 0.347、ROC-AUC 0.975，证明过程变量对临近 onset 具有强排序信息；但阈值漂移仍使
alert fraction 达到 19.2%。它只能形成新的 short-only 假设，必须另行预登记并在 v29b short-g1
兼容控制上验证。

当前决定：

- 关闭 24h hazard-conditioned long-g1 修正；
- 不搜索 C、阈值、更多特征或更复杂分类器来挽救本 run；
- 不用 2h 的事后结果直接改提交；
- 下一主线转为 v28/v29b 的官方区间均值标签对齐实验。
