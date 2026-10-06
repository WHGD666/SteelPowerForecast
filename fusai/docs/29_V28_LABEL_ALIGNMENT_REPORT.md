# v28/v29b g1 标签对齐实验报告

## 1. 裁决

官方 15 分钟区间均值标签不能替换 v28/v29b 现有的时间点标签。预登记 primary
`interval tree + interval fuel` 总体退化 0.0330 pct，三个近期折中两折退化，普通工况也退化；
不生成提交包，并停止均值窗口、标签偏移及混合比例搜索。

## 2. 运行信息

- run_id：`20261004T095143Z_v28_label_alignment_5b8c1cdab3`
- 角色：scientific OOF
- 规模：3 个近期时序折 × 96 horizon，共 288 个 LightGBM
- 耗时：100.438 秒
- 控制：legacy point tree + legacy point fuel
- 唯一晋级候选：interval-mean tree + interval-mean fuel
- holdout：未评估
- submission：未生成

## 3. 总体结果

| 版本 | MAPE | accuracy | 相对控制变化 |
|---|---:|---:|---:|
| point tree + point fuel | 0.133052 | 86.6948% | 0.0000 pct |
| point tree + interval fuel | 0.133175 | 86.6825% | -0.0123 pct |
| interval tree + point fuel | 0.133256 | 86.6744% | -0.0204 pct |
| interval tree + interval fuel | 0.133381 | 86.6619% | -0.0330 pct |

两项标签替换分别为负，组合后继续下降，没有互补性。primary 与组件晋级门槛
+0.5 pct 的差距超过一个数量级。

## 4. 稳定性与分段

primary 的近期三折变化为：

| 折 | control accuracy | primary accuracy | 变化 |
|---|---:|---:|---:|
| BF3-active | 87.4808% | 87.3742% | -0.1066 pct |
| late September | 84.3724% | 84.3477% | -0.0247 pct |
| holder1-active | 87.4572% | 87.4923% | +0.0351 pct |

普通工况相对损失 0.0746 pct；两个可评估 episode 仅改善一个。primary 在四个 horizon
bucket 上分别为 -0.0016、-0.0421、-0.0425、-0.0304 pct，近、中、远端没有形成可利用的
稳定正收益。

一个仅用于理解机制的细节是：`interval tree + point fuel` 在 15–120 分钟 bucket 的 accuracy
由 89.0483% 微升至 89.0533%，约 +0.0050 pct。该幅度远小于验证噪声，且不是预登记 primary，
不能据此事后选型；它只支持把下一研究范围限制在 short-g1，而不是继续改 long-g1 标签。

## 5. 结论与停止条件

- `component_gate_passed=False`
- `standalone_submission_eligible=False`
- 不生成 ZIP，不消耗平台次数。
- 不继续测试 5/10/30 分钟均值、标签平移或 point/interval 融合权重。
- 官方评价采用区间均值，不等于训练标签必须使用区间均值；现有高分点标签可能承担了相位补偿作用。
- v29b 的 long-g1 保持冻结，恢复 v29b 为唯一官方控制组。

下一步只隔离验证历史上尚未单独检验的 short-g1 恢复字段。若该实验也没有跨折稳定收益，
再考虑把已发现的 2h onset 排序信号作为 short-only 特征，而不是修改标签或长周期配方。
