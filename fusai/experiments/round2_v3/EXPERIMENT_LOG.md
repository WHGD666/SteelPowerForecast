# 实验日志

## `20260929T124109Z_r2v3_baseline_b19a28a9ec` — baseline 启动失败

- 日期：2026-09-29
- 角色：competition
- 结果：在训练前因缺失常量导入触发 `NameError`，无模型、无指标。
- 处理：保留失败 run 与 registry，补充回归测试后重新运行。

## `20260929T124936Z_r2v3_baseline_46e02afe85` — v3 baseline v1

- 日期：2026-09-29
- 角色：competition baseline
- 控制组：本 run 为首个 v3 控制组
- 假设：正确区间标签与因果过程历史可以形成可信的 v3 控制线
- 唯一主要变化：相对 legacy 重建了标签、特征、切分和指标合同
- 协议版本：round2_v3
- split 指纹：`0d567fd1ba9e422465316ac2d1daa5cb5424eafc094fa4076ae645a4ad5c578f`
- 运行命令：`python -m src.round2_v3.run_baseline --config configs/round2_v3/baseline_v1.yaml`
- 结果摘要：短期 1-MAPE 88.33%，长期 84.53%
- 分目标：short g1 84.98%，short gall 91.67%，long g1 84.46%，long gall 84.60%
- 稳定性与风险：七月和晚九月折波动大；树模型存在跨工况外推风险
- 结论：接受为控制组，不提交；进入层次目标受控实验
- 下一步：预测 `large_units=generator_all-generator_1`，与既有 gall 组合推导 g1

## `20260929T131221Z_r2v3_hierarchy_af50589323` — 层次目标分解

- 日期：2026-09-29
- 角色：competition
- 控制组：`20260929T124936Z_r2v3_baseline_46e02afe85`
- 唯一主要变化：预测 `generator_all-generator_1` 后反推 `generator_1`
- 结果摘要：短期 87.92%，长期 83.52%，均低于控制组
- 分目标：short g1 84.17%，long g1 82.45%；gall 完全复用控制组
- 物理投影：几乎不改变结果，说明退化来自目标表示而非边界违规
- 结论：拒绝，不进入提交候选
- 下一步：以 baseline v1 建立首次平台外部校准提交

## `20260929T212923_calibration_baseline_v1_91cd219d33` — v3 首次平台校准

- 日期：2026-09-29
- 角色：external calibration；不是新模型 run
- 来源 run：`20260929T124936Z_r2v3_baseline_46e02afe85`
- 假设：无已知泄漏且可复现的 v3 baseline 能提供可信的十月外部校准点
- 唯一主要变化：首次将 baseline v1 全量训练并提交平台
- ZIP SHA-256：`364893241bb5937599b18d1f39780b3dc1e968849672584b81f3e7eeaaccb35a`
- 平台结果：总分 48.9241；平台未返回分项
- 相对历史：比 legacy v9 的 52.4085 低 3.4844
- 结论：校准证据有效，但拒绝为高分候选；协议正确不代表十月工况泛化充分
- 下一步：停止同族重投，恢复 legacy 等价锚点，建立近期工况阻断指标并做单变量实验

## `20261001_external_best_61_pending_exact` — 外部高分候选接管审验

- 日期：2026-10-01
- 角色：external evidence；不是本地模型 run
- 来源：学弟提交 ZIP；用户报告平台约 61 分
- 精确成绩：待补，不填写虚构小数
- 产物身份：ZIP SHA-256 `78b0e4eb024e5cc1d058bb8a77a9ac506246304105bbdb070cedaaa91071b12f`
- 完整性：CRC、成员、行列、时间网格、有限值、物理边界和长短一致性全部通过
- 与 v3 差异：long `generator_all` 平均绝对差 29.81 MW，为最大结构变化
- 机制判断：九月燃料回归层已确认；底层树模型来源未知
- 复现状态：预测产物已冻结；现有 v15/v16 交接源码无法重建代表预测列，更新源码缺失
- 结论：接受为外部强控制组，不接受为已完全复现的正式 run
- 下一步：先取得 61 分版更新源码或生成记录并完成来源复现，再启动 `generator_1` 动态燃料受控实验

## `diag_20261001_external61_source_attribution_v0` — 61 分包来源最小复现

- 日期：2026-10-01
- 角色：diagnostic；只训练代表 horizon，不生成提交
- 目的：判断冻结 61 分包是否等于交接手册中的 v15 或 v16
- 环境：实际导入 Python 3.10.20、LightGBM 4.7.0、NumPy 2.1.3、pandas 2.2.3、scikit-learn 1.5.2、PyYAML 6.0.3
- 数据：八个官方原始文件 SHA-256 全部通过；旧版准备结果为 train 14,496×55、test 960×55、input 960×25
- 已确认：九月 g1 燃料系数 `[0.000134367550138083, 0.0011554463562695452, 0.00019412344433509634, -7.007563744598015]`
- 已确认：九月 gall 燃料系数 `[0.00036850873765689973, 0.0006810445094277805, -3.2028011474733042]`
- v16 九月树结果：g1 step1/96 MAE 0.5551/2.3899 MW；gall step1/33/96 MAE 8.2967/10.6686/9.8311 MW；每列 960 行全部不一致
- v15 全期树结果：g1 step1/96 MAE 1.2707/3.0047 MW；gall step1/33/96 MAE 7.2534/9.2207/8.6214 MW；每列 960 行全部不一致
- 扩展检查：7月、8月、8月中、9月及9月内多个 `train-start` 均未命中；全期树与九月树线性融合也未命中
- 结论：拒绝“当前包就是交接源码原样 v15/v16”的假设；现有证据只能复现燃料层，不能复现底层预测
- 下一步：向产出者索取 61 分版本的实际生成命令、变更代码、base-dir manifest 或完整输出目录；在此之前不运行 192 模型全量任务

## `diag_20261001_dynamic_fuel_g1_v0` — 动态燃料路径只读诊断

- 日期：2026-10-01
- 角色：diagnostic；未生成提交，不进入正式 run registry
- 控制组：`20260929T124936Z_r2v3_baseline_46e02afe85`
- 假设：当前燃料锚点若按 horizon 建模未来路径，可改善 `generator_1` 长短周期精度
- 诊断设置：最近 21 天窗口；当前值、1/4/8/16/96 lag、4/16/96 rolling、目标时刻周期；每个 horizon 独立线性映射；70% 动态燃料 + 30% v3 树
- 信息边界：标签末端必须不晚于训练截止时刻；使用冻结 v3 OOF；无平台反馈参与拟合
- 结果：short g1 84.9802% → 89.0738%，改善 4.0936 pct
- 结果：long g1 84.4625% → 88.5275%，改善 4.0650 pct
- 风险：八月 long 和 BF3-active 折轻微退化；诊断尚未工程化、未形成不可覆盖正式产物
- 结论：接受为下一项正式实验假设，不接受为提交候选或官方成绩
- 下一步：正式实现后复跑相同冻结切分；冻结 61 版 `generator_all`，只评估 `generator_1` 变化

## `audit_20261001_latest_optimal_bundle` — 最新最优版静态接管审阅

- 日期：2026-10-01
- 角色：external evidence audit；未运行训练、预测或打包
- 输入：`给学长_最优版整合(1).zip`
- 外层 ZIP SHA-256：`350ca90b8db4f033cb6c8772d04ff374a11e7fba8c05a92971172aa748965cce`
- 来源结论：冻结高分 CSV 精确命中包内 v28；交接材料记录官方总分 61.5824
- v28 CSV：short `e96e6338...`，long `3c6e55d4...`，均与此前冻结参考完全一致
- v28 机制：恢复 BF3/AH3/AH5/converter_user1；g1 九月树、gall 全期树；燃料融合；实际 no-gate
- 实现风险：`h_step <= 32` 判断的是分钟，因此所谓近端只包含 15/30 分钟；manifest 与代码权重/gate 描述不一致
- v29b：未提交，63.45 仅为分项推算；依赖 v20/v16_final/v29 中间输出，交接包未形成独立生成闭环
- v34：未提交；gall 与 v28 完全相同，只改变 g1，具备单组件归因结构
- 结论：更新官方控制组为 `external_v28_61_5824`；本轮停止于学习和记录，不执行复现
- 详细报告：`docs/15_LATEST_OPTIMAL_BUNDLE_REVIEW.md`

## `20261002T065908Z_component_ladder_sg1w000_lg1w000_f16bf5aa2d` — v29b 平台锚点候选

- 日期：2026-10-02
- 角色：competition candidate；尚未提交平台
- 控制组：`external_v28_61_5824`
- 假设：v29b 汇总的历史强分项能把整体成绩从 61.5824 推向约 63 分区间
- 唯一主要变化：相对 v28 同时采用 v29b 已冻结的组合成品；本 run 作为后续单组件阶梯的零权重锚点
- short/long g1 v34 权重：0 / 0
- short CSV SHA-256：`63c1ed03a9549ec8aeb83012f21edab221dcc74ed6a459e927e2d319fa92be76`
- long CSV SHA-256：`088282fc08e9b9ac568983f846fa65ef26c748de18576eae1cd50107d0984da1`
- ZIP SHA-256：`7025afedf9a45777f5e0ad32c3db13048ea18e06b955a5c845507fb39c8689df`
- 验证：schema、时间、有限值、物理边界、来源哈希和 ZIP 成员全部通过
- 结论：可作为 Slot 1 上传候选；63.45 仍是预估，不是本 run 的官方成绩

## `20261002T065908Z_component_ladder_sg1w000_lg1w050_f16bf5aa2d` — 只改 long g1 的 50% 探针

- 日期：2026-10-02
- 角色：competition candidate；尚未提交平台
- 控制组：`20261002T065908Z_component_ladder_sg1w000_lg1w000_f16bf5aa2d`
- 唯一主要变化：long g1 = 50% v29b + 50% v34；short 全部列和 long gall 逐值冻结
- short/long g1 v34 权重：0 / 0.5
- long g1 相对锚点平均绝对变化：3.163421 MW
- short CSV SHA-256：`63c1ed03a9549ec8aeb83012f21edab221dcc74ed6a459e927e2d319fa92be76`
- long CSV SHA-256：`eccde8d56b192f9a46c5790185b8542e6ca02da0fd007e6d6be418f7c8916751`
- ZIP SHA-256：`ae17408681c5cef4ce032dc89af2fdfac8da978251030cdc90ed67fe2b31d3d9`
- 验证：long gall 逐值冻结；schema、时间、有限值、物理边界和 ZIP 成员全部通过
- 决策门禁：只有与锚点的官方总分比较后，才决定下一步测试 long 权重 1.0 或 0.25

### 2026-10-04 追记

- 新官方证据显示 full v34 long g1 端点已经以 v34b 提交：62.9325，低于 v29b 63.4621；long g1 82.28% 低于 82.96%。
- 本 50% 探针没有提交平台，现标记为 `rejected_before_submission`。
- 原计划的 0.25/1.0 权重扩展取消，避免重复消耗提交次数。

## `audit_20261004_top3_and_full_trial_archive` — 最高三包与近 50 版试错审阅

- 日期：2026-10-04
- 角色：external evidence audit；未运行训练、预测或平台提交
- 输入：`分最高三包_交底文档.zip`
- 外层 ZIP SHA-256：`564fa85fd07b5e5e3c3d73b066a65246d6a7a33e1bbeeeff8e565a13d824c4c`
- 官方最高：v29b 63.4621；ZIP `29756919...`，short `63c1ed03...`，long `088282fc...`
- 组件：short g1 89.47、short gall 93.74、long g1 82.96、long gall 87.54
- 关键否决：v34b 62.9325 否定 v34 long g1；v37 63.1041 否定更深门控；v48 63.3864 表明 0.9 权重无实质增益
- 安全边界：恢复列 gall 的 v35 和 9 月下半月燃料窗 v45 均触发 `cheat_detected`，两条血统永久封存
- 用户补充：接近 50 版后仍未超过 v29b；无精确读数的后续版本不补造分数
- 决定：v29b 升级为官方控制组；取消尚未上传的 v29b×v34 权重阶梯
- 下一步：先做台阶日前兆和目标时刻电价/气柜状态的因果诊断，再决定新模型实验
- 详细报告：`docs/17_TOP3_AND_50_TRIALS_REVIEW.md`

## `20261004T052259Z_step_event_audit_0057a685f6` — 台阶事件与因果前兆审计

- 日期：2026-10-04
- 角色：diagnostic；未训练正式模型、未生成提交
- 控制组：`external_v29b_63_4621`
- 假设：低燃料事件发生前 2–24 小时可能存在可用于 long-g1 的因果过程前兆
- 运行命令：`python -m src.round2_v3.audit_step_events --config configs/round2_v3/diagnostics/step_event_audit_v1.yaml`
- 复现：学弟口径 24 个事件日精确命中；合并为 13 个独立 episode
- 测试诊断：同口径命中 10 月 1 日、10 月 10 日；未使用测试目标
- g1 结果：事件后 2h 中位变化 -4.542 MW，6h -5.718 MW，12h -4.131 MW，24h +4.530 MW
- 分型：2h/6h 同时下降至少 5 MW 的事件只有 6/13
- 前兆：2h 高炉产量与发电耗气最强；4–6h holder_2、高炉生产/平衡和耗气波动仍有候选信号；12–24h 普遍较弱
- 信息边界：224 个描述特征中 196 个测试时可用；真实目标历史 28 个特征全部标记 diagnostic-only
- 结论：Phase A 条件通过；不训练独立事件门控器，转入预登记的小型 process-only 特征组消融
- 详细报告：`docs/19_STEP_EVENT_AUDIT_REPORT.md`

## `20261004T062435Z_event_signal_ablation_3cc01b0f7b` — process-only long-g1 特征组消融

- 日期：2026-10-04
- 角色：scientific OOF；不生成提交
- 控制组：baseline v1 的 long `generator_1` OOF；与原控制 129,024 条记录逐值一致
- 假设：事件前 2–6 小时的高炉生产、发电耗气、holder_2、煤气平衡和用户变化能稳定改善 long-g1
- 唯一主要变化：分别向完全相同的控制模型加入一个预登记过程特征组；另设一个 compact union
- 协议：5 个冻结时序折，7 个变体，共 35 个 LightGBM；196 个 process-only 因果特征，最大来源偏移 0
- 运行命令：`python -m src.round2_v3.run_event_signal_ablation --config configs/round2_v3/experiments/event_signal_ablation_v1.yaml`
- 最佳结果：高炉生产组 84.8420%，相对控制 84.4625% 提升 0.3795 pct
- 次优：煤气平衡 +0.3647 pct；用户/混合煤气 +0.3275 pct
- 稳定性：最佳组仅改善 2/4 个 episode；收益主要由 BF3-active 折贡献；compact union 在 late-September 折退化 1.2411 pct、在 episode_12 退化 4.1435 pct
- 门禁：`promotion_passed=[]`；未评估 holdout；`submission_eligible=false`
- 结论：拒绝为竞赛候选；高炉生产只保留为弱辅助信号，不直接接入 v29b
- 下一步：正式工程化并同协议验证 horizon-specific 动态燃料路径；首轮不混入其他新特征
- 详细报告：`docs/20_EVENT_SIGNAL_ABLATION_REPORT.md`

## `20261004T070858Z_dynamic_fuel_g1_31e2a08193` — 21 天 horizon-specific 动态燃料

- 日期：2026-10-04
- 角色：scientific OOF；不生成提交
- 控制组：`20260929T124936Z_r2v3_baseline_46e02afe85` long-g1 OOF
- 唯一主要变化：每个 horizon 独立使用最近 21 天三路发电耗气历史拟合 OLS；固定 70% dynamic + 30% control
- 规模：5 折 × 96 horizon = 480 个 OLS；27 个燃料特征、36 个模型输入；耗时 91.12 秒
- 总体结果：control 84.4625% → candidate 87.8300%，提升 3.3675 pct
- horizon：near +3.4838、mid1 +1.9864、mid2 +2.5134、far +4.2356 pct
- 正向折：July +5.8826、late September +8.4986、Holder1-active +6.6224 pct
- 失败折：August -1.0513、BF3-active -1.4043 pct
- 工况：pre-event +6.7675、active +6.4504、recovery +5.4546、ordinary -1.6419 pct
- 门禁：普通工况和近期最差单折失败，`promotion_passed=[]`
- 结论：保留为强机制证据，拒绝当前版本提交；不把 baseline OOF 增益直接外推到 v29b
- 下一步：只将窗口从 21 天扩大到 56 天，复验跨工况稳定性与窄九月窗口风险
- 详细报告：`docs/22_DYNAMIC_FUEL_G1_V1_REPORT.md`

## `20261004T072315Z_dynamic_fuel_g1_5ad785fc3f` — 56 天动态燃料稳定性复验

- 日期：2026-10-04
- 角色：scientific OOF；不生成提交
- 控制组：`20260929T124936Z_r2v3_baseline_46e02afe85` long-g1 OOF
- 唯一主要变化：相对 21 天 run，只把 fold-local 历史窗口扩大到 56 天；模型、特征、70/30 融合和切分不变
- 总体结果：control 84.4625% → blend 84.7708%，仅提升 0.3083 pct；standalone dynamic OLS 退化 0.3767 pct
- horizon：near +0.8995、mid1 +0.0580、mid2 -0.0091、far +0.4519 pct
- 稳定性：普通工况 -5.7747 pct，最近三折均值 -1.5309 pct，最近最差折 -6.9628 pct；仅 2/4 episode 改善
- 门禁：总体、普通工况、近期、事件和最差单折均未达到晋级条件，`promotion_passed=[]`
- 结论：扩大窗口没有修复工况不稳定，拒绝 56 天版本；按预登记停止条件关闭全时段 dynamic-fuel 替换路线
- 平台旁证：21 天高风险探针只替换 long-g1 后官方 62.8070，相对 v29b 下降 0.6551
- 下一步：恢复 v29b 为唯一官方控制组，转向未被混杂验证的 g1 异常日样本加权和事件 hazard 研究

## `20261004T092221Z_anomaly_weight_g1_e6563a0b55` — v28 g1 异常样本加权

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 控制：折内重建 v28 g1；权重 ×1
- 唯一主要变化：折内燃料比 <0.8 的 g1 树训练 origin 权重 ×3
- 灵敏度：×2/×5 只作方向检查，不参与 OOF 后选优
- 规模：3 折 ×96 horizon ×4 权重 =1,152 个模型；耗时 248.467 秒
- 主结果：×3 long-g1 +0.0176 pct、short-g1 -0.0523 pct、legacy point long +0.0188 pct
- 分折：BF3-active 0.0000、late September +0.0950、holder1-active -0.0163 pct
- 事件与 horizon：1/2 episode 改善；near -0.0523、far +0.0360 pct
- 门禁：`component_gate_passed=False`；`standalone_submission_eligible=False`
- 结论：拒绝并停止权重/阈值搜索；转入事件 onset hazard 诊断
- 报告：`docs/25_ANOMALY_WEIGHT_G1_REPORT.md`

## `20261004T093404Z_event_onset_hazard_ce66daea3f` — 因果事件 onset hazard

- 日期：2026-10-04
- 角色：diagnostic；未修改 v29b，未生成提交
- 设计：4 个时序 episode 折，留出 8 个完整 episode；2/6/12/24h；calendar vs calendar+19 process
- 模型：固定 L2 logistic；阈值仅由每折训练概率 top 10% 决定
- 24h：process AP 0.1341，低于 prevalence 0.1541；AP lift 0.87，alert 31.46%，只赢 2/4 折
- 6h：process AP 0.2761，低于 calendar 0.3090
- 2h：process AP 0.3474、AP lift 27.06、ROC-AUC 0.9746，但 alert 19.22%
- 门禁：`promotion_gate_passed=False`
- 结论：关闭 24h long hazard；2h 仅保留为 short-only 新假设
- 报告：`docs/27_EVENT_ONSET_HAZARD_REPORT.md`

## `20261004T095143Z_v28_label_alignment_5b8c1cdab3` — v28 g1 标签对齐

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 控制：legacy point tree + legacy point fuel
- 唯一主要变化：tree 与 fuel 的训练标签同时改为官方 15 分钟区间均值
- 消融：只改 tree、只改 fuel 仅用于归因，不允许事后改选
- 规模：3 个近期时序折 ×96 horizon =288 个 LightGBM；耗时 100.438 秒
- primary：accuracy 86.6948% → 86.6619%，退化 0.0330 pct
- 消融：tree-only -0.0204 pct；fuel-only -0.0123 pct
- 分折：BF3-active -0.1066、late September -0.0247、holder1-active +0.0351 pct
- 分段：near/mid1/mid2/far 全负；普通工况损失 0.0746 pct；仅 1/2 episode 改善
- 门禁：`component_gate_passed=False`；`standalone_submission_eligible=False`
- 结论：拒绝；冻结旧点标签；不继续搜索均值窗口、时间偏移或混合权重
- 下一步：只在 short-g1 隔离恢复字段，保持 v29b 其余三个区块冻结
- 报告：`docs/29_V28_LABEL_ALIGNMENT_REPORT.md`

## `planned_short_g1_restored_columns_v1` — v16/v29b short-g1 恢复字段隔离

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待人工运行
- 控制：v16 legacy feature engine + 九月 point tree + 九月 fuel OLS + 20/80 融合
- 唯一主要变化：tree feature engine 改为 v28 restored columns/full balance
- 冻结：short-gall、long-g1、long-gall、标签、模型参数与 fuel 公式
- 协议：最近三个时序折 ×8 short horizon ×2 变体，共 48 个 CPU LightGBM
- 验证：65 项 pytest 与只读 preflight 已通过；legacy 53 features，restored 52 features
- 输出纪律：runner 不生成提交包；门禁通过后才允许制作单区块候选
- 协议：`docs/30_SHORT_G1_RESTORED_COLUMNS_PROTOCOL.md`

## `20261004T102800Z_short_g1_restored_1fb7013867` — v16/v29b short-g1 恢复字段

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 唯一主要变化：tree feature engine 从 v16 legacy columns 切换为 v28 restored columns/full balance
- 冻结：九月 point tree、九月 fuel OLS、20/80 融合、short-gall 与全部 long
- 规模：3 折 ×8 horizon ×2 变体 =48 个 LightGBM；耗时 7.516 秒
- pooled：official short-g1 +0.1024 pct；legacy point +0.0941 pct
- 分折：BF3-active +0.7920、late September -1.0467、holder1-active +0.1789 pct
- horizon：15/30 分钟略负，45–120 分钟为正；late-September 全 8 步均负
- 门禁：`component_gate_passed=False`；`standalone_submission_eligible=False`
- 结论：拒绝，不打包，不继续融合权重或日期门控搜索
- 报告：`docs/31_SHORT_G1_RESTORED_COLUMNS_REPORT.md`

## `planned_long_g1_shared_horizon_v1` — v28 long-g1 共享 horizon 树

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待人工运行
- 控制：冻结 anomaly-weight run 的 v28 ×1 per-horizon OOF
- 唯一主要变化：96 棵独立树替换为一个共享 horizon 树，并加入必要的 horizon/目标时刻编码
- 冻结：九月窗、point 标签、restored 特征、LightGBM 参数、0.8/0.6 fuel 与其余三个提交区块
- 规模：三个近期折，共 3 个 CPU LightGBM；不生成提交包
- 验证：66 项 pytest、点标签 `t+h` 边界测试与只读 preflight 已通过；52 个 origin 特征、63 个模型特征
- 门禁：long-g1 +0.5 pct；最差折、普通工况、episode、near/far 全部安全；+0.8 pct 才可单区块提交
- 协议：`docs/32_LONG_G1_SHARED_HORIZON_PROTOCOL.md`

## `20261004T105317Z_long_g1_shared_aa987f06b5` — v28 long-g1 共享 horizon v1

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 唯一主要变化：96 棵独立树替换为一个共享 horizon 树
- 规模：3 个模型；训练行数 13,776/161,232/216,528；耗时 14.985 秒
- pooled：official +0.3597 pct；legacy point +0.3617 pct
- 分折：BF3-active -0.3840、late September +1.0943、holder1-active +0.6138 pct
- 分段：near/mid1/mid2/far 全正；2/2 episode 改善；普通工况损失 0.1541 pct
- 门禁：总体未到 +0.5 且最差折为负；`component_gate_passed=False`
- 结论：v1 不打包；只做一次 8–9 月树窗稳定性复验
- 报告：`docs/33_LONG_G1_SHARED_HORIZON_V1_REPORT.md`

## `planned_long_g1_shared_horizon_v2_augsep` — 共享 horizon 8–9 月树窗复验

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待人工运行
- parent：`20261004T105317Z_long_g1_shared_aa987f06b5`
- 唯一变化：tree start 2025-09-01 → 2025-08-01；fuel fit 继续为九月
- 规模：3 个 CPU LightGBM；不生成提交包
- 验证：66 项 pytest 与只读 preflight 已通过
- 停止条件：任一负折或 pooled <+0.5 pct 时关闭共享树窗口研究
- 协议：`docs/34_LONG_G1_SHARED_HORIZON_V2_PROTOCOL.md`

## `20261004T122856Z_long_g1_shared_85d58617c3` — v28 long-g1 共享 horizon v2

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 唯一主要变化：相对 v1 将 tree start 从 2025-09-01 扩至 2025-08-01；fuel fit 继续冻结为九月
- 规模：3 个共享模型；训练行数 299,472/446,928/502,224；耗时 24.106 秒
- pooled：official long-g1 +0.6336 pct；legacy point +0.6362 pct
- 分折：BF3-active +0.1911、late September +2.0119、holder1-active +0.1571 pct
- 分段：near/mid1/mid2/far +0.5102/+0.6731/+0.5266/+0.6944 pct；ordinary +0.3845 pct
- 事件：episode 12 +2.8225 pct，episode 13 -0.5877 pct，改善比例 1/2
- 结论：冻结为 `promising_component_not_standalone`；不做事后事件门控，不继续扫描树窗或融合权重
- 报告：`docs/35_LONG_G1_SHARED_HORIZON_V2_REPORT.md`

## `planned_long_gall_shared_horizon_v1` — v20 old-column long-gall 共享 horizon 树

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待人工运行
- 控制：折内重建 v20 的 96 棵 old-column point-label 树，冻结 fuel、origin-time climatology 与 causal fuel-ratio gate
- 唯一主要变化：每折 96 棵独立 gall 树替换为一个共享 horizon gall 树
- 安全边界：不使用 restored-column gall；v35 `cheat_detected` 血统永久排除
- 规模：三个近期时序折，288 个控制模型 +3 个共享模型；不生成提交包
- 验证：68 项 pytest、静态编译与只读 preflight 已通过；old 53 features，shared 64 features，`restored_gall_columns=false`
- 门禁：组件至少 +0.30 pct；组合候选至少 +0.40 pct；分折、普通工况、episode 与 near/far 同时受限
- 协议：`docs/36_LONG_GALL_SHARED_HORIZON_PROTOCOL.md`

## `20261004T125203Z_long_gall_shared_c83510959a` — v20 long-gall 共享 horizon 树

- 日期：2026-10-04
- 角色：scientific OOF；未直接生成提交
- 唯一主要变化：每折 96 棵 old-column gall 树替换为一个共享 horizon gall 树
- 规模：291 个模型；共享训练行 1,128,912/1,276,368/1,331,664；耗时 289.32 秒
- pooled：official +0.3911 pct；legacy point +0.3990 pct
- 分折：+0.0654/+0.7779/+0.4589 pct；2/2 episode 改善；ordinary +0.1252 pct
- 分段：near -0.2304、mid1 +0.7628、mid2 +0.4683、far +0.3322 pct
- 门禁：component gain、fold、ordinary、episode 通过；near/far 与 +0.40 组合阈值失败
- 结论：不自动晋级；按用户授权只制作 long-gall 单区块平台探针，不与 g1 v2 同包
- 报告：`docs/37_LONG_GALL_SHARED_PLATFORM_PROBE.md`

## 记录模板

## `20261004T144354Z_overlap_reconcile_1a82967010` — 因果重叠预测一致性

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 唯一变化：当前预测与上一滚动起点对同一目标时刻的已发布预测递推融合，primary α=0.50
- g1：+0.0972 pct；三折 +0.0906/+0.1262/+0.0845；四段全正；2/2 episode 改善
- gall：+0.0999 pct；三折全正，但 near -0.1549 pct
- 敏感性：g1 α=0.25 +0.2209、α=0.75 +0.0428 pct；均不具事后晋级资格
- 门禁：`promotion_eligible=[]`
- 结论：拒绝，不打包，不扫描更多递推权重；转入十月相似度加权验证
- 报告：`docs/40_CAUSAL_OVERLAP_RECONCILIATION_REPORT.md`

## `20261004T145819Z_test_similarity_d7b3a8efa8` — 十月测试相似度校准

- 日期：2026-10-04
- 角色：diagnostic；未训练预测模型、未生成提交
- 唯一变化：用冻结的 process-only 域概率对三个既有平台探针的 OOF 重新加权
- 域分类：158 个特征，grouped-CV AUC 0.951865；最小 ESS fraction 0.732631
- 加权结果：dynamic fuel +3.9937、shared g1 +0.6566、shared gall +0.4072 pct
- 平台方向：实际分别为 -0.6551、-0.3577、-0.0082 分，三项方向校准全部失败
- 门禁：`calibration_gate_passed=False`
- 结论：拒绝并关闭该验证口径；不做事后权重/阈值搜索，不生成 ZIP
- 报告：`docs/42_TEST_SIMILARITY_CALIBRATION_REPORT.md`

## `planned_long_g1_mape_objective_v1` — v28 long-g1 MAPE 对齐训练

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待人工运行
- 控制：冻结 v28 MAE 树 OOF；不重新选择控制
- 唯一变化：树训练样本权重固定为 mean-one `1/max(abs(y), 20 MW)`
- 冻结：树窗、标签、特征、模型参数、fuel OLS、融合权重、切分和指标
- 规模：288 个 CPU LightGBM；不生成提交包
- 门禁：long-g1 +0.30 pct 才保留，+0.80 pct 且全部安全门禁通过才可考虑平台候选
- 协议：`docs/43_LONG_G1_MAPE_OBJECTIVE_PROTOCOL.md`

## `20261004T151651Z_long_g1_mape_00babc641a` — v28 long-g1 MAPE 对齐树

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 唯一变化：树训练使用 mean-one `1/max(abs(y), 20 MW)` 样本权重
- pooled：long-g1 -0.0997 pct；short 投影 -0.0971 pct
- 分折：-0.4558/+0.1276/+0.1049 pct；ordinary -0.3258 pct
- horizon：near/mid1/mid2/far 全部为负；episode 2/2 改善
- 门禁：`component_gate_passed=False`，`standalone_submission_eligible=False`
- 结论：拒绝并关闭树目标重加权；不做权重或下限扫描，不生成 ZIP
- 报告：`docs/44_LONG_G1_MAPE_OBJECTIVE_REPORT.md`

## `planned_long_g1_mape_fuel_v1` — v28 long-g1 MAPE 对齐 fuel proxy

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待运行
- 唯一变化：九月 g1 fuel OLS 改为 inverse-target weighted median regression
- 冻结：控制树、特征、标签、树窗、horizon 权重、切分和指标
- 固定：quantile=0.5、alpha=0、solver=highs、denominator floor=20 MW
- 规模：3 个燃料线性模型，0 个 LightGBM；不生成提交包
- 门禁：long-g1 +0.30 pct 才保留，+0.80 pct 且全部安全门禁通过才考虑平台
- 协议：`docs/45_LONG_G1_MAPE_FUEL_PROTOCOL.md`

## `20261004T152405Z_long_g1_mape_fuel_20b0cc90f0` — v28 long-g1 MAPE fuel

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- 唯一变化：九月 fuel OLS 替换为 inverse-target weighted median regression
- pooled：long-g1 +0.1085 pct；short 投影 -0.0852 pct
- 分折：+0.0189/+0.6660/-0.1735 pct；ordinary +0.1018 pct
- horizon：near 为负，mid/far 为正；2/2 episode 改善
- 门禁：总体未到 +0.30 且最差折为负；`standalone_submission_eligible=False`
- 结论：拒绝，不做参数或窗口扫描，不生成 ZIP
- 报告：`docs/46_LONG_G1_MAPE_FUEL_REPORT.md`

## `planned_holiday_effect_audit_v1` — 节假日燃料残差跨块迁移

- 日期：2026-10-04
- 角色：preregistered diagnostic；待运行
- 假设：控制三类发电耗气后，节假日 g1 调度比例在劳动节与端午节间可迁移
- 设计：非节假日 OLS 基线；两个节假日块互为 calibration/validation
- 门禁：双向正增益、均值 +0.50 pct、scale 方向一致且距离 ≤0.05
- 输出：只生成诊断表和 manifest，不训练预测模型、不生成 ZIP
- 协议：`docs/48_HOLIDAY_EFFECT_AUDIT_PROTOCOL.md`

## `20261004T153231Z_holiday_audit_75f9b001b3` — 节假日燃料残差跨块迁移

- 日期：2026-10-04
- 角色：diagnostic；不生成提交
- 两块最优 scale：0.908385 / 0.951977
- 双向迁移增益：+0.2127 / +3.9076 pct，平均 +2.0601 pct
- 门禁：方向、均值、同侧、scale 距离和样本数全部通过
- 结论：允许一个固定 0.95 的 long-g1 fuel component 候选；不扫描幅度
- 报告：`docs/49_HOLIDAY_EFFECT_AUDIT_REPORT.md`

## `20261004T234025_holiday_long_g1_probe_v1_239cbda25b` — 国庆 long-g1 固定 0.95 探针

- 日期：2026-10-04
- 角色：competition probe；已上传并判负
- 控制：v29b 63.4621
- 来源诊断：`20261004T153231Z_holiday_audit_75f9b001b3`，门禁通过
- 唯一变化：目标区间起点位于 10 月 1–8 日的 long-g1 fuel component 固定乘 0.95
- 冻结：short 原始字节、long-gall、非假日 long-g1
- 变更：69,168 个单元，平均 3.7472 MW，最大 5.7348 MW；无物理投影触发
- 验收：82 项 pytest；独立 verifier 通过；ZIP SHA-256 `380134a05ad54ce532b92f7cf958ba1dd86df0859a71a7bd16bc40c7880425e0`
- 官方结果：63.3013，相对 v29b 63.4621 为 -0.1608
- 结论：拒绝并关闭节假日修正路线；禁止 scale、日期边界和分 horizon 幅度扫描
- 报告：`docs/51_HOLIDAY_LONG_G1_PROBE_REPORT.md`

## `planned_short_g1_dynamic_fuel_v1` — v16 short-g1 动态燃料隔离

- 日期：2026-10-04
- 角色：preregistered scientific OOF；待用户运行
- 控制：冻结 v16-lineage short-g1 OOF
- 唯一变化：70% horizon-specific causal dynamic-fuel OLS + 30% v16 control
- 规模：3 折 ×8 horizon = 24 个 OLS；不生成提交
- 门禁：short-g1 +1.0 pct，折/horizon/普通工况/episode 全部安全
- 协议：`docs/52_SHORT_G1_DYNAMIC_FUEL_PROTOCOL.md`

## `20261004T155802Z_short_dynamic_fuel_87286d8028` — v16 short-g1 动态燃料隔离

- 日期：2026-10-04
- 角色：scientific OOF；未生成提交
- pooled +1.4365 pct；全部 horizon 为正；普通工况 +2.4547 pct
- 分折 +3.0297/+6.1319/-3.2870 pct；最差折灾难性失稳
- 事件勘误后为 2 个、1 胜1负；初始控制台的 3 个包含空 id
- 门禁：`component_gate_passed=False`
- 结论：原始候选拒绝；允许唯一 ±5 MW 安全限幅诊断
- 报告：`docs/53_SHORT_G1_DYNAMIC_FUEL_REPORT.md`

## `planned_short_g1_dynamic_fuel_clip5_v1` — 固定 ±5 MW 安全限幅

- 日期：2026-10-04
- 角色：preregistered scientific OOF；轻量后处理
- 唯一变化：冻结 70/30 源 OOF，将相对 v16 控制修正裁剪到 [-5,+5] MW
- 不重训模型，不搜索 cap，不生成提交
- 协议：`docs/54_SHORT_G1_DYNAMIC_FUEL_CAP_PROTOCOL.md`

## `20261004T160243Z_short_dynamic_cap_896d8696e5` — 固定 ±5 MW 安全限幅

- 日期：2026-10-05
- 角色：scientific OOF；未生成提交
- pooled +1.4950 pct；最差 horizon +1.3971 pct；普通工况 +1.9267 pct
- 分折 +2.0029/+3.3557/-0.2535 pct；最差折低于 -0.20 安全线
- 事件：2 个、1 胜1负；4,327/6,144 个源修正被限幅
- 门禁：`component_gate_passed=False`
- 结论：不打包，不放宽门槛，不扫描 cap/blend，关闭路线
- 报告：`docs/55_SHORT_G1_DYNAMIC_FUEL_CAP_REPORT.md`

## `planned_short_g1_fuel_trend_v1` — 一小时燃料趋势外推

- 日期：2026-10-05
- 角色：preregistered scientific OOF；轻量机理实验
- 唯一变化：v16 short-g1 叠加最近一小时 fuel proxy 阻尼趋势，固定 ±5 MW
- 规模：3 个 fold-local fuel OLS；不训练 horizon 模型，不生成提交
- 门禁：pooled +0.50 pct，最差折/horizon ≥-0.10 pct，普通/事件安全
- 协议：`docs/56_SHORT_G1_FUEL_TREND_PROTOCOL.md`

## `20261004T160900Z_short_fuel_trend_d6f12f9ba2` — 一小时燃料趋势外推（实现失败）

- 日期：2026-10-05
- 状态：failed；未产生任何可采信指标
- 原因：控制 OOF 保留 `v16_legacy_columns` 名称，而统一门禁要求 `v16_control`
- 影响：预测已构造但在指标查找前终止；没有提交产物
- 修复：加载控制时显式规范化名称，并新增回归测试；修复后使用新 run_id

## `20261004T161012Z_short_fuel_trend_99a68f538d` — 一小时燃料趋势外推

- 日期：2026-10-05
- 角色：scientific OOF；未生成提交
- pooled -0.3171 pct；最差折 -0.4218；最差 horizon -0.4076
- 普通工况 -0.2215 pct；2 个 episode 均未改善
- 门禁：`component_gate_passed=False`
- 结论：不打包、不扫描参数，关闭 short fuel-trend 路线
- 报告：`docs/57_SHORT_G1_FUEL_TREND_REPORT.md`

## `planned_short_g1_expert_gate_v1` — 扩展时间 soft expert gate

- 日期：2026-10-05
- 角色：preregistered scientific OOF；不生成提交
- 来源：冻结 ±5 MW 动态专家 OOF 与 v16 控制
- 时间协议：wf03→wf04；wf03+wf04→wf05
- 唯一变化：20 特征 logistic 概率软门控，不调底层专家
- 门禁：pooled +0.50 pct、各折/horizon/普通不退化、AUC≥0.55、事件安全
- 协议：`docs/58_SHORT_G1_EXPERT_GATE_PROTOCOL.md`

## `20261004T161648Z_short_expert_gate_76bb9085e9` — 扩展时间 soft expert gate

- 日期：2026-10-05
- 角色：scientific OOF；未生成提交
- 顺序验证 pooled +0.6790 pct；最差 horizon +0.4806；普通 +1.3951 pct
- 分折 wf04 +2.1464、wf05 -0.2992 pct；2 个 episode 仅 1 个改善
- gate AUC：wf04 0.6284、wf05 0.3533、pooled 0.3097
- 门禁：`component_gate_passed=False`
- 结论：不打包，不反转概率/搜索阈值，关闭 expert-gate 路线
- 报告：`docs/59_SHORT_G1_EXPERT_GATE_REPORT.md`

## `planned_analog_trajectory_g1_v1` — 四个十日历史相似工况轨迹

- 日期：2026-10-05
- 角色：preregistered scientific OOF；不生成提交
- 控制：同特征、同训练 origin、同标签的 shared-horizon LightGBM
- 唯一候选：25% 历史相似 24h 轨迹 + 75% shared-tree
- 验证：4 个连续十日伪测试，共 3840 个 origin ×96 horizon
- 因果边界：每个历史邻居的完整未来 24h 必须在对应伪测试开始前结束
- 门禁：pooled +0.8 pct、最差块不退化、≥3/4 块改善、near/far 安全
- 验证状态：94 项 pytest 和 preflight 已通过，等待人工运行 CPU 实验
- 协议：`docs/61_ANALOG_TRAJECTORY_G1_PROTOCOL.md`

## `20261005T080116Z_analog_trajectory_g1_4e4f828951` — 四个十日历史相似工况轨迹

- 日期：2026-10-05
- 角色：scientific OOF；未生成提交
- pooled：25/75 融合相对 shared tree +0.057671 pct；纯 analog -0.962892 pct
- 分块：-0.466449/+0.727684/+0.010845/-0.041444 pct，仅 2/4 改善
- horizon：near/mid1/mid2/far 均小幅正向，但 far 仅 +0.034284 pct
- 邻居：平均有效邻居约 23.6，最少 12 个来源日，邻居多样性通过
- 门禁：`component_gate_passed=False`，`standalone_submission_eligible=False`
- 结论：不打包、不调邻居和融合参数，关闭起点摘要相似轨迹路线
- 报告：`docs/62_ANALOG_TRAJECTORY_G1_REPORT.md`

## `planned_sequence_gru_g1_v1` — 24h 历史窗直接多步 GRU

- 日期：2026-10-05
- 角色：preregistered scientific OOF；不生成提交
- 控制：`20261005T080116Z_analog_trajectory_g1_4e4f828951` 的四块 shared tree OOF
- 输入：96×25 因果历史序列；不含目标历史或未来过程量
- 架构：两层单向 GRU、hidden 96、100,800 参数，直接输出 96 步
- 训练：单种子、30 epochs、log-target SmoothL1、RTX 4060 CUDA
- 唯一候选：25% GRU + 75% shared tree
- 门禁：pooled +0.8 pct、最差块不退化、≥3/4 块改善、near/far 安全
- 验证状态：专项测试与 preflight 已通过，等待人工运行 GPU 实验
- 协议：`docs/63_SEQUENCE_GRU_G1_PROTOCOL.md`

## `20261005T083741Z_sequence_gru_g1_c325928f68` — 24h 历史窗直接多步 GRU

- 日期：2026-10-05
- 角色：scientific OOF；未生成提交
- 运行：4 个十日块，单种子，CUDA，49.638 秒；四折训练 loss 正常下降
- pooled：25% GRU 融合相对 shared tree +0.006482 pct；独立 GRU -1.911511 pct
- 分块：+0.440724/-1.184428/+0.712850/+0.057519 pct，3/4 改善但最差块严重退化
- horizon：near/mid1/mid2/far 为 -0.034405/-0.276029/-0.033055/+0.127236 pct
- 门禁：`component_gate_passed=False`，`standalone_submission_eligible=False`
- 验收：6 个结果工件和 4 个模型哈希全部通过，registry 唯一 completed 记录
- 结论：不打包、不做深度模型超参搜索，关闭当前 GRU 路线
- 报告：`docs/64_SEQUENCE_GRU_G1_REPORT.md`

## `planned_holder2_boundary_audit_v1` — holder_2 折内边界压力诊断

- 日期：2026-10-05
- 角色：preregistered diagnostic；零模型、零提交
- 控制：`20261005T080116Z_analog_trajectory_g1_4e4f828951` 的四块 shared tree OOF
- 假设：折内边界压力与控制远端 long-g1 残差存在稳定正向关系
- 唯一主要变化：由 5 个因果列派生的边界特征（位置/贴边/速度/平衡，固定权重）对冻结残差的关系测量
- 门禁：pooled Spearman ≥ 0.08、≥75% 块为正、pooled 高−低压组残差差 ≥ 1.0 pct、≥75% 块为正、覆盖率 ≥ 5%
- 验证状态：`104 passed`；preflight 通过（4 块、3840 起点、最少历史 5664 行）
- 协议：`docs/65_HOLDER2_BOUNDARY_AUDIT_PROTOCOL.md`

## `20261005T100107Z_holder2_boundary_a8309e91a1` — holder_2 折内边界压力诊断

- 日期：2026-10-05
- 角色：diagnostic；零模型、零提交；0.86 秒
- 控制：四开发块 shared tree OOF（哈希强校验）
- 结果：正向门禁 4/4 未过——pooled Spearman -0.1430（阈值 ≥+0.08）、正向块 0/4、
  pooled 高−低残差差 -7.451 pct（阈值 ≥+1.0）、正向差值块 0/4；覆盖率通过
- 事后观察：反向关联 4/4 块一致（near ≈0、far 一致为负；高压组真实 far/near 比 -4.0%
  对低压组 +6.0%），远端特异、超前于真实负荷回落
- 结论：正向路线关闭；反向假设不得在开发块上验证，转入新鲜块确认协议
- 报告：`docs/66_HOLDER2_BOUNDARY_AUDIT_REPORT.md`

## `planned_holder2_inverse_confirm_v1` — 反向边界信号新鲜块确认

- 日期：2026-10-05
- 角色：preregistered diagnostic；重建 4 个控制树、零修正模型、零提交
- 新鲜块：6/11-20、6/21-30、7/18-27、8/21-30（≥30 天折内历史，与全部开发块零重叠）
- 控制：按 analog 冻结配方逐项重建 shared-horizon tree（同 21 特征、同超参、同库起点规则）
- 反向门禁：pooled Spearman ≤ -0.08、高−低差 ≤ -1.0 pct、各 ≥75% 块同向；
  机制条件（压力 vs 真实远-近变化 ≤ -0.05、≥75% 块为负）；覆盖率 ≥5%
- 失败决策：holder_2 路线（含反向）永久关闭；通过决策：仅允许预登记固定幅度高压组远端修正
- 验证状态：`113 passed`；等待人工运行
- 协议：`docs/67_HOLDER2_INVERSE_CONFIRM_PROTOCOL.md`

## `20261005T102336Z_holder2_inverse_confirm_befe6e5687` — 反向边界信号新鲜块确认

- 日期：2026-10-05
- 角色：preregistered diagnostic；重建 4 个控制树、零修正模型、零提交
- 运行状态：4 树训练与 6 工件全部完成、哈希复核一致，但登记簿写入因 runner 记账缺陷
  （`_validate` 遗漏 `label_manifest` 键）标记 failed；缺陷已修复，干净复跑待执行，
  deterministic 下数字应逐位复现
- 控制：新鲜块上按 analog 冻结配方重建 shared tree，MAPE 0.100-0.124
- 结果：反向门禁判负——pooled 残差 Spearman -0.0138（阈值 ≤-0.08）、负向块 2/4、
  pooled 高−低差 -0.422 pct（阈值 ≤-1.0）
- 机制条件大幅通过：压力 vs 真实远−近变化 pooled -0.441、4/4 块为负
- 结论：残差链接不成立（控制误差方向随工况翻转）；holder_2 路线（含反向）永久关闭；
  机制信号确认为全项目复现性最强的目标领先信号，仅可作为未来模型重设计的依据
- 报告：`docs/68_HOLDER2_INVERSE_CONFIRM_REPORT.md`

## `20261005T111953Z_holder2_inverse_confirm_d28fee07b0` — 反向边界信号新鲜块确认（正式）

- 日期：2026-10-05
- 角色：preregistered diagnostic；重建 4 个控制树（27.1 秒）、零修正模型、零提交
- 控制：新鲜块上按 analog 冻结配方重建 shared tree，MAPE 0.100-0.124
- 结果：反向门禁判负——pooled 残差 Spearman -0.0138（阈值 ≤-0.08）、负向块 2/4、
  pooled 高−低差 -0.422 pct（阈值 ≤-1.0）；机制条件通过（pooled -0.441、4/4 块为负）
- 结论：残差链接不成立；holder_2 路线（含反向）永久关闭；与首次 failed run 逐位一致
- 报告：`docs/68_HOLDER2_INVERSE_CONFIRM_REPORT.md`

## `20261005T103758Z_october_shift_0b06b4230c` — 十月分布偏移定位审计

- 日期：2026-10-05
- 角色：diagnostic；零模型、零提交；0.5 秒，26 列原始过程变量
- 结果：十月二号气柜中位数比九月高 0.80 个九月 IQR（88,897 → 145,267，接近训练上界区）；
  三号高炉 6.96% 的十月取值低于全训练期最小值（硬外推）；一号气柜 13.5% 落在训练软支撑外；
  五号高炉 +0.54 IQR、发电耗气 +0.26 IQR 高于九月
- 结论：十月是持续高气柜制度，机制前提在测试期持续激活；BF3 硬外推是九月前训练模型的
  共同缺陷；已形成给学弟的情报简报
- 参考：`docs/69_OCTOBER_REGIME_BRIEFING.md`

### `<run_id>` — `<实验名称>`

- 日期：
- 角色：diagnostic / scientific / competition / final
- 控制组：
- 假设：
- 唯一主要变化：
- 协议版本：
- split 指纹：
- 运行命令：
- 结果摘要：
- 分目标、分短长结果：
- 稳定性与风险：
- 结论：采用 / 拒绝 / 继续诊断
- 下一步：
