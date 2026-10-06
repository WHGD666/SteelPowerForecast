# 决策日志

本日志只追加。每项重要协议变化必须说明日期、决定、依据、影响范围和后续动作。

## D-001：建立复赛 v3 隔离工作区

- 日期：2026-09-29
- 状态：已采纳
- 决定：在同一个 SteelPowerForecast 大仓库的 `shiyan/fusai/` 下建立复赛 v3 工作区。
- 原因：保留初赛与旧复赛历史，同时避免错误标签、旧配置和旧输出污染新实验。
- 影响：后续复赛新代码、配置、处理数据、实验输出和提交登记均进入本目录。

## D-002：旧版只作为 legacy 证据

- 日期：2026-09-29
- 状态：已采纳
- 决定：学弟旧代码不在原地继续修改，新版不得直接依赖旧固定输出目录。
- 原因：旧版同时包含有效思想、错误口径和复现缺口。
- 影响：需要复用的组件必须经过审查和测试后迁移。

## D-003：重建协议而不是延续 v2

- 日期：2026-09-29
- 状态：已采纳
- 决定：新版协议命名为 `round2_v3`，不静默修改旧 v2 配置。
- 原因：目标区间、字段状态、提交合同和 holdout 状态均已发生实质变化。
- 影响：v2 和 v3 本地指标不可直接声称同口径提升。

## D-004：15 分钟目标采用区间均值

- 日期：2026-09-29
- 状态：已采纳并通过自动测试
- 决定：t+15 使用 `[t,t+14]`，t+30 使用 `[t+15,t+29]`，依次顺延。
- 依据：官方答疑对 10:00 起点示例的确认。
- 影响：旧点值标签和 shift(-step) 区间标签均需废止。

## D-005：Sep 28–29 不再作为 sealed holdout

- 日期：2026-09-29
- 状态：已采纳
- 决定：该时段仅作为普通开发验证证据。
- 原因：已被用于多次缩放、平滑、递归和路径设计选择。
- 影响：当前不存在真正未消费的内部封存集。

## D-006：短期与长期独立优化

- 日期：2026-09-29
- 状态：已采纳
- 决定：允许并优先评估不同短期、长期模型，不再默认短文件只是长模型前 8 步。
- 依据：官方答疑明确允许两个不同模型。
- 影响：实验、指标和提交均需分别登记短期与长期来源 run。

## D-007：平台提交采用单变量设计

- 日期：2026-09-29
- 状态：已采纳
- 决定：固定一个文件或目标，只替换一个主要组成部分。
- 原因：每日五次机会应产生可归因证据，避免混合改动导致无法判断。
- 影响：submission manifest 必须声明控制组和唯一主要变化。

## D-008：冻结 v3 walk-forward 切分

- 日期：2026-09-29
- 状态：已采纳并生成逐起点清单
- 决定：使用 6 个 expanding-window 折；前 5 折用于模型选择，Sep 28–29 仅作为 `consumed_development_only`。
- 原因：需要覆盖常规、三号高炉新投运、晚九月和一号气柜新投运工况，同时明确隔离已被旧实验消费的日期。
- 影响：v3 baseline 及后续受控实验必须使用同一份 `validation_origins_v3.csv`；修改切分必须升版本。

## D-009：冻结本地指标而不冒充官方总分

- 日期：2026-09-29
- 状态：已采纳并通过人工样例测试
- 决定：本地以有效单元格 pooled MAPE 和 `1-MAPE` 为主；零真值排除并计数，同时报告 MAE、RMSE、缺失计数和分组指标。
- 原因：官方隐藏映射未公开，不能从本地准确率武断换算平台总分。
- 影响：所有 run 必须保存短/长、分目标、分 horizon、分折结果。

## D-010：v3 baseline 不做默认填补和异常替换

- 日期：2026-09-29
- 状态：已采纳
- 决定：因果 base 只补齐时间轴并生成缺失/投运标记；原始协变量缺失值和数值原样保留，不后向填补、不插值、不默认前向填补、不删除异常行。
- 原因：LightGBM 可原生处理缺失；在协议基线中先保留证据，避免未来信息和未经验证的清洗收益。异常阈值若参与模型，必须在每个训练折内部拟合。
- 影响：任何有界前向填补、异常置空或 winsorize 都必须作为独立实验，并重新生成版本化产物与 manifest。

## D-011：第一版 origin 特征不使用目标历史

- 日期：2026-09-29
- 状态：已采纳
- 决定：v1 只使用起点及以前的官方过程协变量、观测聚合、缺失/投运状态、lag、rolling 和起点日历；不加入 `generator_1`、`generator_all` 历史值。
- 原因：正式测试目标列全空。先建立不依赖目标观测的可部署控制组，避免本地验证因真实目标 lag 获得无法在测试期复现的虚高收益。
- 影响：目标预测反馈只能在后续独立实验中加入，而且必须从测试期首个起点开始完整滚动回放。

## D-012：首个 v3 baseline 采用弱基线加共享 horizon LightGBM

- 日期：2026-09-29
- 状态：配置及 preflight 已通过，待正式运行
- 决定：同一套冻结 OOF 上同时评估 target-time 日历中位数弱基线与按短/长、按目标独立训练的共享 horizon LightGBM；长期训练起点按 60 分钟抽样，验证仍覆盖所有 15 分钟起点。
- 原因：先证明非线性过程特征相对可解释弱基线的增益，同时控制 24 小时长表展开后的内存和运行时。
- 影响：首次正式运行固定为 5 个 `model_selection` 折、20 个 LightGBM；Sep 28–29 消费开发折不参与模型选择指标，且本次不生成提交文件。

## D-013：首个 baseline 启动失败保留为失败证据

- 日期：2026-09-29
- 状态：已修复并增加回归测试
- 失败 run：`20260929T124109Z_r2v3_baseline_b19a28a9ec`
- 原因：运行器写特征名清单时漏导入 `TARGET_CALENDAR_COLUMNS`，在模型训练前触发 `NameError`。
- 影响：未训练模型、未产生 OOF 指标；失败目录和 registry 行保留，不删除、不冒充实验结果。
- 修复：显式导入冻结常量，并新增运行器常量合同测试。

## D-014：拒绝层次目标分解 v1

- 日期：2026-09-29
- 状态：已拒绝
- run：`20260929T131221Z_r2v3_hierarchy_af50589323`
- 证据：短期总体由 88.33% 降至 87.92%，长期由 84.53% 降至 83.52%；short/long g1 均退化。
- 决定：不提交层次模型，不与控制组做无证据融合。

## D-015：建立 baseline v1 平台校准提交

- 日期：2026-09-29
- 状态：提交构建器已通过单元测试，待全量生成
- 决定：使用首个可复现 baseline v1 全量重训，生成一次平台外部校准提交。
- 原因：十月国庆工况和隐藏评分映射无法仅由本地 OOF确认；平台结果可用于判断方向，但不得作为测试标签反推器。
- 边界修复：测试 lag/rolling 必须把 9 月训练历史接到 10 月测试期，不能在 10 月 1 日重新冷启动。

## D-016：拒绝 baseline v1 作为高分候选

- 日期：2026-09-29
- 状态：已拒绝并冻结为外部校准锚点
- submission：`20260929T212923_calibration_baseline_v1_91cd219d33`
- 官方结果：总分 48.9241；平台未返回短期、长期或分目标明细
- 对照：低于 legacy v9 的 52.4085，共 3.4844 分
- 决定：不再提交同族 baseline 微调；保留正确标签、信息边界、验证与复现资产，转入历史机制复现和单变量改进。
- 原因：本地门槛本已未通过，晚九月和投运工况退化明显；提交格式检查均通过，当前应优先解决跨工况泛化而不是重复打包。
- 影响：后续平台提交必须通过近期工况阻断条件，且尽量只替换短文件或长文件中的一个。

## D-017：平台总分不得用于无分项归因

- 日期：2026-09-29
- 状态：已采纳
- 决定：48.9241 只作为整体候选证据，不据此判断短期/长期、`generator_1`/`generator_all` 谁导致下降。
- 原因：平台未提供组成明细，同次提交同时包含两个周期和两个目标。
- 影响：下一轮平台实验必须设计成即使只返回总分也能支持或拒绝单一假设。

## D-018：冻结外部约 61 分预测产物

- 日期：2026-10-01
- 状态：已采纳
- artifact：`20261001_external_best_61_pending_exact`
- 决定：按原字节冻结收到的 ZIP，并登记 ZIP 与两个 CSV 的 SHA-256、结构和物理检查结果。
- 成绩口径：仅记录“用户报告约 61 分”；精确官方成绩和分项保持空值，待证据补齐。
- 原因：高分产物必须先建立不可混淆的身份，避免后续版本覆盖或把口头近似值写成精确事实。

## D-019：将约 61 分版本作为外部强控制，而非已复现 run

- 日期：2026-10-01
- 状态：已采纳
- 决定：后续候选以该预测输出为性能控制；在源码、配置、模型和 ZIP 完成字节级闭环前，不把它登记成 round2_v3 可复现模型 run。
- 机制口径：近期窗口树模型 + 九月燃料后处理仅为高可信推断，必须保留 inference 标签。
- 原因：ZIP 只有两个预测 CSV，没有 provenance manifest；结构正确和平台高分不能替代工程复现证据。

## D-020：下一轮只改动态燃料 `generator_1`

- 日期：2026-10-01
- 状态：诊断通过，待正式实现
- 决定：冻结约 61 分版本的 `generator_all`；下一正式实验仅替换 `generator_1`，采用 21 天近期窗口、horizon-specific 动态燃料映射，并以 70% 动态模型 + 30% v3 树作为首个冻结设置。
- 诊断证据：short g1 OOF 准确率提高 4.0936 pct，long g1 提高 4.0650 pct；这些是开发诊断，不是官方增分。
- 风险：八月和 BF3 投运折存在轻微退化，正式实验必须保存逐折指标并执行近期工况阻断。
- 原因：这是目前最接近“只改变一个评分组件”的高价值假设，平台只返回总分时仍可形成方向性归因。

## D-021：先做来源鉴定，再修正外部候选的协议缺陷

- 日期：2026-10-01
- 状态：已采纳；v15/v16 来源假设已完成最小检验
- 决定：第一轮保持旧逻辑不变，用少量代表 horizon 鉴定 61 分包来源；只有来源命中后才运行全量 192 模型。
- 原因：若同时修正标签或后处理，就无法判断当前约 61 分提交包的真实来源。
- 验收：代表 horizon 首先应逐值命中；全量阶段两个 CSV 必须与冻结参考成员逐字节一致；ZIP 容器因时间戳差异只作辅助证据。
- 隔离：新交接材料保存在 `legacy/junior_handover_20261001_v16_candidate/`，训练在忽略的 `outputs/reproduction_workspaces/` 中进行，不修改 legacy 快照。
- 后续：只有复现通过后，才把动态燃料、目标时刻气候项、区间均值标签或独立短模型作为新的单变量实验。

## D-022：拒绝把当前约 61 分包归因为交接源码原样 v15/v16

- 日期：2026-10-01
- 状态：已拒绝来源假设
- 证据：v15 全期树、v16 九月树在 g1 step 1/96 与 gall step 1/33/96 上均为 960 行全部不一致；多个训练起点和全期/九月树融合均未命中。
- 已确认组件：九月 g1/gall 燃料回归系数与参考预测结构一致。
- 决定：停止全量 v15/v16 重训，避免消耗 CPU 得到已知不匹配的产物。
- 阻塞信息：61 分版实际基础预测脚本、参数、base-dir 或额外后处理未随 ZIP 提供。
- 下一步：取得更新源码或生成记录后恢复字节级复现；在此之前只做不会冒充来源复现的独立研究。

## D-023：将外部高分包来源更正为 v28 / 61.5824

- 日期：2026-10-01
- 状态：已采纳
- 新证据：`给学长_最优版整合(1).zip` 的 v28 两个 CSV 与冻结高分包成员 SHA-256 完全一致；交接 README 记录官方总分 61.5824。
- 决定：把控制组 ID 更新为 `external_v28_61_5824`；此前“约 61、来源未知”与 v15/v16 排查保留为历史调查，不再代表当前结论。
- 复现口径：来源已解析不等于本项目已复现。状态为 `source_resolved_independent_rebuild_pending`，待全量运行后以两个 CSV 哈希验收。

## D-024：v29b 的 63.45 只登记为 projected

- 日期：2026-10-01
- 状态：已采纳
- 决定：v29b 不登记 official score，不称为 63.45 分版本。
- 原因：63.45 是由四个历史分项拼装推算，整包尚未提交；脚本依赖 v20、v16_final 和 v29 中间输出，最新版交接包未提供可独立生成 v20 的闭环。
- 门禁：提交前必须完成组成列来源核对、依赖重建、文件合同校验和候选哈希冻结。

## D-025：v34 作为单组件开发候选，不直接外推平台增分

- 日期：2026-10-01
- 状态：已采纳
- 决定：保留 v34 的“冻结 v28 `generator_all`、只修改 `generator_1`”实验结构，但三折结果只能作为开发证据。
- 证据：v34 short/long gall 与 v28 逐值相同，g1 全部发生变化；源码实际加入 g1 异常样本权重，并把燃料权重改为 15/30 分钟 0.6、其余 0.1。
- 风险：源码 docstring/manifest 陈旧，且 `h_step <= 32` 实际只覆盖 15/30 分钟；验证协议需独立审计后才能决定是否提交。

## D-026：采用 v29b × v34 自适应平台阶梯冲击 64–65

- 日期：2026-10-02
- 状态：已采纳并完成前两个候选构建
- 决定：不把独立复现 v28 设为冲分前置条件；使用已经冻结的 v29b 成品作为锚点，永久冻结 short/long gall，只在 g1 上与 v34 做权重 0/0.25/0.5/1.0 的凸组合。
- 第一探针：先提交 v29b 原样锚点，再只把 long g1 替换为 50% v29b + 50% v34。
- 自适应规则：long 0.5 改善才测试 1.0；退化则测试 0.25。选定最佳 long 后，才用相同逻辑测试 short g1。
- 原因：平台只返回总分，单区块变化仍能给出方向性因果证据；一次改四个区块无法归因。
- 停止条件：v29b 锚点明显低于预估、输出合同失败，或 0.5 已退化却试图盲目放大权重。
- 配置：`configs/round2_v3/experiments/component_ladder_v1.yaml`
- 方案：`docs/16_64_65_PLATFORM_LADDER.md`

## D-027：取消 v29b × v34 权重阶梯，冻结 v29b 63.4621 为新控制组

- 日期：2026-10-04
- 状态：已采纳；覆盖 D-026 的待提交建议
- 新证据：`分最高三包_交底文档.zip` 明确给出 v29b 官方 63.4621；其 ZIP/CSV 哈希与此前预估包完全一致。
- 否决证据：v34b 已在官方平台完成端点实验，保持 v29b short/gall、只使用 v34 long g1，得 62.9325；long g1 82.28% 低于 v29b 的 82.96%。
- 决定：不提交已经生成的 long-g1 50% 探针，不继续测试 0.25/1.0；本地候选保留并标记 `rejected_before_submission`。
- 新控制组：`external_v29b_63_4621`。
- 新方向：停止既有权重、门控、窗口和多种子微调；优先审计台阶日前兆、电价目标时刻特征、气柜边界状态，以及隔离的 g1 异常事件样本权重。
- 详细报告：`docs/17_TOP3_AND_50_TRIALS_REVIEW.md`。

## D-028：将 long generator_1 台阶事件预警设为 65 分主攻方向

- 日期：2026-10-04
- 状态：已采纳；当前只完成研究设计，尚未训练或提交
- 控制组：`external_v29b_63_4621`，距离 65 分 1.5379 分。
- 决定：永久冻结首轮实验的 short 和全部 `generator_all`；先建立未来台阶事件的专用验证，只有发现起点前稳定因果前兆后，才训练 long `generator_1` 的事件概率/残差修正模型。
- 原因：long g1 官方准确率 82.96%，是最低且可独立替换的区块；近 50 版已基本否定权重、窗口、门控、种子与旧预测融合的小幅微调。
- 平台门禁：离线 long-g1 至少提升 0.8–1.0 pct、普通日退化不超过 0.2 pct、近端无明显退化，并通过跨事件稳定性检查。
- 次级假设：保持 v29b/v28 燃料配方不变时，隔离测试异常事件样本权重；holder_2 边界只作为事件特征；电价只做低优先级受控消融。
- 风险边界：v35、v45 的 `cheat_detected` 仅记录为平台事实；其触发机制仍是作者推断，不能冒充官方规则。
- 方案：`docs/18_BREAK_65_RESEARCH_PLAN.md`。

## D-029：不直接训练事件门控器，先做 process-only long-g1 特征组消融

- 日期：2026-10-04
- 状态：Phase A 已完成；Phase B 已预登记、尚未训练
- diagnostic run：`20261004T052259Z_step_event_audit_0057a685f6`
- 复现结果：学弟口径的 24 个低燃料事件日精确复现；连续日期合并后只有 13 个独立 episode；测试同口径只命中 10 月 1 日和 10 月 10 日。
- 目标核验：事件后 2h g1 中位下降 4.542 MW，但 24h 中位恢复 4.530 MW；仅 6/13 同时满足 2h 与 6h 下降至少 5 MW。
- 决定：拒绝把 `ratio<0.8` 直接作为 long-g1 下台阶标签；拒绝基于 13 个 episode 训练高自由度事件分类器。
- 允许方向：只把提前 2–6h 显示稳定性的高炉生产、发电耗气、holder_2、煤气平衡和用户变化构造成少量过程特征，进行同切分 feature-off/feature-on 消融。
- 信息边界：真实 g1/gall 历史在正式测试不可得，相关诊断特征不得进入候选；所有统计必须只使用起点及以前数据，并在训练折内拟合。
- 门禁：long-g1 至少提高 0.8–1.0 pct，普通日起点退化不超过 0.2 pct，同时通过近期折、事件折和 near/far 检查。
- 证据：`docs/19_STEP_EVENT_AUDIT_REPORT.md`。
- 配置：`configs/round2_v3/experiments/event_signal_ablation_v1.yaml`。

## D-030：关闭直接事件特征扩充路线，转入动态燃料正式验证

- 日期：2026-10-04
- 状态：已采纳；Phase B 已完成，无候选晋级
- run：`20261004T062435Z_event_signal_ablation_3cc01b0f7b`
- 对照有效性：control 与 baseline v1 long-g1 OOF 的 129,024 条真值和预测逐值一致。
- 最佳结果：`control_plus_bf_production` 提升 0.3795 pct；煤气平衡 +0.3647 pct；用户/混合煤气 +0.3275 pct。
- 失败证据：所有变体低于 0.8–1.0 pct 总体门槛；最佳组也仅改善 2/4 个 episode；收益明显依赖 BF3-active 折。
- 决定：不把任何 Phase B 变体接入 v29b，不生成提交包；`bf_production` 只保留为未来可复验的弱辅助信号，`compact_union` 和 holder_2 主信号路线拒绝。
- 协议修正：后续门禁除近期折均值外，增加近期单折退化下限，防止均值掩盖单个十月相似工况失败。
- 下一方向：把 `diag_20261001_dynamic_fuel_g1_v0` 工程化成冻结的同五折 long-g1 科学实验；首轮不得同时加入 BF production 或其他新信号。
- 证据：`docs/20_EVENT_SIGNAL_ABLATION_REPORT.md`。

## D-031：冻结动态燃料 v1 协议，禁止首轮权重搜索

- 日期：2026-10-04
- 状态：已预登记并通过轻量预检；尚未正式训练
- 控制组：`20260929T124936Z_r2v3_baseline_46e02afe85` 的 long-g1 OOF，运行时必须按键和真值逐值核对。
- 唯一主要变化：三路发电耗气的当前、15/60/120/240/1440 分钟 lag、1/4/24 小时均值和目标时刻日历项；每个 horizon 独立 OLS。
- 窗口：每折每 horizon 只取标签已成熟的最近 21 天；imputer、scaler 和 OLS 全部折内拟合。
- 唯一晋级候选：70% dynamic OLS + 30% control；standalone OLS 只作机制诊断，不允许在 OOF 后挑选融合权重。
- 规模：5 折 × 96 horizon = 480 个轻量模型；27 个燃料特征、36 个总模型特征；最大来源偏移 0。
- 新门禁：最近三折除平均非退化外，最差单折不得低于 -0.2 pct；总体仍须至少 +0.8 pct。
- 隔离：首轮不加入 BF production 或其他 Phase B 信号，不生成提交包。
- 配置：`configs/round2_v3/experiments/dynamic_fuel_g1_v1.yaml`。
- 方案：`docs/21_DYNAMIC_FUEL_G1_PROTOCOL.md`。

## D-032：动态燃料 v1 保留为强机制证据，不直接提交

- 日期：2026-10-04
- 状态：已完成；未通过晋级门禁
- run：`20261004T070858Z_dynamic_fuel_g1_31e2a08193`
- 总体：70% dynamic + 30% control 的 long-g1 提升 3.3675 pct；四个 horizon bucket 全部改善。
- 稳定性失败：August -1.0513 pct、BF3-active -1.4043 pct；普通起点 -1.6419 pct；近期最差折低于 -0.2 pct。
- 事件证据：事件目标 +7.3886 pct，3/4 episode 改善；但事后 episode 标签不能作为线上开关。
- 外推风险：控制是 baseline OOF，不是 v29b OOF；十月动态特征中心最接近发生退化的 BF3-active 折。
- 合规风险：21 天全量窗口将集中于 9 月 10–30 日，与历史 `cheat_detected` 的窄九月燃料窗风险相似；官方机制未知，不宣称必然触发。
- 决定：不生成本版本提交包。保持模型、特征和 70/30 权重不变，只做 56 天窗口稳定性复验。
- 停止条件：56 天版仍未通过时，关闭全时段替换；不得在同一 OOF 上继续搜索窗口或权重。
- 报告：`docs/22_DYNAMIC_FUEL_G1_V1_REPORT.md`。

## D-033：按用户明确授权生成动态燃料 v1 高风险平台探针

- 日期：2026-10-04
- 状态：已上传并返回官方 62.8070；判负封存
- submission：`20261004T153207_dynamic_fuel_probe_v1_98feef8d8a`
- 覆盖说明：本决定是用户在获知失败门禁后对 D-032“不提交”的明确覆盖，仅授权生成和试传这一包，不改变其高风险性质。
- 精确公式：long g1 = 70% full-refit dynamic-fuel OLS + 30% 冻结 v3 baseline long-g1。
- 冻结区块：v29b short 字节级冻结；v29b long gall 逐值冻结。
- 未采用的不同公式：没有把 dynamic OLS 与 v29b long-g1 再做 70/30，因为该组合没有对应的本地 OOF证据。
- 验收：短表 960×17、长表 960×193；96 个模型哈希、CSV、ZIP 成员、时间网格、有限值和物理边界全部通过。
- ZIP SHA-256：`9c5e7fb36cac5f53db2053198f5180f926ccdbed7745d507979e887f34a5f1de`。
- 平台结果：62.8070，相对 v29b 63.4621 下降 0.6551；平台未返回分项，不反推 long-g1 准确率。
- 裁决：本地失败门禁得到平台验证；该包不升级为正式最优。
- 后续：停止该模型的窗口和融合权重搜索，恢复 v29b 为唯一官方控制组。

## D-034：关闭全时段 dynamic-fuel 替换，转向未被混杂验证的独立改进

- 日期：2026-10-04
- 状态：生效
- 21 天证据：本地 pooled long-g1 +3.3675 pct，但普通工况 -1.6419、August -1.0513、BF3-active -1.4043；平台总分 -0.6551。
- 56 天证据：blend 仅 +0.3083 pct；普通工况 -5.7747、最近三折均值 -1.5309、最近最差折 -6.9628，`promotion_passed=[]`。
- 结论：动态燃料可以解释部分事件，但不能稳定替换全时段 long-g1；不得继续在同一 OOF 上搜索窗口、权重或仅靠减小融合比例试探平台。
- 下一优先级：严格保留 v29b 配方，仅隔离验证 g1 异常日样本加权；该因素此前在本地三折均为正，但官方 v34b 同时改变了燃料权重，尚不能单独归因。
- 高上限研究线：预先预测“未来 horizon 是否进入低燃料/台阶事件”的因果 hazard，再只对高置信事件起点做小幅校正；不得使用事件事后标签或测试未来过程量。

## D-035：预登记 v28 g1 异常样本加权的纯净隔离实验

- 日期：2026-10-04
- 状态：代码、65 项测试与 preflight 已通过；待人工运行
- 控制：折内重建 v28 long-g1；不宣称等于不可得的 v29b 历史 OOF。
- 唯一主要变化：燃料隐含水平相对 7 日基线小于 0.8 时，g1 树训练样本权重从 1 改为 3。
- 冻结：v28 恢复列、九月树、模型参数、燃料回归和源码实际 0.8/0.6 近远融合。
- 灵敏度：×2/×5 只验证方向稳定性，禁止 OOF 后从中另选赢家。
- 标签：官方区间均值为主，legacy 瞬时点为辅助。
- 范围：最近三个时序折、96 个 horizon、4 个权重，共 1,152 个 CPU 模型；不生成提交包。
- 晋级：×3 必须跨折、普通工况、episode、near/far 和两种标签同时过门禁；单独提交还要求 long-g1 至少 +0.8 pct。
- 协议：`docs/24_ANOMALY_WEIGHT_G1_PROTOCOL.md`。

## D-036：异常样本加权判负，禁止打包与继续扫权重

- 日期：2026-10-04
- 状态：已完成；门禁失败
- run：`20261004T092221Z_anomaly_weight_g1_e6563a0b55`
- ×3：official interval long-g1 +0.0176 pct，short-g1 -0.0523 pct，legacy point long +0.0188 pct。
- 分折：BF3-active 0.0000、late September +0.0950、holder1-active -0.0163 pct。
- 事件：仅 1/2 episode 改善；near -0.0523、far +0.0360 pct。
- 灵敏度：×5 pooled long 略高至 +0.0275 pct，但 short/near 降至 -0.1019 pct，且不是预登记候选。
- 门禁：`component_gate_passed=False`，`standalone_submission_eligible=False`。
- 决定：不生成提交包；停止阈值和权重搜索；转入事件 onset hazard 因果可预测性诊断。
- 报告：`docs/25_ANOMALY_WEIGHT_G1_REPORT.md`。

## D-037：24h onset hazard 判负，2h 信号仅保留为 short-only 新假设

- 日期：2026-10-04
- 状态：诊断完成；24h 门禁失败
- run：`20261004T093404Z_event_onset_hazard_ce66daea3f`
- 24h process：AP 0.1341、AP lift 0.87、precision lift 0.94、recall 29.43%、alert 31.46%、仅赢 2/4 折。
- 24h 检测 7/8 episode 由过量告警换得，不能进入 residual correction。
- 6h process：AP 0.2761，低于 calendar 0.3090；不构成独立优势。
- 2h process：AP 0.3474、AP lift 27.06、ROC-AUC 0.9746，但 alert 漂至 19.22%。
- 决定：按预登记停止 24h long hazard；2h 只作为 future short-only 假设，不直接改提交。
- 下一主线：隔离验证 v28/v29b 训练目标从瞬时点切换为官方 15 分钟区间均值的标签对齐。
- 报告：`docs/27_EVENT_ONSET_HAZARD_REPORT.md`。

## D-038：预登记 v28/v29b g1 官方区间均值标签对齐实验

- 日期：2026-10-04
- 状态：代码、测试与 preflight 已通过；待人工运行
- 控制：冻结 anomaly-weight run 的 v28 ×1 point-tree + point-fuel OOF。
- 唯一晋级候选：interval-mean tree + interval-mean fuel；保持特征、模型参数和 0.8/0.6 燃料权重不变。
- 消融：只改 tree、只改 fuel 均仅解释收益来源，不允许事后替代 primary。
- 公平性：interval tree 使用与 legacy control 相同训练 origin；所有区间标签在 fold train_end 前成熟。
- 规模：3 折 ×96 horizon =288 个 CPU LightGBM；不生成提交包。
- 门禁：组件至少 +0.5 pct，单独候选至少 +0.8 pct，且近期折、普通工况、episode、near/far 全部安全。
- 协议：`docs/28_V28_LABEL_ALIGNMENT_PROTOCOL.md`。

## D-039：官方区间均值训练标签判负，冻结旧点标签

- 日期：2026-10-04
- 状态：已完成；门禁失败
- run：`20261004T095143Z_v28_label_alignment_5b8c1cdab3`
- primary：interval tree + interval fuel 相对 point control 退化 0.0330 pct。
- 消融：只改 tree 退化 0.0204 pct；只改 fuel 退化 0.0123 pct，两项没有互补收益。
- 分折：BF3-active -0.1066、late September -0.0247、holder1-active +0.0351 pct。
- 分段：near/mid1/mid2/far 分别 -0.0016/-0.0421/-0.0425/-0.0304 pct；普通工况损失 0.0746 pct。
- 门禁：`component_gate_passed=False`，`standalone_submission_eligible=False`。
- 决定：不打包；停止标签均值窗口、时间偏移和融合权重搜索；v29b long-g1 继续冻结。
- 弱提示：interval tree 在 short bucket 仅 +0.0050 pct，属于噪声量级，只用于限定下一步研究范围。
- 下一步：隔离验证 v29b/v16 short-g1 恢复字段，不混入标签对齐或事件修正。
- 报告：`docs/29_V28_LABEL_ALIGNMENT_REPORT.md`。

## D-040：预登记 v16/v29b short-g1 恢复字段隔离实验

- 日期：2026-10-04
- 状态：代码与协议已准备；待人工运行
- 控制：精确复建 v16 的旧字段、九月点标签树、九月 fuel OLS 和全 short 0.8 fuel 权重。
- 唯一变化：tree feature engine 切换到 v28 恢复字段与完整高炉平衡。
- 冻结：short-gall、全部 long、标签口径、模型参数和燃料公式。
- 范围：最近三个折、8 个 short horizon、2 个变体，共 48 个 CPU LightGBM；不生成 ZIP。
- 门禁：official short-g1 +0.15 pct、最差折 ≥-0.05 pct、legacy point 不退化、最差 horizon ≥-0.10 pct；+0.50 pct 才可单区块提交。
- 协议：`docs/30_SHORT_G1_RESTORED_COLUMNS_PROTOCOL.md`。

## D-041：short-g1 恢复字段总体微升但跨折失稳，判负封存

- 日期：2026-10-04
- 状态：已完成；门禁失败
- run：`20261004T102800Z_short_g1_restored_1fb7013867`
- pooled：official short-g1 +0.1024 pct；legacy point +0.0941 pct。
- 分折：BF3-active +0.7920、late September -1.0467、holder1-active +0.1789 pct。
- 分 horizon：15/30 分钟略负，45–120 分钟为正；但 late-September 的 8 个 horizon 全负。
- 因果门控诊断：BF3/AH3 水平与逐起点收益没有稳定关系；仅 8 个验证日不足以安全搜索其他阈值。
- 门禁：`component_gate_passed=False`，`standalone_submission_eligible=False`。
- 决定：不打包，不扫 restored/control 权重，不按日期或事后误差门控；v29b short-g1 保持冻结。
- 下一步：单独研究既有 2h onset process 信号如何以 fold-local 因果特征进入 short-g1。
- 报告：`docs/31_SHORT_G1_RESTORED_COLUMNS_REPORT.md`。

## D-042：暂停 short 微增益，预登记 v28 long-g1 共享 horizon 树

- 日期：2026-10-04
- 状态：根据用户建议转向 long；代码、66 项测试与 preflight 已通过，待人工运行
- 原因：short-g1 恢复字段仅 +0.1024 pct 且跨折失稳，无法覆盖 65 分缺口；long-g1 82.96% 仍是主要瓶颈。
- 控制：折内 v28 独立 horizon 树 OOF。
- 唯一变化：96 棵树替换为一个包含 horizon/目标时刻日历编码的共享树。
- 冻结：九月窗、点标签、恢复字段、模型参数、0.8/0.6 fuel 公式和其余三个提交区块。
- 范围：三个近期折，仅 3 个 CPU LightGBM；不生成 ZIP。
- 门禁：组件 +0.5 pct、单独候选 +0.8 pct，并要求跨折、普通工况、episode、near/far 安全。
- 协议：`docs/32_LONG_G1_SHARED_HORIZON_PROTOCOL.md`。

## D-043：共享 horizon v1 广泛正向但早期折失败，保留一次扩窗复验

- 日期：2026-10-04
- 状态：v1 已完成；未过门禁、不打包
- run：`20261004T105317Z_long_g1_shared_aa987f06b5`
- pooled：official long-g1 +0.3597 pct；legacy point +0.3617 pct。
- 分折：BF3-active -0.3840、late September +1.0943、holder1-active +0.6138 pct。
- 分段：near/mid1/mid2/far 全部提升 +0.3258/+0.5866/+0.2237/+0.3578 pct。
- 事件：2/2 episode 改善；普通工况损失 0.1541 pct，在 0.2 安全线内。
- 机制：失败折只有 13,776 个成熟训练对，后两折为 161,232/216,528；共享结构存在早期样本饥饿。
- 决定：v1 不提交；只允许一次 8–9 月树窗稳定性复验，fuel 仍冻结九月。
- 报告：`docs/33_LONG_G1_SHARED_HORIZON_V1_REPORT.md`。

## D-044：预登记共享 horizon v2 的 8–9 月树窗复验

- 日期：2026-10-04
- 状态：配置、66 项测试与 preflight 已通过；待人工运行
- 唯一变化：tree start 由 2025-09-01 改为 2025-08-01。
- 冻结：fuel fit start 仍为 2025-09-01；其余配方、切分、标签和门禁与 v1 完全相同。
- 停止条件：仍有负折或 pooled <+0.5 pct 即关闭路线；不继续扫窗口或融合 v1/v2。
- 协议：`docs/34_LONG_G1_SHARED_HORIZON_V2_PROTOCOL.md`。

## D-045：共享 horizon g1 v2 冻结为强组合积木，不做事后事件门控

- 日期：2026-10-04
- 状态：已完成；组件强正向但未获单独提交资格
- run：`20261004T122856Z_long_g1_shared_85d58617c3`
- pooled：official long-g1 +0.6336 pct；legacy point +0.6362 pct。
- 分折：+0.1911/+2.0119/+0.1571 pct；near/mid1/mid2/far 全正。
- 工况：ordinary +0.3845 pct；pre/active/recovery 也均为正。
- 事件：episode 12 +2.8225 pct，episode 13 -0.5877 pct，因此改善比例仅 50%。
- 决定：不按高气柜/低燃料的事后模式做门控；不再扫树窗或融合。冻结为 `promising_component_not_standalone`。
- 下一步：在 v20 old-column long-gall 上隔离共享 horizon 结构，寻找第二个独立增益。
- 报告：`docs/35_LONG_G1_SHARED_HORIZON_V2_REPORT.md`。

## D-046：预登记 v20 old-column long-gall 共享 horizon 实验

- 日期：2026-10-04
- 状态：代码、68 项 pytest、静态编译与只读 preflight 已通过；待人工运行
- 控制：折内重建 v20 per-horizon old-column tree + fuel + climatology + causal gate。
- 唯一变化：96 棵 gall 树替换为一个共享 horizon gall 树。
- 安全边界：不恢复 BF3/AH3 到 gall；v35 `cheat_detected` 血统永久排除。
- 规模：288 个控制模型 +3 个共享模型；不生成 ZIP。
- preflight：old features=53、shared features=64、models=291、`restored_gall_columns=false`。
- 组合门槛：gall 至少 +0.40 pct 且跨折/near/far/episode 安全，才与 g1 v2 组成候选。
- 协议：`docs/36_LONG_GALL_SHARED_HORIZON_PROTOCOL.md`。

## D-047：long-gall 共享树未过自动门禁，按用户授权制作单区块平台探针

- 日期：2026-10-04
- run：`20261004T125203Z_long_gall_shared_c83510959a`
- pooled：official long-gall +0.3911 pct；legacy point +0.3990 pct。
- 分折：+0.0654/+0.7779/+0.4589 pct；普通工况 +0.1252 pct；2/2 episode 改善。
- 分段：near -0.2304、mid1 +0.7628、mid2 +0.4683、far +0.3322 pct。
- 裁决：组件收益为正但 near 安全门禁失败，且比组合阈值低 0.0089 pct，不自动晋级。
- 用户授权：允许消耗一次平台机会；只替换 v29b long-gall，short 与 long-g1 冻结，作为可归因探针。
- 停止条件：平台不高于 63.4621 即关闭该提交路线；不做 near 权重或事后门控扫描。
- 协议：`docs/37_LONG_GALL_SHARED_PLATFORM_PROBE.md`。

## D-048：long-gall 平台 63.4539 与控制持平，关闭该路线

- 日期：2026-10-04
- submission：`20261004T211137_long_gall_shared_probe_v1_13414ee11a`
- 官方结果：63.4539；相对 v29b 63.4621 为 -0.0082。
- 解释：差值落在已记录的平台 ±0.1～0.2 噪声内，不构成提升证据。
- 归因：short 与 long-g1 冻结，故只能确认 shared long-gall 未稳定迁移到十月。
- 决定：关闭 gall 共享树及 near 权重/门控搜索；不与 g1 v2 组合。
- 下一步：只测试 long-g1 shared v2 单区块；其余三块保持 v29b。

## D-049：预登记 long-g1 shared v2 单区块平台校准

- 日期：2026-10-04
- 来源 run：`20261004T122856Z_long_g1_shared_85d58617c3`
- 本地证据：pooled +0.6336 pct；三折 +0.1911/+2.0119/+0.1571；四个 horizon 段全正。
- 唯一变化：v29b long-g1 替换为 8–9 月共享树 + 冻结 v28 fuel；short 与 long-gall 冻结。
- 预期边界：按历史局部斜率仅作算术参考，完整迁移约可到 64.4，不承诺达到 65。
- 门禁：构建器、72 项 pytest、静态编译和只读 preflight 已通过。
- 停止条件：平台不高于 63.4621 则关闭 shared horizon 全路线；不得扫描树窗或 fuel 权重。
- 协议：`docs/38_LONG_G1_SHARED_PLATFORM_PROBE.md`。

## D-050：long-g1 shared v2 官方 63.1044，关闭共享 horizon 路线

- 日期：2026-10-04
- submission：`20261004T222400_long_g1_shared_probe_v1_c43f5871b9`
- 官方结果：63.1044；相对 v29b 63.4621 为 -0.3577。
- 归因：short 与 long-gall 冻结，故退化可归因于 shared long-g1 整体替换。
- 裁决：g1 与 gall 两个 shared horizon 探针均未提升；关闭共享结构、窗口和融合权重搜索。
- 新方向：验证答疑明确允许的“上一预测作为下一起点输入”，进行因果重叠轨迹一致性校正。

## D-051：因果重叠校正稳定但收益不足，转入十月相似度验证校准

- 日期：2026-10-04
- run：`20261004T144354Z_overlap_reconcile_1a82967010`
- primary g1 α=0.50：+0.0972 pct；三折全正、2/2 episode 改善、四段全正，但未到 +0.30。
- gall α=0.50：+0.0999 pct；near -0.1549 pct，安全门禁失败。
- 敏感性：g1 α=0.25 为 +0.2209 pct，但协议禁止事后改选，且收益仍不足。
- 决定：不打包，不继续扫描递推权重或聚合器。
- 下一步：建立 test-similarity 加权 OOF，并先要求它能解释 dynamic/shared-g1/shared-gall 三次平台结果。
- 报告：`docs/40_CAUSAL_OVERLAP_RECONCILIATION_REPORT.md`。

## D-052：测试相似度权重无法解释平台方向，关闭该验证路线

- 日期：2026-10-04
- run：`20261004T145819Z_test_similarity_d7b3a8efa8`
- 域分类：158 个 process-only 因果特征，grouped-CV AUC 0.951865；最小有效样本比例 0.732631。
- 已知探针：dynamic fuel、shared g1、shared gall 的相似度加权 OOF 仍分别为 +3.9937、+0.6566、+0.4072 pct。
- 平台事实：三个探针实际为 -0.6551、-0.3577、-0.0082 分；预登记的负/负/中性方向全部识别失败。
- 门禁：`direction_calibration_passed=False`，`calibration_gate_passed=False`。
- 决定：不调域模型、特征、裁剪或 top-k；该权重不得用于候选选择，不生成提交包。
- 下一步：冻结 v28 long-g1 配方，只验证树模型的 MAPE 对齐训练权重，修复训练目标与官方指标不一致这一独立问题。
- 报告：`docs/42_TEST_SIMILARITY_CALIBRATION_REPORT.md`。

## D-053：预登记 v28 long-g1 MAPE 对齐训练实验

- 日期：2026-10-04
- 状态：协议、配置、实现与单测已准备；待人工运行
- 控制：冻结 anomaly-weight run 的 v28 `control_weight1` OOF，文件哈希固定。
- 唯一变化：候选树使用 `1/max(abs(y), 20 MW)` 的均值归一样本权重，使 weighted L1 对齐 MAPE。
- 冻结：九月树窗、legacy point 标签、restored 特征、LightGBM 参数、fuel OLS、0.8/0.6 融合和三个验证折。
- 规模：3 折 ×96 horizon =288 个 CPU LightGBM；不生成提交包。
- 门禁：组件 +0.30 pct、三折非负、ordinary/near/far/short/episode 安全；+0.80 pct 才可考虑单区块提交。
- 停止条件：失败后不扫描 denominator floor、权重指数或树窗。
- 协议：`docs/43_LONG_G1_MAPE_OBJECTIVE_PROTOCOL.md`。

## D-054：MAPE 对齐树跨折失稳并总体退化，判负封存

- 日期：2026-10-04
- run：`20261004T151651Z_long_g1_mape_00babc641a`
- pooled long-g1：-0.0997 pct；short 投影 -0.0971 pct；legacy point long -0.0980 pct。
- 分折：BF3-active -0.4558、late September +0.1276、holder1-active +0.1049 pct。
- ordinary：-0.3258 pct；near/far：-0.0971/-0.1376 pct；2/2 episode 改善。
- 门禁：总体、最差折、ordinary、near/far 均失败；`standalone_submission_eligible=False`。
- 决定：不打包；不扫描 20 MW 下限、权重指数或树窗，关闭树目标重加权路线。
- 下一步：只对占最终预测 60%–80% 的 fuel proxy 做 MAPE 对齐 weighted-median 回归，树和融合逐值冻结。
- 报告：`docs/44_LONG_G1_MAPE_OBJECTIVE_REPORT.md`。

## D-055：预登记 v28 long-g1 MAPE 对齐 fuel proxy 实验

- 日期：2026-10-04
- 状态：协议、配置、代码和单测已准备；待运行
- 控制：冻结 v28 `control_weight1` OOF；树预测通过控制结果逐值保持不变。
- 唯一变化：九月 g1 fuel proxy 从 OLS 改为 inverse-target weighted median linear regression。
- 固定参数：quantile 0.5、alpha 0、solver highs、分母下限 20 MW；禁止事后扫描。
- 重构式：`candidate = control + fuel_weight × (median_fuel - ols_fuel)`；融合权重保持 0.8/0.6。
- 规模：3 个小型线性规划模型，0 个 LightGBM；不生成 ZIP。
- 门禁：组件 +0.30 pct、三折和各安全视图通过；+0.80 pct 才可考虑平台候选。
- 协议：`docs/45_LONG_G1_MAPE_FUEL_PROTOCOL.md`。

## D-056：MAPE fuel 仅小幅正向且跨折不稳，判负封存

- 日期：2026-10-04
- run：`20261004T152405Z_long_g1_mape_fuel_20b0cc90f0`
- pooled long-g1：+0.1085 pct；short 投影 -0.0852 pct；legacy point long +0.1083 pct。
- 分折：+0.0189/+0.6660/-0.1735 pct；ordinary +0.1018 pct；2/2 episode 改善。
- horizon：near -0.0852、mid1 +0.0111、mid2 +0.2362、far +0.1094 pct。
- 门禁：组件增益不足且最差折为负；`component_gate_passed=False`。
- 决定：不打包，不扫描 quantile、alpha、下限或 fuel 窗口，关闭 fuel 损失函数路线。
- 报告：`docs/46_LONG_G1_MAPE_FUEL_REPORT.md`。

## D-057：九月与十月电价逐时完全相同，电价消融在训练前关闭

- 日期：2026-10-04
- 来源：官方 `price.xlsx` 只读审计，48 个半小时时段 ×12 月。
- 事实：9 月与 10 月 48 个价格逐格一致，均为 0.22/0.53/0.92 三档。
- 控制已知信息：v28 为 per-horizon 树，已有起点小时正余弦；对固定 horizon 可确定目标时刻价格。
- 决定：电价只是现有时间编码的确定性重表达，不投入 288 模型；不建立 run、不生成 ZIP。
- 报告：`docs/47_PRICE_FEATURE_AUDIT.md`。

## D-058：预登记节假日燃料残差跨块迁移诊断

- 日期：2026-10-04
- 状态：协议、日期来源、配置、实现和单测已准备；待运行
- 日历：劳动节可观察段 5 月 3–5 日、端午节 5 月 31 日–6 月 2 日；国庆/中秋测试段 10 月 1–8 日。
- 方法：在非节假日拟合 g1 fuel OLS；劳动节学一个 MAPE 最优 scale 验证端午节，再反向验证。
- 角色：diagnostic；不直接修改 v29b，不使用测试目标或平台反馈。
- 门禁：双向均正、平均至少 +0.50 pct、scale 同侧且差不超过 0.05、每块至少 200 行。
- 协议：`docs/48_HOLIDAY_EFFECT_AUDIT_PROTOCOL.md`。

## D-059：节假日燃料残差双向迁移通过，允许一个保守平台候选

- 日期：2026-10-04
- run：`20261004T153231Z_holiday_audit_75f9b001b3`
- scale：劳动节 0.908385，端午节 0.951977；同为下修且差 0.043592。
- 双向迁移：+0.2127/+3.9076 pct，平均 +2.0601 pct；两个块各 288 行。
- 门禁：全部通过，`diagnostic_gate_passed=True`。
- 边界：仅为 fuel residual diagnostic，不等于 v29b OOF，也不保证国庆迁移。
- 决定：只允许一个 scale=0.95 的 long-g1 holiday fuel component 候选；禁止幅度扫描。
- 报告：`docs/49_HOLIDAY_EFFECT_AUDIT_REPORT.md`。

## D-060：预登记国庆目标时刻 long-g1 0.95 fuel 修正候选

- 日期：2026-10-04
- 控制：v29b 63.4621。
- 唯一变化：目标区间起点位于 10 月 1–8 日时，将 long-g1 的九月 OLS fuel component 乘 0.95。
- 冻结：short 全文件、long-gall、非假日 long-g1 和 v28 0.8/0.6 horizon 权重。
- 状态：构建、完整性和物理检查待执行；不自动上传。
- 停止条件：官方不高于控制即关闭路线，不扫描 scale。
- 协议：`docs/50_HOLIDAY_LONG_G1_PROBE_PROTOCOL.md`。

## D-061：国庆 long-g1 单一探针通过构建与独立验收

- 日期：2026-10-04
- submission：`20261004T234025_holiday_long_g1_probe_v1_239cbda25b`
- 唯一变化：只下调目标区间起点位于 10 月 1–8 日的 v28 long-g1 fuel component，固定 scale=0.95。
- 变更规模：69,168/92,160 个 long-g1 单元；平均绝对变化 3.7472 MW，最大 5.7348 MW。
- 冻结验证：short 字节级一致；long-gall 和 22,992 个非假日 long-g1 单元逐值一致。
- 完整性：82 项 pytest；独立 verifier 960×17、960×193、CRC、成员和哈希全部通过。
- 决定：作为唯一节假日平台探针允许人工上传；禁止邻域幅度和日期扫描。
- 报告：`docs/51_HOLIDAY_LONG_G1_PROBE_REPORT.md`。

## D-062：国庆 long-g1 修正平台退化，关闭节假日路线

- 日期：2026-10-04
- submission：`20261004T234025_holiday_long_g1_probe_v1_239cbda25b`
- 官方分数：63.3013；v29b 控制 63.4621；差值 -0.1608。
- 归因边界：short、long-gall 和非假日 long-g1 均冻结，退化只能归因于 holiday-target long-g1 修正整体；无平台分项，不能继续拆分日期或 horizon。
- 决定：严格执行停止条件，关闭节假日 fuel scale 路线；不扫描 scale、日期边界或混合权重。
- 当前官方最佳仍为 v29b 63.4621。

## D-063：预登记 short-g1 动态燃料隔离实验

- 日期：2026-10-04
- 背景：动态燃料此前只做 long-g1 平台探针；其 15–120 分钟 OOF 信号尚未对 v16 short-g1 控制直接检验。
- 唯一变化：候选固定为 70% horizon-specific dynamic-fuel OLS + 30% v16 OOF 控制。
- 范围：3 折 ×8 horizon，共 24 个 OLS；只评估 short-g1，不生成 ZIP。
- 门禁：pooled 至少 +1.0 pct，最差折/horizon ≥-0.20 pct，普通工况和 episode 安全。
- 协议：`docs/52_SHORT_G1_DYNAMIC_FUEL_PROTOCOL.md`。

## D-064：short 动态燃料 pooled 强正但单折失稳，不直接提交

- 日期：2026-10-04
- run：`20261004T155802Z_short_dynamic_fuel_87286d8028`
- pooled：+1.4365 pct；8 个 horizon 全正，最差 +0.9016 pct；普通工况 +2.4547 pct。
- 分折：+3.0297/+6.1319/-3.2870 pct，最差折不满足稳定性门禁。
- 勘误：初版空字符串被误计为 episode；真实事件为 2 个、1 胜1负，不改变门禁失败结论。
- 决定：原始候选不打包；只允许一次有业务尺度依据的对称 ±5 MW 限幅，不扫描阈值。
- 报告：`docs/53_SHORT_G1_DYNAMIC_FUEL_REPORT.md`。

## D-065：预登记 short 动态燃料 ±5 MW 安全限幅

- 日期：2026-10-04
- 唯一变化：冻结源 OOF，仅把 70/30 候选相对 v16 控制的修正裁剪到 [-5,+5] MW。
- 依据：训练事件后 2–6 小时典型 g1 变化约 4.5–5.7 MW。
- 状态：轻量评估器与回归测试已准备；禁止 3/7/10 MW 扫描。
- 协议：`docs/54_SHORT_G1_DYNAMIC_FUEL_CAP_PROTOCOL.md`。

## D-066：±5 MW 限幅仍以 0.0535 pct 未过最差折门禁，关闭路线

- 日期：2026-10-05
- run：`20261004T160243Z_short_dynamic_cap_896d8696e5`
- pooled +1.4950 pct；最差 horizon +1.3971 pct；普通工况 +1.9267 pct。
- 分折：+2.0029/+3.3557/-0.2535 pct；预登记最差折下限为 -0.20 pct。
- 事件：真实 2 个 episode，1 胜1负。
- 决定：门禁失败，不打包；不放宽阈值、不扫描 cap 或 blend，关闭 short 动态燃料路线。
- 报告：`docs/55_SHORT_G1_DYNAMIC_FUEL_CAP_REPORT.md`。

## D-067：预登记 short-g1 一小时燃料趋势外推

- 日期：2026-10-05
- 假设：低自由度的最近一小时耗气趋势外推，比跨工况 dynamic OLS 更稳定。
- 唯一变化：按 v16 0.8 fuel 权重将 fuel proxy 的一小时阻尼趋势叠加到 short-g1，修正限幅 ±5 MW。
- 冻结：lag=60 分钟，60 分钟达到完整趋势，h>60 不继续放大；不做参数扫描。
- 门禁：pooled +0.50 pct、最差折/horizon ≥-0.10 pct、普通和事件安全。
- 协议：`docs/56_SHORT_G1_FUEL_TREND_PROTOCOL.md`。

## D-068：燃料趋势首个 run 因控制名称契约失败，修复后重跑

- 日期：2026-10-05
- failed run：`20261004T160900Z_short_fuel_trend_d6f12f9ba2`
- 异常：`v16_legacy_columns` 未规范化为统一门禁使用的 `v16_control`。
- 结果边界：失败发生在指标查询阶段，不登记实验收益或损失。
- 修复：控制加载器显式设置统一 variant，并加入回归测试；保留失败 run，不覆盖。

## D-069：一小时燃料趋势全面负向，关闭外推路线

- 日期：2026-10-05
- run：`20261004T161012Z_short_fuel_trend_99a68f538d`
- pooled -0.3171 pct；最差折 -0.4218；最差 horizon -0.4076；普通工况 -0.2215 pct。
- 事件：2 个 episode 均未改善。
- 门禁：所有关键性能门禁失败，`component_gate_passed=False`。
- 决定：不打包；不扫描滞后、阻尼、权重或 cap，关闭 short fuel-trend 路线。
- 报告：`docs/57_SHORT_G1_FUEL_TREND_REPORT.md`。

## D-070：预登记 short-g1 扩展时间专家门控

- 日期：2026-10-05
- 假设：起点工况和专家分歧可识别 ±5 MW 动态专家的适用区间。
- 协议：wf04 只用 wf03 训练；wf05 只用 wf03+wf04，禁止未来折反向训练。
- 模型：20 特征 L2 logistic，固定 C=0.1；概率软融合，无阈值搜索。
- 门禁：pooled +0.50 pct、逐折/逐 horizon/普通工况不退化、AUC≥0.55、事件安全。
- 协议文档：`docs/58_SHORT_G1_EXPERT_GATE_PROTOCOL.md`。

## D-071：专家门控在最后制度切换反向失效，关闭路线

- 日期：2026-10-05
- run：`20261004T161648Z_short_expert_gate_76bb9085e9`
- pooled +0.6790 pct；最差 horizon +0.4806；普通工况 +1.3951 pct。
- 分折：wf04 +2.1464，wf05 -0.2992 pct。
- 门控：wf04 AUC 0.6284，wf05 AUC 0.3533；pooled AUC 0.3097，未达到 0.55。
- 机制：wf05 动态胜率降至 46.88%，门控平均概率仍为 0.8953，未识别制度切换。
- 决定：不打包，不反转概率或搜索阈值，关闭 expert-gate 路线。
- 报告：`docs/59_SHORT_G1_EXPERT_GATE_REPORT.md`。

## D-072：用户授权一次 short-g1 ±5 MW 失败门禁平台探针

- 日期：2026-10-05
- 来源 run：`20261004T160243Z_short_dynamic_cap_896d8696e5`
- 例外依据：用户已知该候选最差折 -0.2535 pct、未通过 -0.20 pct 门禁，仍明确要求一次平台验证。
- 唯一变化：`v29b + clip(0.70 × (dynamic OLS - v29b), -5,+5) MW`，仅 short-g1。
- 冻结：short-gall 逐值不变；完整 long CSV 与 v29b 字节级不变。
- 约束：只构建固定候选，不扫描阈值、窗口或权重；平台结果必须如实登记。
- submission：`20261005T141706_short_dynamic_cap_probe_v1_6399bf5d47`
- 报告：`docs/60_SHORT_G1_DYNAMIC_CAP_PLATFORM_PROBE.md`。

## D-073：short-g1 动态燃料限幅平台判负，关闭全路线

- 日期：2026-10-05
- submission：`20261005T141706_short_dynamic_cap_probe_v1_6399bf5d47`
- 官方结果：62.7247；相对 v29b 63.4621 为 -0.7374。
- 归因边界：long 字节级冻结、short-gall 逐值冻结，只能判定 short-g1 动态限幅整体方向有害；无平台分项，不反推组件精确分数。
- 证据更新：本地 pooled +1.4950 pct 未迁移，且负差远大于轻微平台波动，当前 OOF 口径再次方向性误判十月。
- 决定：关闭 dynamic short-g1、±5 MW cap、专家门控及相关参数搜索；不得把该组件与其他平台判负组件组合。
- 下一步原则：回到 v29b 控制，只探索机制独立、严格隔离、能解释十月制度差异的新组件。

## D-074：预登记四个十日伪测试的历史相似工况轨迹实验

- 日期：2026-10-05
- 背景：多个三日 OOF 正收益组件在平台均判负，需要先扩大连续验证窗口并改变预测机制。
- 假设：起点前因果过程状态的历史近邻，其后 24 小时 g1 轨迹可提供共享树缺失的台阶信息。
- 验证：7 月、8 月、9 月初、9 月下旬共四个连续十日块；历史邻居完整轨迹必须在块开始前结束。
- 唯一候选：25% 历史相似轨迹 + 75% 同特征 shared-horizon tree；所有检索参数预先固定。
- 门禁：pooled +0.8 pct、最差块不退化、至少 3/4 块改善、near/far 安全、邻居日期多样。
- 状态：94 项测试和只读 preflight 已通过；等待用户运行 CPU 正式实验。
- 协议：`docs/61_ANALOG_TRAJECTORY_G1_PROTOCOL.md`。

## D-075：相似轨迹收益过小且跨十日块失稳，判负封存

- 日期：2026-10-05
- run：`20261005T080116Z_analog_trajectory_g1_4e4f828951`
- pooled：融合候选 +0.057671 pct；纯 analog 相对控制 -0.962892 pct。
- 分块：-0.466449/+0.727684/+0.010845/-0.041444 pct，仅 2/4 改善。
- horizon：四段虽均正，但从 near +0.132607 收缩至 far +0.034284 pct，没有远端台阶优势。
- 邻居：平均有效邻居约 23.6，至少 12 个来源日；9 月下旬距离较近仍退化，不能归因于邻居不足。
- 决定：不生成 ZIP，不扫描 K、日上限、距离或融合权重，关闭起点摘要 KNN/相似日路线。
- 下一方向：先审计现有 PyTorch 环境和预算，再考虑固定架构的完整历史窗直接多步序列模型。
- 报告：`docs/62_ANALOG_TRAJECTORY_G1_REPORT.md`。

## D-076：预登记固定架构的 24h→96 步直接 GRU

- 日期：2026-10-05
- 假设：完整 24h 过程演化包含起点摘要丢失的时序信息，可改善 long-g1 未来轨迹。
- 输入：96 步×25 特征；21 个因果过程特征加每步 4 个周期特征；无真实目标历史、无未来过程量。
- 模型：两层单向 GRU、hidden=96、100,800 参数、直接输出 96 步；log-target SmoothL1。
- 训练：30 epochs、batch 256、AdamW、单种子 20261005；禁止架构和训练参数搜索。
- 唯一候选：25% GRU + 75% 四个十日块 shared-tree 控制。
- 门禁：pooled +0.8 pct、最差块不退化、≥3/4 块改善、near/far 安全。
- 状态：专项测试和 CUDA preflight 通过；等待用户运行 GPU 正式实验。
- 协议：`docs/63_SEQUENCE_GRU_G1_PROTOCOL.md`。

## D-077：直接 GRU 独立预测显著退化，关闭深度序列调参路线

- 日期：2026-10-05
- run：`20261005T083741Z_sequence_gru_g1_c325928f68`
- pooled：25% GRU 融合相对 shared tree 仅 +0.006482 pct；独立 GRU -1.911511 pct。
- 分块：融合 +0.440724/-1.184428/+0.712850/+0.057519 pct；最差块未过门禁。
- horizon：near/mid1/mid2 为负，仅 far +0.127236 pct；不存在稳定全程优势。
- 训练诊断：四折 loss 均正常下降，工件和模型哈希全部通过；失败属于泛化而非运行故障。
- 决定：不打包；不扫描 epoch、hidden、窗口、损失、种子或融合权重，关闭当前 GRU 路线。
- 下一方向：先诊断气柜边界距离、贴边持续时间和逼近速度是否具有跨十日块领先关系。
- 报告：`docs/64_SEQUENCE_GRU_G1_REPORT.md`。

## D-078：预登记 holder_2 折内边界压力诊断

- 日期：2026-10-05
- 角色：diagnostic；零模型、零提交，只测特征与冻结控制残差的关系。
- 假设：折内 holder_2 边界压力（贴近历史边界、持续贴边、多尺度逼近速度、煤气平衡状态，
  固定权重 0.45/0.25/0.20/0.10）与 shared-tree 控制远端（h735–1440）long-g1 残差存在
  稳定正向关系。
- 控制：run `20261005T080116Z_analog_trajectory_g1_4e4f828951` 的四块 shared tree OOF，
  manifest 与 SHA-256 强校验。
- 输入：event feature registry 的 5 个因果列（level、delta_1h/4h/12h、balance proxy），
  最大来源偏移 0，无目标历史；边界合同（5%/95% 参考、10%/90% 贴边区、IQR 缩放、
  96 步贴边上限、25%/75% 压力分组）只用块前 ≥14 天历史拟合。
- 门禁：pooled Spearman ≥ 0.08；≥75% 块为正；pooled 高−低压组远端残差差 ≥ 1.0 pct；
  ≥75% 块为正；每块极端组覆盖率 ≥ 5%。通过才允许预登记模型实验；不通过则关闭该路线，
  不回扫任何阈值。
- 验证状态：全项目 `104 passed`；preflight 通过（4 块、3840 起点、最少历史 5664 行）；
  上游工件哈希全部通过。等待人工运行正式诊断。
- 协议：`docs/65_HOLDER2_BOUNDARY_AUDIT_PROTOCOL.md`。

## D-079：正向边界假设判负、反向信号转入新鲜块确认协议

- 日期：2026-10-05
- run：`20261005T100107Z_holder2_boundary_a8309e91a1`（耗时 0.86 秒；5 个工件哈希复核一致）
- 正向门禁 4/4 条件未过：pooled Spearman **-0.1430**（阈值 ≥ +0.08）、正向块 0/4、
  pooled 高−低残差差 **-7.451 pct**（阈值 ≥ +1.0）、正向差值块 0/4；覆盖率通过。
- 分块 Spearman：-0.186/-0.214/-0.071/-0.188，四块方向一致为负——预登记正向假设
  （高边界压力 → 控制远端低估）被一致否定，正向模型路线按协议当场关闭。
- 事后观察（不得直接采用）：反向关联跨 4/4 块一致且量级大（高−低差 -2.1~-11.2 pct）；
  near 端 Spearman ≈ 0 而 far 端一致为负，属远端特异；高压组 far/near 真实负荷比
  -4.0% 对低压组 +6.0%，说明信号超前于未来 24 小时真实负荷回落，是机制而非水平偏置。
- 纪律边界：该反向假设形成于看过开发块结果之后，禁止翻符号后在相同四块上训练修正器。
- 决定：预登记新鲜块确认协议——在四个从未使用的新鲜十日块（6/11-20、6/21-30、
  7/18-27、8/21-30）上重建冻结 shared-tree 控制，应用符号翻转门禁（pooled Spearman
  ≤ -0.08、高−低差 ≤ -1.0 pct、≥75% 块同向）加机制条件（压力 vs 真实远-近变化
  Spearman ≤ -0.05、≥75% 块为负）、覆盖率 ≥5%；任何一条不满足则 holder_2 路线
  （含反向）永久关闭。通过也只允许进入下一轮预登记的固定幅度高压组远端修正实验。
- 验证状态：全项目 `113 passed`；等待人工运行确认实验。
- 报告：`docs/66_HOLDER2_BOUNDARY_AUDIT_REPORT.md`；协议：`docs/67_HOLDER2_INVERSE_CONFIRM_PROTOCOL.md`。

## D-080：反向确认判负，holder_2 边界路线永久关闭；机制信号确认真实

- 日期：2026-10-05
- run：`20261005T111953Z_holder2_inverse_confirm_d28fee07b0`（正式 completed 记录；首次
  运行 `20261005T102336Z_holder2_inverse_confirm_befe6e5687` 因 runner 记账缺陷标记
  failed，其科学工件完整且与正式 run 逐位一致）
- 反向门禁 2/7 未过即判负：pooled 残差 Spearman **-0.0138**（阈值 ≤ -0.08）、
  负向块 2/4（阈值 ≥75%）、pooled 高−低残差差 **-0.422 pct**（阈值 ≤ -1.0）。
- 分块残差方向混杂：6 月 A -0.226 与 8 月 -0.111 复现开发块方向，6 月 B +0.137 与
  7 月晚 +0.071 反向——控制对机制的吸收随工况翻转，固定修正无法稳定获益。
- 机制条件大幅通过：压力 vs 真实远−近负荷变化 pooled Spearman **-0.441**、4/4 块为负
  （开发块 -0.291）——全项目迄今复现性最强的目标领先信号，但残差链接不成立。
- 决定：holder_2 边界路线（含反向、含任何阈值/分块/分桶变体）**永久关闭**，不回扫。
  新鲜块自此视为开发数据；≥30 天历史的十日伪测试窗口已耗尽，后续验证设计须另行声明。
  让控制稳定吸收该机制的模型重设计属于新路线，须全新预登记。
- 下一方向：转向十月分布偏移定位诊断——用测试期过程量（仅过程量，无目标）逐列对比
  九月/全训练期分布，定位树模型外推失效区域，为下一个预登记实验提供依据。
- 报告：`docs/68_HOLDER2_INVERSE_CONFIRM_REPORT.md`。
