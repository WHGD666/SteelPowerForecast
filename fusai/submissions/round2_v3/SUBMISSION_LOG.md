# 平台提交日志

## 历史外部基准

- 版本：学弟 v9
- 官方分数：52.4085
- 短期准确率：89.08%
- 长期准确率：82.93%
- 复现状态：incomplete
- 说明：该记录不属于 round2_v3 正式提交，只作为历史目标。

## `external_v15_60_5717`

- 证据来源：`AI复现手册_60分保底.md` 与 `复赛代码_给学长.zip`
- 手册 SHA-256：`e03d304c1def1bca8bcf37cf1f97d5e519491cae4f955ba914e3eb3ccbcc421c`
- 代码 ZIP SHA-256：`cc9067dd1b7da565dbf3f17677b68f185c8baf70f618dca97a8109e92478d728`
- 平台总分：**60.5717**
- 短期：总体 91.09%，`generator_1` 89.14%，`generator_all` 93.04%
- 长期：总体 84.47%，`generator_1` 82.49%，`generator_all` 86.44%
- 核心机制：g1 使用 20% 树模型 + 80% 九月燃料映射；远期 gall 使用 60% 混合树/起点气候项 + 40% 九月燃料映射
- 复现状态：手册与代码已取得，但本证据集中未冻结该次平台提交 ZIP，不能做提交文件字节级复核
- 作用：作为约 61 分版本之前的可解释中间基准，不再是当前最高分

## `20260929T212923_calibration_baseline_v1_91cd219d33`

- 提交时间：2026-09-29 21:44:10（北京时间）
- 平台状态：`DONE`
- 来源 run_id：`20260929T124936Z_r2v3_baseline_46e02afe85`
- 控制提交：无；这是 v3 首次外部校准
- 唯一主要变化：首次提交正确标签、因果特征和共享 horizon LightGBM 的 v3 baseline
- ZIP：`这模型调的有力量_gas_predict_final.zip`
- ZIP SHA-256：`364893241bb5937599b18d1f39780b3dc1e968849672584b81f3e7eeaaccb35a`
- 平台总分：**48.9241**
- 短期/长期及分目标平台明细：平台未返回，保持空值，禁止猜测
- 相对历史 v9：低 **3.4844** 分（48.9241 vs 52.4085）
- 异常审查：提交前 schema、行列数、时间覆盖、模型哈希和 ZIP 成员均通过；没有证据表明掉分来自格式错误
- 是否新高：否
- 假设结论：协议正确性没有自动转化为隐藏十月工况上的精度；该模型族作为高分候选被拒绝，只保留为可复现校准锚点
- 下一步：暂停同族重复提交；先复现 legacy v9 的有效机制，并在冻结切分上做单变量、近期工况优先的受控实验

## `20261001_external_best_61_pending_exact`

> 本条已由后续证据解析为 v28；保留以下原始记录作为调查历史。当前正式口径见 `external_v28_61_5824`。

- 收件日期：2026-10-01
- 平台状态：用户报告已获得约 **61 分**；精确总分、分项、提交时间和 slot 待补
- 来源 run_id：待完成源码到产物的一一对应后补录
- 控制意义：当前外部强控制组，高于已知 60.5717 版本和 v3 的 48.9241
- 机制判断：九月燃料回归层已确认；交接包中的 v15 全期树和 v16 九月树均被代表列复现否定，底层模型或额外后处理源码缺失
- ZIP：`这模型调的有力量_gas_predict_semi.zip`
- ZIP SHA-256：`78b0e4eb024e5cc1d058bb8a77a9ac506246304105bbdb070cedaaa91071b12f`
- `s_result.csv` SHA-256：`e96e633801c2d577006670ea178904e253f5e1da75faaa000a0e264eaf3b4a4f`
- `l_result.csv` SHA-256：`3c6e55d4d92b8ecee65609abb6784a9fe15178afc2f1f7cf04966372bf28de28`
- 结构检查：两个文件均为 960 个连续 15 分钟起点；短 17 列、长 193 列；数值均有限；物理边界通过；短期逐值等于长期前 8 步
- 平台总分：留空，等待精确结果，不以“约 61”伪造小数
- 是否新高：是（按用户报告，暂定）
- 复现状态：预测产物已冻结；现有手册与代码不足以重建该包，需要学弟补充 61 分版实际入口、参数或源码
- 假设结论：近期工况适配与燃料锚定明显优于全年共享 v3 树模型，但当前远期路径仍过度依赖起点燃料水平
- 下一步：冻结该版 `generator_all`；正式验证 horizon-specific 动态燃料 `generator_1`，每次只改变一个评分组件
- 详细报告：`docs/13_EXTERNAL_61_CANDIDATE_REVIEW.md`

## `external_v28_61_5824`

- 收件与来源解析日期：2026-10-01
- 平台状态：最新版交接材料记录已提交，官方总分 **61.5824**
- 来源版本：v28 / `20260930T_nogate_v28`
- 外部证据包 SHA-256：`350ca90b8db4f033cb6c8772d04ff374a11e7fba8c05a92971172aa748965cce`
- 本次包内提交 ZIP SHA-256：`028c49371cc6fed3773e3e63b2c745019e71255a602a4d3573475c978c71799f`
- 先前平台下载/重包装 ZIP SHA-256：`78b0e4eb024e5cc1d058bb8a77a9ac506246304105bbdb070cedaaa91071b12f`
- `s_result.csv` SHA-256：`e96e633801c2d577006670ea178904e253f5e1da75faaa000a0e264eaf3b4a4f`
- `l_result.csv` SHA-256：`3c6e55d4d92b8ecee65609abb6784a9fe15178afc2f1f7cf04966372bf28de28`
- 容器解释：两个 ZIP 容器哈希不同，但内部两个 CSV 完全相同，因此预测身份一致
- 结构检查：短 960×17、长 960×193；时间完整无重复；数值有限、非负；内外 CSV 一致
- 主要机制：恢复被旧流程错误排除的投运字段；g1 九月树、gall 全期树；燃料回归和远期气候融合；实际未应用 gate
- 已知分项：交接材料仅明确给出 long g1 82.96%；其余 v28 分项未取得可独立核验的完整平台表，不补猜
- 是否新高：是
- 复现状态：`source_resolved_independent_rebuild_pending`
- 下一步：收到复现指令后，在隔离工作区运行 v28，并以两个 CSV SHA-256 为唯一通过条件
- 详细报告：`docs/15_LATEST_OPTIMAL_BUNDLE_REVIEW.md`

## 未提交候选（不计入官方成绩）

- v29b：此前登记为预估候选；2026-10-04 新证据已确认其官方成绩为 63.4621，正式记录见下文。
- v34：三折开发候选；冻结 v28 gall、只改 g1；无平台成绩。

## `external_v29b_63_4621`

- 证据接收日期：2026-10-04
- 平台状态：交接档案明确记录已提交，`anomaly=normal`
- 官方总分：**63.4621**
- short：总准确率 91.61%，g1 89.47%，gall 93.74%
- long：总准确率 85.25%，g1 82.96%，gall 87.54%
- ZIP SHA-256：`297569197c6b7e6752d53670b51e89b67c793b852f50e0c791c2bb2f2662dbb9`
- `s_result.csv` SHA-256：`63c1ed03a9549ec8aeb83012f21edab221dcc74ed6a459e927e2d319fa92be76`
- `l_result.csv` SHA-256：`088282fc08e9b9ac568983f846fa65ef26c748de18576eae1cd50107d0984da1`
- 结构检查：短 960×17、长 960×193；CRC、时间、有限值和物理边界通过
- 核心配方：short g1=v16_final；long g1=v28；short/long gall=v20 门控血统
- 是否新高：是
- 复现状态：成品与来源组件已冻结；作者侧声明逐位复现，本项目尚未独立重跑完整上游链
- 后续裁决：v34b、v37、v48 均未超过该包；当前作为唯一平台控制组
- 详细报告：`docs/17_TOP3_AND_50_TRIALS_REVIEW.md`

## `20261004T153207_dynamic_fuel_probe_v1_98feef8d8a`

- 日期：2026-10-04
- 状态：已上传并返回官方结果；判负封存
- 角色：用户明确授权的高风险单区块平台探针；不是通过本地晋级门禁的正式候选
- 来源科学 run：`20261004T070858Z_dynamic_fuel_g1_31e2a08193`
- 平台控制：`external_v29b_63_4621` / 官方 63.4621
- 唯一变化：long `generator_1 = 0.70 × dynamic_fuel_ols + 0.30 × v3_baseline_long_g1`
- 完全冻结：v29b short 全文件字节级一致；long `generator_all` 逐值一致
- 模型：96 个全量 horizon-specific OLS；最近 21 天成熟标签；因果燃料历史最大来源偏移 0
- 本地证据：相对 baseline OOF 的 long-g1 提升 3.3675 pct，但普通工况和两个折退化，`promotion_passed=[]`
- 风险：不是 v29b OOF 上验证的融合；十月特征中心接近退化的 BF3-active 折；21 天窗口存在窄九月训练风险
- 变更幅度：92,160/92,160 个 long-g1 单元变化；相对 v29b 平均绝对变化 7.401705 MW，最大 41.289711 MW
- 物理检查：投影前后均无非正、越界或 g1>gall；没有实际触发裁剪
- short SHA-256：`63c1ed03a9549ec8aeb83012f21edab221dcc74ed6a459e927e2d319fa92be76`
- long SHA-256：`0bc1ac8f3e58fd8ab3c7a70c39e2c5fdaeafe36577f01b0b24618ba37900414b`
- ZIP SHA-256：`9c5e7fb36cac5f53db2053198f5180f926ccdbed7745d507979e887f34a5f1de`
- 独立验收：960×17、960×193、96 个模型、2 个 ZIP 成员全部通过
- 上传文件：`submissions/round2_v3/20261004T153207_dynamic_fuel_probe_v1_98feef8d8a/这模型调的有力量_gas_predict_semi.zip`
- 官方总分：**62.8070**；相对 v29b 63.4621 为 **-0.6551**。
- 归因边界：因为 short 与 long `generator_all` 冻结，只能确认替换 long `generator_1` 的整体方向有害；平台未返回分项，不反推 long-g1 准确率。
- 异常状态：用户未提供，登记为 `unknown_not_reported`。
- 结论：拒绝该候选；不再搜索该模型的窗口、融合权重或全时段替换变体。

## v3 提交记录模板

## `20261004T211137_long_gall_shared_probe_v1_13414ee11a`

- 日期：2026-10-04
- 状态：已上传并返回官方结果；判为与控制持平后关闭路线
- 角色：用户授权的 long-gall 单区块平台探针
- 来源 run：`20261004T125203Z_long_gall_shared_c83510959a`
- 唯一变化：只替换 v29b long `generator_all`；short 原始字节、long `generator_1` 逐值冻结
- ZIP SHA-256：`088cf72118dceb9345160090e8387f4384868f8e16edea589d52b4ba22fccf05`
- 官方总分：**63.4539**；v29b 控制 **63.4621**；差值 **-0.0082**
- 解释：差异远小于已记录的平台约 ±0.1～0.2 噪声，没有提升证据
- 结论：关闭 shared long-gall、near 权重和门控搜索；不进入组合候选

## `planned_long_g1_shared_probe_v1`

- 日期：2026-10-04
- 状态：构建器、72 项 pytest、静态编译和只读 preflight 已通过；待人工构建
- 来源 run：`20261004T122856Z_long_g1_shared_85d58617c3`
- 唯一变化：只替换 v29b long `generator_1`；short 原始字节和 long `generator_all` 逐值冻结
- 本地证据：pooled +0.6336 pct，三折和四个 horizon bucket 全正
- 风险：2 个 episode 仅 1 个改善；即使完整迁移也更可能先到约 64.4，而非直接达到 65
- 协议：`docs/38_LONG_G1_SHARED_PLATFORM_PROBE.md`

## `20261004T222400_long_g1_shared_probe_v1_c43f5871b9`

- 日期：2026-10-04
- 状态：已上传并返回官方结果；判负封存
- 来源 run：`20261004T122856Z_long_g1_shared_85d58617c3`
- 唯一变化：只替换 v29b long `generator_1`；short 原始字节、long `generator_all` 逐值冻结
- ZIP SHA-256：`5a77a0ce506ddf1e56c43b5d0dc39c6cb7836bbd2417adb18f6fc8fc2147a3df`
- 官方总分：**63.1044**；v29b 控制 **63.4621**；差值 **-0.3577**
- 归因：本地 +0.6336 pct 未迁移到十月，shared long-g1 整体替换有害
- 结论：关闭 shared horizon 的 g1/gall 全路线；不扫描训练窗、fuel 权重或组合比例

## `20261004T234025_holiday_long_g1_probe_v1_239cbda25b`

- 日期：2026-10-04
- 状态：已上传并返回官方结果；判负封存
- 角色：通过预登记诊断后生成的单一节假日平台探针
- 控制：v29b 63.4621
- 唯一变化：目标区间起点位于 10 月 1–8 日的 long-g1 fuel component 固定乘 0.95
- 完全冻结：v29b short 原始字节、long `generator_all`、非假日 long-g1
- 变更幅度：69,168 个 long-g1 单元；平均绝对变化 3.7472 MW，最大 5.7348 MW
- 物理检查：无非正、越界或 g1>gall，未触发投影
- short SHA-256：`63c1ed03a9549ec8aeb83012f21edab221dcc74ed6a459e927e2d319fa92be76`
- long SHA-256：`01c4e31ed9745d83cbb2be6b8a8b203e4fe9f184d1835835e719cb569daa990a`
- ZIP SHA-256：`380134a05ad54ce532b92f7cf958ba1dd86df0859a71a7bd16bc40c7880425e0`
- 独立验收：short 960×17、long 960×193、0 个模型、2 个 ZIP 成员全部通过
- 上传文件：`submissions/round2_v3/20261004T234025_holiday_long_g1_probe_v1_239cbda25b/这模型调的有力量_gas_predict_semi.zip`
- 官方总分：**63.3013**；v29b 控制 **63.4621**；差值 **-0.1608**
- 归因：只改变 holiday-target long-g1，因此该修正整体没有迁移；平台未给分项，不能反推逐日效果
- 异常状态：用户未提供，登记为 `unknown_not_reported`
- 裁决：判负并关闭路线；不扫描 scale、日期边界或分 horizon 幅度

## `20261005T141706_short_dynamic_cap_probe_v1_6399bf5d47`

- 日期：2026-10-05
- 状态：已上传并返回官方结果；判负封存
- 角色：用户明确授权的失败门禁高风险平台探针
- 控制：v29b 63.4621
- 来源 run：`20261004T160243Z_short_dynamic_cap_896d8696e5`
- 唯一变化：short `generator_1 = v29b + clip(0.70 × (dynamic OLS - v29b), -5,+5) MW`
- 完全冻结：short `generator_all` 逐值不变；long CSV 与 v29b 字节级一致
- 本地证据：pooled +1.4950 pct、8 个 horizon 全正；最近折 -0.2535 pct，自动门禁失败
- 修改幅度：7680 个 short-g1 单元；限幅后平均绝对修正 3.948585 MW，最大 5 MW
- 物理检查：无非正、越界或 g1>gall
- short SHA-256：`cb162f24767a5197599389c17251730f18a4bff659a8b97a03d636e9e84762b6`
- long SHA-256：`088282fc08e9b9ac568983f846fa65ef26c748de18576eae1cd50107d0984da1`
- ZIP SHA-256：`658ffde82aadcc0b69b2fa555915611f4e37c222e25947d35e9563488709ef2d`
- 独立验收：short 960×17、long 960×193、8 个模型、2 个 ZIP 成员，89 项 pytest 通过
- 上传文件：`submissions/round2_v3/20261005T141706_short_dynamic_cap_probe_v1_6399bf5d47/这模型调的有力量_gas_predict_semi.zip`
- 官方总分：**62.7247**；v29b 控制 **63.4621**；差值 **-0.7374**
- 异常状态：用户未提供，登记为 `unknown_not_reported`
- 归因：long 字节级冻结、short-gall 逐值冻结，故 short-g1 动态限幅整体替换方向有害；无分项时不反推精确 short-g1 分数
- 裁决：关闭 short 动态燃料、cap 和专家门控路线；禁止事后扫描阈值、窗口、融合比例或组合既有判负组件
- 详细报告：`docs/60_SHORT_G1_DYNAMIC_CAP_PLATFORM_PROBE.md`

### `<submission_id>`

- 日期与 slot：
- 来源 run_id：
- 控制提交：
- 唯一主要变化：
- ZIP SHA-256：
- 平台总分：
- 短期：总准确率 / g1 / gall / 文件得分
- 长期：总准确率 / g1 / gall / 文件得分
- 异常审查：
- 是否新高：
- 假设结论：
- 下一步：
