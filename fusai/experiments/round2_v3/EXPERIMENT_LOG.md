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

## 记录模板

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
