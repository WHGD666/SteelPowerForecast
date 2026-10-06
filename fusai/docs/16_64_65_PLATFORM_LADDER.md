# 64–65 冲分实验：v29b × v34 单组件平台阶梯

> **2026-10-04 已取消，禁止按本文继续上传。** 最新官方档案确认 v29b 已得 63.4621；v34b（冻结 v29b short/gall、只替换 long g1 为 v34）仅得 62.9325，long g1 从 82.96% 降至 82.28%。因此本文原计划的 v29b×v34 long-g1 权重阶梯已被平台证据否决。历史设计保留用于审计，当前策略见 `docs/17_TOP3_AND_50_TRIALS_REVIEW.md`。

## 1. 当前目标

当前官方控制组是 v28 / 61.5824。冲击 64–65 不以复刻 v28 为前置条件，而是利用最新版交接包中已经冻结的高价值组件：

- v29b：组合了历史官方分项最强的 gall 与 g1，整包预估约 63.45，但未提交；
- v34：开发折上只改 `generator_1`，short/long `generator_all` 与 v28 完全相同。

本实验以 v29b 为底座，把 v29b 的 `generator_all` 完全冻结，只在 `generator_1` 上与 v34 做凸组合。这样平台即使只返回总分，也能判断所改变 g1 区块的方向。

## 2. 为什么不先做大模型

当前最强证据不是“换一个新模型”，而是历史平台分项已经显示：

- v20 gall 明显强于 v28 gall；
- v16_final short g1 和 v28 long g1 已有平台证据；
- v34 试图进一步改善 g1，但尚无平台验证。

因此第一轮最划算的是先把这些组件的真实平台增益测清楚。直接训练 Transformer、LSTM 或重做整套 LightGBM 会同时改变多个区块，既耗时，也无法从总分判断成功原因。

## 3. 固定实验合同

- base：v29b。
- alternate：v34。
- 永久冻结：short/long `generator_all` 全部列，逐值等于 v29b。
- 唯一可变：short/long `generator_1` 的 v34 权重。
- 权重候选：0、0.25、0.5、1.0。
- 所有候选必须通过列顺序、960 个起点、有限值、正值、`g1 <= gall` 和 ZIP 成员检查。
- 不从平台总分反推隐藏标签或分项目标值。

## 4. 自适应五次提交阶梯

### Slot 1：v29b 锚点

```text
short g1 v34 weight = 0
long  g1 v34 weight = 0
```

目的：获得 v29b 整包真实总分。63.45 只是预估，不以它作为事实。

### Slot 2：只改 long g1，权重 0.5

```text
short g1 保持 v29b
long  g1 = 0.5*v29b + 0.5*v34
```

这是最重要的单变量平台探针。

### Slot 3：根据 Slot 2 自适应

- 如果 Slot 2 高于锚点：测试 long 权重 1.0。
- 如果 Slot 2 低于锚点：测试 long 权重 0.25，不直接跳到 1.0。
- 如果几乎持平：保留剩余次数，先检查差值是否小于平台噪声或四舍五入影响。

### Slot 4：在最佳 long 权重上测试 short 权重 0.5

只改变 short g1，long 和两个 gall 全部保持上一最佳版本。

### Slot 5：再次自适应

- short 0.5 改善：测试 short 1.0；
- short 0.5 退化：测试 short 0.25；
- 若前面出现异常分数：停止，不为用完次数而提交。

## 5. 成功与停止判据

成功不是“某个离线版本看起来更先进”，而是：

1. v29b 锚点真实总分显著高于 61.5824；
2. 只改 long g1 后平台总分继续提升；
3. 只改 short g1 后再次提升；
4. 最终候选通过完整结构和哈希验证，达到 64–65。

出现以下情况停止阶梯：

- v29b 锚点明显低于预期，说明分项拼装推算不可直接相加；
- 0.5 权重已经退化却仍想盲目扩大到 1.0；
- 同一次候选同时修改 short、long 或 gall，导致归因失效；
- 输出合同、物理边界或来源哈希任何一项失败。

## 6. 构建命令

以下命令只组合现有预测，通常数秒完成，不训练模型：

```powershell
cd D:\daima\aic\shiyan\fusai

# Slot 1：v29b 原样锚点
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.build_component_ladder `
  --short-g1-v34-weight 0 `
  --long-g1-v34-weight 0

# Slot 2：只替换一半 long g1
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.build_component_ladder `
  --short-g1-v34-weight 0 `
  --long-g1-v34-weight 0.5
```

后续权重根据平台返回结果决定，不预先盲建或盲交全部候选。每个运行目录都会保存 manifest、CSV、确定性 ZIP、源码/config/来源哈希和实际变化统计。
