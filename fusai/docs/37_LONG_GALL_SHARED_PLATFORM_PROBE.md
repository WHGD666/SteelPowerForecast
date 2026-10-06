# long-gall 共享 horizon 平台探针

## 决策

开发 run `20261004T125203Z_long_gall_shared_c83510959a` 的 pooled long-gall 提升为
`+0.3911 pct`，三个近期折均为正、2/2 事件改善，但 near 段为 `-0.2304 pct`，且未达到
预登记的 `+0.40 pct` 组合阈值。因此该组件没有通过自动晋级门禁。

用户明确授权使用一次平台机会。本探针仅用于判断该信号能否迁移至十月：

- short 文件完全复用 v29b 原始字节；
- long `generator_1` 逐值冻结为 v29b；
- 只将 long `generator_all` 替换为全训练期重拟合的共享 horizon 树；
- fuel、origin-time climatology、causal fuel-ratio gate 与 v20 完全一致；
- 不使用 restored-column gall 血统；
- 不叠加尚未平台验证的 g1 v2，确保平台升降可以归因。

## 风险与停止条件

这是 `competition_probe`，不是已通过验证的最终候选。若平台成绩不高于 63.4621，则关闭
shared long-gall 的提交路线；不得继续扫描 near 权重、阈值或事后日期门控。若平台明确提升，才允许
另行预登记 `g1 v2 + gall shared` 组合。

## 构建命令

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.build_submission_long_gall_shared_probe --config configs\round2_v3\submission_long_gall_shared_probe_v1.yaml
```

构建器会重新训练一个约 133 万行的 CPU LightGBM，输出 ZIP、模型、测试期预测、manifest 和哈希；
不会上传平台。

## 官方结果

- submission：`20261004T211137_long_gall_shared_probe_v1_13414ee11a`
- ZIP SHA-256：`088cf72118dceb9345160090e8387f4384868f8e16edea589d52b4ba22fccf05`
- 官方总分：**63.4539**
- v29b 控制：**63.4621**
- 差值：**-0.0082**

该差值远小于档案记录的平台约 ±0.1～0.2 分噪声，统计上只能视为与控制持平，不能宣称突破。
由于唯一变化是 long `generator_all`，本次结果没有支持共享 gall 的十月增益。按预登记停止条件，关闭
shared long-gall 提交路线，不叠加到后续候选。

