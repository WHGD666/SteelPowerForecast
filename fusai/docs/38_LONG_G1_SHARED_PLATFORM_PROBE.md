# long-g1 共享 horizon v2 平台探针

## 目标与边界

long-gall 单区块平台结果 63.4539 与 v29b 63.4621 持平，故 gall 路线关闭。本探针只验证
`20261004T122856Z_long_g1_shared_85d58617c3` 的 long-g1 结构是否迁移到十月：

- v29b short 文件原始字节冻结；
- v29b long `generator_all` 逐值冻结；
- long `generator_1` 使用 8–9 月共享 horizon 树；
- 九月 fuel OLS、15/30 分钟 0.8 fuel、其余 horizon 0.6 fuel 完全冻结为 v28 公式；
- 不叠加已经平台判为中性的 shared gall。

## 证据与预期

- pooled long-g1：+0.6336 pct；
- 三折：+0.1911/+2.0119/+0.1571 pct；
- near/mid1/mid2/far 全正；ordinary 改善；
- episode 仅 1/2 改善，因此不是自动晋级的最终候选。

按历史 long 评分局部斜率粗略估计，即使 OOF 收益完整迁移，总分也更可能落在约 64.4，而不是直接
达到 65。平台结果的任务是校准共享结构是否可迁移；不得把该估计写成官方预期或保证。

## 停止条件

- 若不高于 63.4621：关闭 shared horizon 全路线；
- 若明显提高但仍低于 65：冻结该组件，下一轮只研究独立的新残差信号；
- 不提交 gall+g1 共享组合，不扫描树窗、fuel 权重或事后日期门控。

## 构建命令

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.build_submission_long_g1_shared_probe --config configs\round2_v3\submission_long_g1_shared_probe_v1.yaml
```

## 官方结果

- submission：`20261004T222400_long_g1_shared_probe_v1_c43f5871b9`
- ZIP SHA-256：`5a77a0ce506ddf1e56c43b5d0dc39c6cb7836bbd2417adb18f6fc8fc2147a3df`
- 官方总分：**63.1044**
- v29b 控制：**63.4621**
- 差值：**-0.3577**

结果明确低于平台噪声范围，说明本地 +0.6336 pct 没有迁移到十月。由于只有 long-g1 改变，
shared horizon 的 long-g1 提交路线关闭；不得继续扫描训练窗、fuel 权重或与 shared gall 组合。

