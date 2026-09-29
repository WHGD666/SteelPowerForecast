# 提交与平台实验协议

## 1. 提交资格

只有 `experiment_role` 为 `competition` 或 `final` 的 run 才能产生提交包。

提交前必须满足：

- 协议、标签、切分和指标版本已记录。
- 无未来信息。
- 所有预测值有限且无缺失。
- 960 个起点完整、唯一、升序。
- `s_result.csv` 精确 17 列。
- `l_result.csv` 精确 193 列。
- 两目标和 horizon 列顺序与合同一致。
- 物理边界和 `generator_1 <= generator_all` 已检查。
- 提交包可以从 run 产物重新生成。

## 2. submission_id

推荐格式：

```text
YYYYMMDD_slotNN_<short-description>_<short-hash>
```

示例：

```text
20260930_slot02_correct_interval_labels_12ab34cd
```

## 3. 提交目录

每次提交使用独立目录：

```text
submissions/round2_v3/<submission_id>/
├── s_result.csv
├── l_result.csv
├── submission_manifest.yaml
├── validation_report.json
└── teamname_gas_predict_final.zip
```

已经上传的目录和文件不可修改。需要修复时创建新的 submission_id。

## 4. 提交 manifest

至少记录：

- submission_id。
- source_run_id。
- 提交假设和唯一主要变化。
- s/l 文件与 ZIP 的 SHA-256。
- 行列数、时间范围和预测范围。
- 物理检查结果。
- 生成命令、Git commit、环境版本。
- 上传时间、当天 slot。
- 平台总分和短/长、g1/gall 明细。
- 是否成为新的 official best。

## 5. 每日五次机会的使用

建议顺序：

1. 锚点或控制组，只在需要确认平台环境时重复。
2. 只改短文件的高价值假设。
3. 只改长文件的高价值假设。
4. 只改 g1 或某个 horizon 段的定向假设。
5. 组合当天已经获得正证据的版本。

没有明确假设时不必用满五次。不得把平台当成随机超参数搜索器。

## 6. 单变量提交原则

为了正确归因：

- 测短模型时固定 `l_result.csv`。
- 测长模型时固定 `s_result.csv`。
- 测 g1 时保持 gall 完全不变。
- 测 gall 时保持 g1 完全不变。
- 测后处理时保持基础预测完全一致。

若平台只返回总分，单变量原则尤其重要。

## 7. 提交后记录

每次平台返回后立即追加：

- 实际平台分数和时间。
- 排名快照。
- 四项准确率明细（若提供）。
- 异常审查状态。
- 与控制组差异。
- 是否支持原假设。
- 下一步决策。

低分提交不得删除。排行榜使用历史最高分不等于可以丢失低分实验的证据。

## 8. 最终复现

最终最高分版本必须能在干净工作目录中：

1. 读取官方原始数据。
2. 运行预处理和特征生成。
3. 训练或载入模型。
4. 完成滚动推理。
5. 生成哈希一致或合理数值误差范围内一致的 s/l 文件。
6. 通过全部提交合同检查。

若平台最高分与最终代码无法合理复现，视为严重审计风险。
