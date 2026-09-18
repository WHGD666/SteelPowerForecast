# 数据审计计划 v0

本文件定义审计范围和放行门禁。审计脚本已执行，边界重复和当前时刻目标观测已按官方滚动答复纳入 v1 信息合同；当前状态为 WARN，尚未进入模型训练。

## 审计范围

1. 对所有原始文件计算 SHA-256 指纹。
2. 核对文件编码、字段名、行列数、数据类型和时间范围。
3. 检查时间戳解析、排序、重复、缺口和采样间隔。
4. 统计字段缺失率、整列为空、零值、负值、常数列和异常范围。
5. 检查四表按 `datetime` 对齐后的覆盖率。
6. 检查训练/测试边界重叠和测试目标可见性。
7. 检查直接目标、确定性代理、后验字段和目标可重构关系。
8. 对煤气平衡、气柜状态和发电耗气做诊断性一致性检查。

## 必须输出的审计结论

- 每个任务的暂定目标和评价单位。
- 允许字段、禁止字段和仅诊断字段。
- 缺失和异常处理原则。
- 时间滚动验证的候选切分。
- 未解决的单位、容量、字段定义和提交格式问题。

## 通过条件

- 所有源文件均有指纹和结构记录。
- 时间边界和表连接关系可解释。
- 测试标签不进入开发特征或模型选择。
- 每个异常字段都有保留、修复或剔除理由。
- 协议草案中的待确认问题被标记为已确认或阻塞。

## 已执行脚本

```powershell
D:\anaconda\envs\vocs\python.exe src/data_audit.py --root . --out outputs/data_audit_v0
```

机器可读结果保存在本地 `outputs/data_audit_v0/`，包括 `audit_report.json`、`audit_report.md` 和 `csv_summary.csv`。该目录被 Git 忽略，不上传原始数据或测试标签。

初赛质量项筛查命令：

```powershell
D:\anaconda\envs\vocs\python.exe src/data_quality_audit.py --root . --out outputs/data_quality_audit_v1
```

该筛查只生成缺失、重复、无效列和统计异常候选证据，不自动删除或修正原始值。详见 [data_quality_audit_v1.md](data_quality_audit_v1.md)。

因果清洗与派生数据生成：

```powershell
D:\anaconda\envs\vocs\python.exe src/prepare_dataset.py --root . --config configs/cleaning_v1.yaml
```

该步骤只在 `outputs/prepared_v1/` 生成派生文件；目标原值不填，协变量仅使用最多 4 步的历史前向填充，并保留缺失与边界标志。详见 [cleaning_policy_v1.md](cleaning_policy_v1.md)。
