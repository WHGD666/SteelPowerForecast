# round2_v3 配置区

计划包含：

- `protocol_v3.yaml`：任务、目标、允许/禁止字段和提交合同。
- `cleaning_v3.yaml`：时间轴、缺失、异常和部分投运字段策略。
- `splits_v3.yaml`：walk-forward 时间折定义。
- `metrics_v3.yaml`：本地指标和聚合规则。
- `baseline_v3.yaml`：正确标签 baseline。
- `experiments/*.yaml`：一次只改变一个主要因素的实验配置。

当前暂不创建这些正式配置，因为数据指纹、标签测试和工况切分尚未完成。配置一旦用于正式 run，不原地重写，创建新版本。
