# 本地运行产物

正式运行按不可变 `run_id` 创建目录：

```text
outputs/<run_id>/
├── run_manifest.yaml
├── resolved_config.yaml
├── metrics.json
├── metrics_by_horizon.csv
├── predictions/
├── models/
└── reports/
```

目录存在时运行应拒绝覆盖。大规模产物默认不提交 Git，重要身份通过哈希和 manifest 登记。
