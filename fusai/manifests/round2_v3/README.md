# v3 Manifest 区

本目录保存小型、可提交 Git、用于证明数据与协议身份的文件：

- 原始数据指纹。
- prepared 数据指纹。
- 逐起点 split assignment。
- split SHA-256。
- 字段可用性合同。
- 协议版本与配置指纹。

当前已生成：

- `prepared_labels_v3.json`：15 分钟区间标签的来源、公式、完整性与哈希。
- `validation_origins_v3.csv`：冻结后的逐验证起点清单。
- `validation_origins_v3.sha256`：逐起点清单指纹。
- `splits_v3.json`：各折边界、角色和完整标签上界。

Manifest 不保存大规模预测或模型文件，也不得包含私密信息和外部数据。
