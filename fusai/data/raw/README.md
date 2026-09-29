# 官方原始数据，只读

本目录已放置复赛官方训练、测试和原始 ZIP 的本地副本：

- `packages/`：原始训练包与测试包。
- `train/`：4 个训练 CSV。
- `test/`：4 个测试 CSV。

来源与副本 SHA-256 已逐文件核验一致，详情见 `references/SOURCE_INVENTORY.md`。

禁止：

- 直接修改 CSV/XLSX。
- 在这里生成清洗文件、缓存或模型输入。
- 覆盖同名文件。
- 将原始大文件提交到 Git。

纳入数据后必须生成 `manifests/round2_v3/data_fingerprints.json`，记录文件大小、SHA-256、schema、时间范围和来源 ZIP。
