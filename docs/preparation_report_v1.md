# IronFlow 派生数据准备报告 v1

配置：`configs/cleaning_v1.yaml`

流水线：`src/prepare_dataset.py`

状态：**PASS**

## 结果摘要

| 项目 | 结果 |
|---|---:|
| 训练原始时间点 | 11,519 |
| 训练完整 15 分钟网格 | 11,521 |
| 测试时间点 | 192 |
| 内部派生表列数 | 55 |
| `input.csv` 列数 | 26（1 个时间键 + 25 个特征） |
| `input.csv` 缺失单元格 | 0 |
| `input.csv` 常量非时间列 | 0 |
| 非 `feat_` 新增字段 | 0 |
| 自动测试 | 7/7 通过 |

训练区补入 `2025-04-28 18:00`、`18:15` 两个时间点。所有补入点都有 `feat_inserted_timestamp` 标志。原始目标列未填补，训练两个目标仍各有 11,519 个真实标签；测试 `generator_1`、`generator_all` 分别保留 190、191 个当前观测。

## 提交输入视图

内部 `test_prepared.csv` 保留未填的原始目标和全部审计标志。提交输入候选 `input.csv` 与内部表分离：

- 排除原始 `generator_1`、`generator_all`，使用无缺失的 `feat_current_generator_1`、`feat_current_generator_all` 表示当前起点观测；
- 排除测试期全常量的 `air_heater_5`、`converter_user1`；
- 排除全零缺失标志、`feat_inserted_timestamp` 和仅供审计的边界标志；
- 保留两个目标当前观测的缺失标志，以区分真实观测与历史前向填充值；
- 所有新增字段均使用 `feat_` 前缀。

完整剔除名单、源文件 SHA-256、填充值数量和输出指纹位于本地 `outputs/prepared_v1/preparation_manifest.json`。

## 验证结果

- 15 分钟网格连续且时间戳唯一；
- 训练和测试内部派生表字段顺序相同；
- 已知缺失点只使用前一历史值填充，未使用 `bfill` 或未来插值；
- 原始目标缺失没有被写成合成标签；
- `input.csv` 完整覆盖 192 个测试起点；
- 契约校验、特征前缀校验和源文件指纹校验均通过。

该结果只证明数据准备流程符合 v1 契约，不代表官方数据质量得分，也不包含模型训练结果。
