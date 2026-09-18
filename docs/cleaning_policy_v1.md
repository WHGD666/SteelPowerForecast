# IronFlow 清洗策略 v1

状态：**冻结**。该策略只生成派生数据，不覆盖 `data/` 或 `test/` 原始文件。若规则发生变化，必须创建新版本并说明旧实验为何不可比较。

## 1. 时间对齐

- 每个分区独立建立 15 分钟完整时间网格。
- 训练区从 `2025-01-01 00:00` 到 `2025-05-01 00:00`，完整网格应为 11,521 行。
- 测试区从 `2025-05-01 00:00` 到 `2025-05-02 23:45`，完整网格为 192 行。
- 新插入的时间点使用 `feat_inserted_timestamp=1` 标记。
- `2025-05-01 00:00` 使用 `feat_boundary_context_overlap=1` 标记，只作为滚动起点上下文，不作为独立泛化证据。

## 2. 全空列

以下列在训练和测试均全空，从第一版派生数据中排除，但原始文件和审计证据保留：

- `blast_furnace_3`
- `air_heater_3`
- `blast_furnace_gas_holder_1`
- `converter_user3`

测试期常量 0 的 `air_heater_5`、`converter_user1` 不删除。它们可能表示停机或工况切换。

## 3. 缺失处理

- 所有发生过缺失的字段都创建 `feat_missing_<字段名>` 指示列。
- 非目标协变量只使用因果前向填充，最多连续 4 个 15 分钟步长；禁止 `bfill`，禁止利用未来点线性插值。
- `generator_1`、`generator_all` 原始列保持不变，缺失值不填，避免生成虚假训练标签。
- 另建 `feat_current_generator_1`、`feat_current_generator_all`，仅用于表达滚动起点的当前发电量；缺失时使用最多 4 步的因果前向填充。
- 目标原值缺失的行不得作为相应预测目标的监督标签，但仍可作为时间窗口上下文。

## 4. 异常值

- 本阶段不删除、不缩尾、不裁剪任何统计异常候选。
- IQR、MAD 或其他阈值必须在每个训练折内部拟合，再应用到对应验证段。
- 零流量、停机、炉况切换和机组出力区间变化优先视为业务状态，除非有明确采集错误证据。

## 5. 重复与冲突

- 重复时间戳或重复整行视为硬错误，流水线立即停止。
- 四表同一时间字段冲突时不得静默覆盖。
- 训练/测试边界重复保留在各自分区，通过边界标志解释。

## 6. 输出与审计

执行命令：

```powershell
D:\anaconda\envs\vocs\python.exe src/prepare_dataset.py --root . --config configs/cleaning_v1.yaml
```

输出保存在 `outputs/prepared_v1/`：

- `train_prepared.csv`
- `test_prepared.csv`
- `input.csv`：提交输入候选；排除未填的原始目标列、测试期全常量列和仅供审计的边界标志，保留无缺失的 `feat_current_generator_*` 当前观测特征。被排除列记录在清单中
- `preparation_manifest.json`

清洗清单记录源文件 SHA-256、插入时间点、填充值数量、剩余缺失、目标标签可用行数和输出文件指纹。
