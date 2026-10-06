# 外部约 61 分候选来源鉴定与复现合同

> **2026-10-01 来源更新：** 最新交接包已经证明冻结 CSV 来自 v28，而非本合同最初排查的 v15/v16。此前最小复现失败仍是有效的排除证据，但 `updated_source_missing` 阻塞已解除。v28 的独立全量重建尚未执行；新的源码、哈希与门禁见 `docs/15_LATEST_OPTIMAL_BUNDLE_REVIEW.md`。本合同后续内容作为历史复现记录保留。

## 1. 复现目标

目标不是先改进模型，而是判断当前外部约 61 分预测包能否由 2026-09-30 学弟代码包稳定生成。

第一轮复现的唯一通过条件是：重新生成的 `s_result.csv` 与 `l_result.csv` 分别匹配冻结参考文件的 SHA-256。任何标签、采样、特征或后处理修正都必须等复现通过后另立实验。

冻结参考：

| 产物 | SHA-256 |
|---|---|
| 提交 ZIP | `78b0e4eb024e5cc1d058bb8a77a9ac506246304105bbdb070cedaaa91071b12f` |
| `s_result.csv` | `e96e633801c2d577006670ea178904e253f5e1da75faaa000a0e264eaf3b4a4f` |
| `l_result.csv` | `3c6e55d4d92b8ecee65609abb6784a9fe15178afc2f1f7cf04966372bf28de28` |

## 2. 版本归因结论

最初静态证据提示该包可能是手册中的 v16：手册将 v16 定义为“树模型 9 月重训 + v15 配方”，新代码也只为基线增加了 `--train-start`。但最小复现已经否定这一归因。

在相同数据、环境、参数和后处理下，分别重建 v15 全期树与 v16 九月树的五个代表列：

| 来源假设 | g1 step1 MAE | g1 step96 MAE | gall step1 MAE | gall step33 MAE | gall step96 MAE |
|---|---:|---:|---:|---:|---:|
| v15 全期树 + 九月燃料层 | 1.2707 | 3.0047 | 7.2534 | 9.2207 | 8.6214 |
| v16 九月树 + 九月燃料层 | 0.5551 | 2.3899 | 8.2967 | 10.6686 | 9.8311 |

每个代表列的 960 行全部与参考包不一致，差异是 MW 级，不是浮点或换行误差。继续扫描 7 月、8 月、8 月中、9 月及 9 月内多个训练起点仍未命中，全期树与九月树线性融合也未命中。

当前可确认：

- g1 九月燃料系数为 `[0.000134367550138083, 0.0011554463562695452, 0.00019412344433509634, -7.007563744598015]`；
- gall 九月燃料系数为 `[0.00036850873765689973, 0.0006810445094277805, -3.2028011474733042]`；
- 参考包确实属于九月燃料后处理家族；
- 底层模型、训练参数或额外后处理不在当前交接包中。

因此现阶段来源状态为 `updated_source_missing`，不再标记为 v16 高可信推断。

## 3. 环境合同

学弟 `requirements.txt` 与当前 `vocs` 环境完全匹配：

| 组件 | 要求 | 当前检查值 |
|---|---:|---:|
| Python | 3.10 | 3.10.20 |
| LightGBM | 4.7.0 | 4.7.0 |
| NumPy | 2.1.3 | 2.1.3 |
| pandas | 2.2.3 | 2.2.3 |
| scikit-learn | 1.5.2 | 1.5.2 |
| PyYAML | ≥6.0 | 6.0.3 |

因此本轮不安装任何包、不新建环境，直接使用 `D:\anaconda\envs\vocs\python.exe`。

预检发现该环境同时残留 `numpy-2.0.1.dist-info` 和 `numpy-2.1.3.dist-info`，导致 `pip show numpy` 错报 2.0.1；实际导入的 `numpy.__version__` 为 2.1.3，模块也来自当前 `vocs`。复现工具以实际导入版本为执行合同，同时把重复元数据写入 warning。本轮不擅自卸载或修复环境；若结果不能复现，再把清理残留元数据作为独立环境问题处理。

## 4. 数据合同

复现只允许使用已经审计的官方原始文件。关键 SHA-256：

| 文件 | SHA-256 |
|---|---|
| train `Semi_gas.csv` | `b54d7b4e9f4ec41d2b74822b7fabd87e8f7bb0fedfe09f7a8b5374379b20b58c` |
| train `Semi_gas_holder.csv` | `241261644742a63f3587b0fd02441989e71ef43edcfdafaabcd35a1ac68569fa` |
| train `Semi_gas_user.csv` | `6810a0edf71d16a3b01121167d16adedcee0afd81147f44d8384c82161b2288d` |
| train `Semi_load.csv` | `0d8c14e3abffe91137c18abde00ec19a62a42e80683f7ebda15da8bd955dc309` |
| test `Semi_test_gas.csv` | `60d1d23b43ebf71c9a6161f8030699ce831908c1dcd29c64d8d19f1fa7121ef2` |
| test `Semi_test_gas_holder.csv` | `bb06cd2e509645c4e1e22b04df17d8e187a6349c73dcaf4f93c0c91018a8f989` |
| test `Semi_test_gas_user.csv` | `18a5783cf62f385e44988c59536a5a18a3f4a6437cf544416179b71fd553c48c` |
| test `Semi_test_load.csv` | `ca06847286fcbafe7c06f8617638895404ee266810735c8850c615af850b7bf7` |

隔离工作区使用硬链接或复制品，不修改 `data/raw/`。

## 5. 已取得交接代码的计算链

### 5.1 旧版数据准备

- 四表按 `datetime` 外连接。
- 通过 `reindex(15min)` 取 00/15/30/45 分的瞬时记录；这不是 15 分钟区间均值。
- 协变量因果前向填充，最多 4 个 15 分钟步长。
- 删除四个在旧训练认知中全空的字段。
- 预期产物：train 14,496×55，test 960×55，input 960×25。

这里与当前 v3 正确区间标签协议不同，但第一轮复现不得修改。

### 5.2 九月树模型基线

- 调用 `build_submission_semi.py --train-start 2025-09-01`。
- 只保留 2025-09-01 00:00 至 2025-09-30 23:45 的训练起点。
- 每个目标、每个 horizon 独立训练，共 2×96=192 个 LightGBM。
- 参数：L1、300 棵树、学习率 0.03、31 叶、`min_child_samples=50`、seed 42、deterministic、force_col_wise。
- 特征不含目标历史；使用起点原始协变量、日历、煤气平衡 rolling、气柜变化和发电耗气变化。
- 标签是 15 分钟点值表上的 `shift(-step)`，并非区间均值。
- 输出保留 6 位小数；短期等于长期前 8 步。

### 5.3 v15 九月燃料后处理

`generator_1`：

```text
fuel_g1(t) = September OLS(three generator gas-use columns -> generator_1)
final_g1(t,h) = 0.20 * SeptemberTree_g1(t,h) + 0.80 * fuel_g1(t)
```

同一个起点燃料水平被广播到全部 96 个 horizon。

`generator_all`：

- step 1–32：保持九月树模型预测。
- step 33–96：

```text
fuel_gall(t) = September OLS(BFG use, converter-gas use -> generator_all)
clim(t) = full-training mean generator_all grouped by origin weekday and origin hour
final_gall(t,h) = 0.60 * (0.75 * SeptemberTree_gall(t,h) + 0.25 * clim(t))
                  + 0.40 * fuel_gall(t)
```

注意 `clim(t)` 使用起点星期和小时，不是目标时刻 `t+h`。第一轮复现必须保留这一实现。

## 6. 已发现的复现风险

1. 交接包没有独立 `v16.py`；v16 是两个命令的组合，容易因 base-dir 指错而复现成 v15。
2. v15 脚本写出的 manifest 固定声称 base 为 v14，即使传入 v16 树预测，manifest 也会误报来源。
3. 交接包的 `official_submissions.csv` 只登记到 v7，v9–v16 的平台证据主要来自手册，台账不完整。
4. 模型不落盘，复现需要重新训练全部 192 个模型。
5. LightGBM 虽已设 deterministic，但线程调度、CPU 或库变化仍可能产生微小差异；当前环境版本已完全对齐。
6. ZIP 成员时间戳会改变 ZIP 容器哈希；CSV 成员哈希才是第一复现判据。
7. 当前 61 分只有近似总分，没有分目标明细，不能以平台分反推哪个组件改善。
8. 测试协变量在推理时只能使用起点及以前行；旧树模型确实只取起点特征，`ramp_lag_diagnosis.py` 读取未来测试协变量只能作为诊断，绝不能进入提交模型。

## 7. 隔离工作区与已完成预检

隔离工作区已经建立。该步骤校验了源码、环境和数据哈希，并以硬链接接入八个原始文件：

```powershell
cd D:\daima\aic\shiyan\fusai
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.prepare_external_v16_reproduction --create
```

旧版数据准备也已执行并通过：train 14,496×55、test 960×55、input 960×25。

以下是曾计划用于全量 v16 验证的命令，但代表 horizon 已经证明该来源不匹配，**现在不要运行这组 192 模型任务**：

```powershell
& D:\anaconda\envs\vocs\python.exe src\prepare_dataset.py --config configs\cleaning_v2.yaml

& D:\anaconda\envs\vocs\python.exe src\build_submission_semi.py `
  --mode build `
  --train-start 2025-09-01 `
  --submission-id reproduce_v16_sep_tree_base

& D:\anaconda\envs\vocs\python.exe src\build_submission_semi_v15_sepwindow.py `
  --base-dir outputs/submissions/reproduce_v16_sep_tree_base `
  --out-id reproduce_v16_final
```

若以后取得证据确认新版本确实基于该入口，再回到 `fusai/` 使用以下工具逐字节验证：

```powershell
& D:\anaconda\envs\vocs\python.exe -m src.round2_v3.verify_external_v16_reproduction `
  --candidate-dir outputs/reproduction_workspaces/external_v16_61_candidate/outputs/submissions/reproduce_v16_final
```

全量训练预计为 192 个 LightGBM，按交接手册约 5–10 分钟。当前因来源假设被否定而停止，不消耗这次 CPU 任务。

## 8. 通过与失败判据

通过：

- 数据准备的形状与手册一致；
- 两个候选 CSV 的列、时间和数值合同通过；
- 两个 CSV 的 SHA-256 与冻结参考完全一致。

当前失败已经定位为真实数值差，而不是环境、数据哈希、换行或序列化问题。恢复复现至少需要以下任一项：

1. 61 分版本实际运行的完整命令；
2. 相对 9 月 30 日交接包的代码 diff；
3. 生成该 ZIP 的 base-dir 及其中 `submission_manifest.json`；
4. 当次完整输出目录或保存的模型；
5. 若做过人工融合，提供参与融合的版本、权重及列范围。

在更新来源取得前，不开展声称“从 61 分版继续改进”的实验；可以继续独立研究，但必须明确控制组来源尚未复现。
