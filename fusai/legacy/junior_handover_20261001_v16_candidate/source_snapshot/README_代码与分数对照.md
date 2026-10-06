# 复赛代码包说明 —— 每个脚本对应哪一版分数

> 配套文档：`复赛交底书_给学长.md`（完整交底：规则破解、数据事实、经验教训、冲 60 路径）
> 环境：Python 3.10 + `pip install lightgbm==4.7.0 numpy==2.1.3 pandas==2.2.3 PyYAML`（见 requirements.txt）
> 运行目录：把本包解到任意目录，`data/`、`test/`、`outputs/prepared_v2/` 需按仓库结构放置（数据不出网，仅本地）

## 一、脚本 → 版本 → 官方分数 对照表

| 脚本（src/） | 作用 | 对应提交 | 官方分 |
|---|---|---|---|
| `build_submission_semi.py --mode build` | **现行主力**：96+8 步双结果文件构建（协变量 LightGBM 直接多步，192 个模型） | v2 | **48.9103** |
| 同脚本 + g1 燃料混合（见 `--help` 与交底书 §5） | g1 = 50%树 + 50%燃料耗气线性反推 | **v9** | **52.4085（最高）** |
| `build_submission_semi_v10_twostage.py` | 两阶段：阶段一预测 12 条煤气聚合指标的未来 24h 路径（1,152 个模型），阶段二发电量模型增加"目标时刻特征块" | v10 | **未提交，待审核**（留出集证据：g1 +2.2 点） |
| `build_submission_semi_min1.py` | 真·1 分钟起点版本（14400 起点） | v4/v5 | 47.6312（与 960 行同分 → 官方按 15 分钟网格评分） |
| `build_submission_semi_min1_broadcast.py` | 960→14400 起点广播（网格假设检验） | v3 | —（中间实验） |
| `recursive_semi_v8.py --mode dev/build` | 递推式模型（自预测滚动补特征） | v8 | 39.5631（**负结果**：近距离误差自我放大，弃） |
| `holdout_calibration_semi.py` | 封存留出集（9/28-29）校准：点值 vs 区间均值口径、收缩系数扫描 | 诊断 | 发现官方口径=15分钟区间均值（g1 点值 MAPE 0.60 vs 区间均值 0.0996） |
| `holdout_perstep_semi.py` | 逐 step 精度曲线诊断 + 平滑/收缩/气候基线候选评估 | 诊断 | 震荡 47/95、40/95 → 触发 v7 |
| `tau_block_check.py` | **v10 的设计验证**：目标时刻(tau)协变量特征是否有用（真实路径上界测试） | 诊断 | g1 acc 0.9004→0.9225（+2.2 点）→ 才有 v10 |
| `validate_contract.py` | 提交门禁：列名/行数/缺失/常量列/时间戳检查 | 每版都跑 | — |
| `prepare_dataset.py` + `stage_config.py` | 1 分钟→15 分钟因果清洗（配置驱动，初赛/复赛共用） | 底层 | — |
| `run_baseline.py` / `run_experiment_v2.py` | 学长原仓库的基线与 v2 特征工程（被 semi 构建器复用） | 底层 | — |

## 二、流程（v9 现行最优的完整复现）

```powershell
$py = "E:\anaconda\envs\vocs\python.exe"
cd D:\zuoye\sai\SteelPowerForecast          # 完整仓库（含 data/test，本包不含大数据）

# 1. 数据准备（1 分钟 → 15 分钟预测网格）
& $py src/prepare_dataset.py --config configs/cleaning_v2.yaml

# 2. 构建双文件提交包（s_result 短期 17 列 + l_result 长期 193 列）
& $py src/build_submission_semi.py --mode build
#    输出: outputs/submissions/<时间戳>/  内含 zip + manifest

# 3. 提交门禁
& $py src/validate_contract.py --input <提交目录>\input.csv --result <提交目录>\s_result.csv
```

v9 的 g1 燃料混合构建与 v10 两阶段构建分别见各自脚本 `main()` 内注释（思路：物理线性模型与树模型误差不同源，等权互补；v10 在此之上引入阶段一预测的协变量路径）。

## 三、关键结论（详细版在交底书）

1. 官方按 15 分钟网格（960 起点）评分，用 15 分钟区间均值做实际值；
2. 评分是凸映射：短期 1 点准确率≈1.5 分、长期≈3 分；90%/85% 及格线正好对应 ≈60 分；
3. **g1（小机组）是瓶颈**：10 月工况漂移下树模型掉分，燃料物理反推更稳，等权混合 +3.5 分（v9）；
4. 测试期目标列全空 → 一切"目标滞后/当前值"特征在复赛不可用；
5. 负结果也 valuable：递推近距离崩（v8）、9 月调参不迁移（v7）、gall 燃料混合无收益——全部留档在 validation_outputs/。

## 四、文件清单

```
复赛代码_给学长/
├── README_代码与分数对照.md   ← 本文件
├── 复赛交底书_给学长.md        ← 完整交底（先看这个）
├── requirements.txt
├── src/                        ← 16 个脚本（上表）
├── configs/                    ← 复赛 v2 配置（清洗/切分/协议/提交）
├── docs/                       ← 任务合同 v2、复赛适配计划、交接文档
├── manifests/                  ← 切分指纹 + 官方提交记录（全部分数留档）
└── validation_outputs/         ← 留出集/诊断的机器可读证据（JSON）
```

*注：`data/`、`test/`、`outputs/prepared_v2/` 为本地大数据与派生数据，未包含在本包（原始数据你这里有）。所有训练与提交均在本机完成，未向任何远端推送。*
