# 给学长 —— 复赛最优方案整合包（可逐位复现）

> 整理日期：2026-10-01。队名"这模型调的有力量"（AIC-2026-84933645）。
> **本包承诺：三个提交包中的每一个，都能用本包内的代码 + 官方原始数据，跑出与提交内容逐字节一致的结果。** 全部经过"重跑+逐位比对"验证。

---

## 一、三个提交包是什么

| 包 | 状态 | 官方分/预期 | 内容一句话 |
|---|---|---|---|
| `01_已验证最高分_v28_官方61.58` | **已提交，官方 61.5824（当前最高）** | 61.58 | 恢复列引擎 + 无门控 + g1 分步长燃料权重（0.8 近/0.6 远） |
| `02_修正版_v29b_预期63.45` | 已构建未提交（明天第 1 发） | **预期 ≈63.45** | 组件最优拼装：short=g1 v16_final（89.47）+gall v20（93.74）；long=gall v20 门控版（87.54）+g1 v28（82.96）。**四项组件全部官方验证** |
| `03_三折最优候选_v34` | 已构建未提交（明天第 1 发） | 三折全胜（+2.87/+0.99/+0.13） | v29 的 long g1 再升级：权重细扫至近 0.6/远 0.1 |

- 每个包内含：`这模型调的有力量_gas_predict_semi.zip`（上传平台的就是它）+ 解包后的 `s_result.csv`/`l_result.csv` + `submission_manifest.json`（配方与指纹）。
- 提交格式（官方群答疑确认）：复赛只交 `s_result.csv`（短期 17 列）+ `l_result.csv`（长期 193 列），不要 input.csv。
- 排行榜取最高分；每天 5 次提交。

## 二、复现步骤（以 v28 为例，最简单；v29/v34 同理）

1. 数据就位（官方原始数据，本包不含）：
   - `复赛-参赛者使用.zip` → `Semi_gas.csv / Semi_gas_holder.csv / Semi_gas_user.csv / Semi_load.csv` 放 `<repo>/data/`
   - `复赛-评分所用测试集.zip` → `Semi_test_*.csv` 4 个文件放 `<repo>/test/`
2. 环境：`pip install -r requirements.txt`（Python 3.10）
3. 运行：

```powershell
# v28（自包含：原始 1 分钟数据 → 清洗 → 特征 → 192 个 LightGBM → 打包）
python src/build_submission_semi_v28.py
# 输出 outputs/submissions/20260930T_nogate_v28/{s_result.csv, l_result.csv}
# 与本包 01 文件夹里的 CSV 逐字节一致（已验证）

# v29b（组件拼装：依赖 v2 引擎输出链 + v28 输出，链路见下）
python src/build_submission_semi.py --mode build          # v2 引擎（端到端重训已验证）
python src/build_submission_semi_v15_sepwindow.py         # v15 配方（在 v2 输出上）
python src/build_submission_semi.py --mode build --train-start 2025-09-01 --submission-id 20260930T_septrees_v16
python src/build_submission_semi_v15_sepwindow.py --base-dir outputs/submissions/20260930T_septrees_v16 --out-id 20260930T_v16_final
python src/build_submission_semi.py --mode build          # v20 底层（含 v20 配方链，见 manifest）
python src/build_submission_semi_v28.py                   # v28 引擎（自包含，重训已验证）
python src/build_submission_semi_v29b.py                  # 拼装 v29b
# 期望输出与 02 文件夹里的 CSV 逐字节一致

# v34（同 v28 引擎 + 权重细扫参数）
python src/build_submission_semi_v34.py
# 期望输出与 03 文件夹里的 CSV 逐字节一致

# 提交包内 CSV 校验（逐位比对）
python -c "import pandas as pd; print(pd.read_csv('01_已验证最高分_v28_官方61.58/s_result.csv').equals(pd.read_csv('outputs/submissions/20260930T_nogate_v28/s_result.csv')))"
```

4. **确定性保证**：LightGBM `deterministic=true`、`seed 42`、`force_col_wise=true`；管线无随机成分；requirements 锁版本后同环境逐位一致。已完成的验证：v2 引擎端到端重训 IDENTICAL、v28 重跑 IDENTICAL、v29 脚本输出 IDENTICAL、v34 重跑 IDENTICAL、v9/v11/v12/v21/v23 IDENTICAL（详见 `复现验证总表.md`）。

## 三、方法一句话（详细见各 manifest 与复现验证总表）

- 预测口径：960 个 15 分钟起点（10-01~10-10），每目标 × 96 步（长）+ 8 步（短），LightGBM 直接多步（L1、300 树、冻结参数）。
- 特征：只用起点 t 及之前的信息（官方因果红线）；煤气产耗 + 气柜 + 发电耗气 + 日历 + 煤气平衡物理特征；**恢复列**（blast_furnace_3=3号高炉，10 月满负荷，初赛审计误标全空）。
- 两个 10 月结构利器：① 燃料耗气线性反推与树模型按步长加权混合（物理模型抗工况漂移）；② regime 门控（官方验证净正，v28 的无门控版已回滚）。
- 已证伪路线（勿重试）：两阶段路径预测、递推自反馈、预测平滑、14400 起点网格、gall 燃料权重 >0.5——每条死因都有官方分数背书。

## 四、文件清单

```
给学长_最优版整合/
├── README.md                          ← 本文件
├── requirements.txt
├── 01_已验证最高分_v28_官方61.58/      ← 已提交，官方 61.5824
├── 02_修正版_v29b_预期63.45/           ← 待提交（明天第 1 发）
├── 03_三折最优候选_v34/                ← 待提交（明天第 1 发）
├── src/                               ← 全部构建脚本（17 个）
└── configs/                           ← 提交契约与清洗配置
```

## 五、注意事项

1. 上传前核对 zip 大小/SHA256（manifest 里有），防同名包传错；
2. 出分有延迟（几分钟到 1 小时+）；"未提交"状态先查参赛作品名称；
3. **最终需交代码**：官方要求 `requirements.txt` + `python -u xxx.py` + 入口 `predict from models.py`——本包 src/ 已是完整可复现代码，届时按官方模板包一层即可；
4. 数据与提交包不得上传公开渠道。
5. **组装类包的纪律（v29 事故教训）**：v29 的 short 文件曾因组装代码只替换了 long 文件的 g1 列而带病上传（short g1 拿的是 v20 的 85.56 而非 89.47，官方 59.72 暴露）。修复版 v29b 已四项来源逐位校验。**纪律：组合包的每一列必须与最终来源的官方验证包逐位比对，禁止与中间组装件互比。**
