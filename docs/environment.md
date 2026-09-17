# 开发环境记录

当前本地环境：`D:\anaconda\envs\vocs`。

已验证的核心版本：

```text
Python       3.10.20
NumPy        2.1.3
Pandas       2.2.3
SciPy        1.15.3
scikit-learn 1.5.2
PyTorch      2.5.1
CUDA runtime 11.8
LightGBM     4.7.0
XGBoost      3.2.0
openpyxl     3.1.5
```

GPU 已验证可用：NVIDIA GeForce RTX 4060 Laptop GPU，显存约 8 GB。

依赖一致性检查已通过：`No broken requirements found.`

## 说明

- 当前环境足够进行数据审计、树模型 Baseline 和 PyTorch LSTM/GRU 开发。
- TensorFlow/Keras 尚未安装。只有旧代码确实依赖它们时才考虑补充。
- OR-Tools 等优化求解器等进入调度建模阶段后再安装。
- 当前记录用于本地开发，不代表赛事官方 Linux 运行环境。正式提交前需要重新验证离线运行。
