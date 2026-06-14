[![CI](https://github.com/sun-yongji/taichi-mtp/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-mtp/actions/workflows/ci.yml)

# TaiChi-MTP 🔮 六爻深度调度的多 Token 预测引擎

> 华为云杯 2026 OPC 大赛 | 太极矩阵 M2 | CC-BY-SA-4.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-34/34-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-CC--BY--SA--4.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-taichi--mtp-blue)](https://pypi.org/project/taichi-mtp/)

## 核心创新

现有 MTP（如 DeepSeek-V3）使用固定预测深度。TaiChi-MTP 以 **六爻承乘比应关系** 构造自适应深度调度——归一化熵 < 0.90 浅爻单步预测，0.90~0.97 中爻逐步细化，> 0.97 深爻全六步推演。六个预测 head 通过 C6 循环群耦合矩阵连接：c_{ij} = |∑ω^{k(i-j)}|/6。

**关键现象**：湍流模式下 head 耦合差异达 100:1（标准 MTP 仅 10:1），意味着六爻调度能从高熵输入提取 10 倍于标准方法的结构信息。

## 性能

| 指标 | 数值 |
|------|------|
| Head 数 | 6（对应六爻） |
| 深度模式 | 浅(1) / 中(2-4) / 深(5-6) |
| 耦合差异（湍流） | 100:1 |
| 零方差保护阈值 | < 1e-8 |
| 测试通过率 | 34/34 |

## 安装

```bash
pip install taichi-mtp
```

## 快速开始

```python
from taichi_mtp import create_mtp_engine
import numpy as np

engine = create_mtp_engine(hidden_dim=512, output_dim=256, preset="balanced")
result = engine(np.random.randn(128, 512))
print(f"Depth: {result.depth_used}, Mode: {result.mode.value}")
```

## 太极矩阵体系

TaiChi-MTP 是太极矩阵六站体系的 M2 站：

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | [taichi-router](https://github.com/sun-yongji/taichi-router) | MoE 动态路由 |
| **M2** | **taichi-mtp** ← 你在这里 | 多 token 预测 |
| M3 | [taichi-quant](https://github.com/sun-yongji/taichi-quant) | 熵量化 |
| M4 | [taichi-hex](https://github.com/sun-yongji/taichi-hex) | 六边形注意力 |
| M5 | [taichi-correct](https://github.com/sun-yongji/taichi-correct) | 共识校正 |
| M6 | [taichi-matrix](https://github.com/sun-yongji/taichi-matrix) | 统一入口 |

技术白皮书：[太极矩阵技术白皮书(中文)](https://docs.qq.com/aio/DTldDRGpIbGdseG1H) | [WHITEPAPER.md](https://github.com/sun-yongji/taichi-matrix/blob/master/WHITEPAPER.md)

## 参与贡献

欢迎提交 Issue 和 Pull Request。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

CC-BY-SA-4.0 · 易宇本源研究中心 · 2026
