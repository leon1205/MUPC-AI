# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MUPC 强化学习模型训练管线，在 x86 PC 上训练 RL 模型，导出 ONNX 交付给 MUPC AI 引擎（RK3588 NPU 部署）。

**核心文档**（本项目的规格定义）：
- PRD：`docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-PRD.md` v2.0
- 设计文档：`docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-设计文档.md` v2.0
- 部署端规格：`docs/MUPC/05-MUPC-AI引擎-PRD.md` v2.2

## 常用命令

```bash
# 快速训练测试（5万步，约1分钟）
python train.py --mode MODE-01 --total-timesteps 50000 --no-lstm

# 完整训练（多模式单模型，一个模型覆盖全部5种场景）
python train.py --mode all --total-timesteps 175200

# 5个独立模型训练（推荐生产方案）
python train_single_modes.py

# LSTM 模型训练
python train_lstm.py

# 环境自测
python mupc_env.py

# 数据加载器自测
python data_loader.py

# 导出 ONNX
python export_onnx.py --checkpoint checkpoints/MODE-01_model.zip
python export_onnx.py --lstm checkpoints/lstm_model.pt

# 下载数据集
python data/download_smart_ds.py
```

## 架构

```
SMART-DS CSV/Parquet
        ↓
data_loader.py       → 加载光伏/负荷/气象，合成 TOU电价/需量/调度
        ↓
lstm_model.py        → LSTM 预测模型（或 Oracle 后备）
        ↓
mupc_env.py          → Gymnasium 环境：48维观测 + 4维动作 + 5场景奖励
        ↓
train.py             → PPO/SAC 训练（或 NumPy PPO 后备）
        ↓
export_onnx.py       → ONNX 导出
        ↓
MUPC AI Engine (Rust, RK3588 NPU)
```

## 核心规格

**观测空间（48维单模式，49维多模式）**：
```
[0..9]   D1 实时数据: SOC/光伏/负荷/电网功率/变压器负载/电池功率/三相电压
[9..24]  D2 光伏预测 (15维，LSTM或Oracle)
[24..39] D2 负荷预测 (15维，LSTM或Oracle)
[39..42] D3 电价: 当前价/下时段价/时段ID
[42..45] D4 需量: 当前需量/合同值/本月峰值
[45..47] D5 气象: 辐照/温度
[47]     D6 调度指令
[48]     mode_id (仅多模式训练)
```

**动作空间（4维）**：`[p_batt, q_batt, load_shedding, pv_limit]`
- p_batt ∈ [-500, 500] kW（充电<0，放电>0）
- q_batt ∈ [-300, 300] kVar
- load_shedding ∈ [0, 500] kW
- pv_limit ∈ [0, 1]

**5种场景模式**：
| 模式 | 优化目标 |
|------|----------|
| MODE-01 | 农网灌溉：光伏消纳 + 防过载 |
| MODE-02 | 自主套利：峰谷价差 + 电池保护 |
| MODE-03 | 需量控制：减免需量罚金 |
| MODE-04 | 虚拟电厂：辅助服务 + 响应精度 |
| MODE-05 | 极致绿色：绿电消纳 + 碳减排 |

**关键物理约束**：
- SOC 硬限制：10%~90%（不可突破）
- 变压器容量：500 kVA，过载阈值 85%
- 动作约束 ACT-01~05（见 action_validator.py）

## 模块依赖

```
train.py
├── data_loader.py          (SMART-DS + 状态合成)
├── lstm_model.py           (PyTorch LSTM / Oracle 后备)
├── mupc_env.py             (Gymnasium 环境)
│   └── action_validator.py  (5条动作约束)
├── stable_baselines3       (PPO/SAC，可选)
│   └── _ppo_core.py       (NumPy PPO 后备)
└── export_onnx.py         (ONNX 导出)
```

**降级规则**：
- SB3 不可用 → `_ppo_core.py`（纯 NumPy PPO）
- Gymnasium 不可用 → `_gym_stub.py`
- LSTM 未提供 → Oracle（真实值 + 噪声）

## 环境参数

| 参数 | 值 |
|------|-----|
| 变压器容量 | 500 kVA |
| 电池容量 | 200 kWh |
| 最大充放电功率 | 500 kW |
| SOC 硬限制 | 10%~90% |
| 过载阈值 | 85% |
| 时间步长 | 15分钟 |

## 代码规范

1. 安全相关代码标注 `SAFETY`
2. 动作 clamp 标注约束规则 ID (ACT-01~05)
3. 所有函数包含 Type Hints 和 Docstrings
4. `mupc_env.py` 不依赖 RL 框架，可独立运行
