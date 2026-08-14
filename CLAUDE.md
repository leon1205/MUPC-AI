# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MUPC 强化学习模型训练管线，在 x86 PC 上训练 RL 模型，导出 ONNX 交付给 MUPC AI 引擎（RK3588 NPU 部署）。

**核心文档**（本项目的规格定义）：

- PRD：`docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-PRD.md` v2.0
- 设计文档：`docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-设计文档.md` v2.0

下游项目文档（MUPC AI引擎）

- 部署端规格：`docs/MUPC/05-MUPC-AI引擎-PRD.md` v2.6
- docs/MUPC/ 目录保存的是下游项目中文档，请不要随意改动！如果需要改动一定要经过确认同意！！！

## 常用命令

```bash
# 快速训练测试（5万步，约1分钟）
python train.py --mode MODE-01 --total-timesteps 50000 --no-lstm

# 完整训练（多模式单模型，一个模型覆盖全部5种场景）
python train.py --mode all --total-timesteps 175200

# 5个独立模型训练（推荐生产方案）
python train.py --mode single --total-timesteps 200000

# LSTM训练 + RL训练（合并数据，1M步）
python train.py --mode MODE-01 --data-source merged --train-lstm \
       --lstm-params hidden_dim=128,num_layers=3,epochs=200,patience=30 \
       --total-timesteps 1000000 --export-onnx

# 独立训练 LSTM（仅LSTM，不跑RL）
python train.py --train-lstm --data-source merged --lstm-params epochs=100

# 中国数据 + 低学习率衰减
python train.py --mode all --data-source china --lr-decay --total-timesteps 500000

# 生成中国合成数据（需指定经纬度）
python data_loader.py --generate --lat 31.23 --lon 121.47 --year 2023

# 统一数据加载（自动检测数据源，支持中国合成/SMART-DS）
python data_loader.py --unified --lat 31.23 --lon 121.47

# 环境自测
python mupc_env.py

# 数据加载器自测
python data_loader.py

# 导出 ONNX
python export_onnx.py --checkpoint checkpoints/MODE-01_model.zip
python export_onnx.py --lstm checkpoints/lstm_checkpoint.pt

# 下载 SMART-DS 数据集
python data/download_smart_ds.py

# 运行测试
python tests/test_v31_features.py        # v3.1 核心功能（QuantileLoss/Log-barrier/ErrorCorrection 等）
python tests/test_grid2op_replacement.py # Grid2Op 替换/降级
python tests/test_modes.py               # 多模式训练结果验证（需 SB3，独立脚本）
python -m pytest tests/ -q               # 等价运行可被 pytest 收集的单元测试
```

## 架构

```
SMART-DS CSV/Parquet
        ↓
data_loader.py          → 加载光伏/负荷/气象，合成 TOU电价/需量/调度
        ↓
models/                 → ML 模型 (v3.1)
  ├── lstm.py           → LSTM + TCN + Attention + BiLSTM 预测
  ├── error_correction.py → BiLSTM 误差修正
  └── vmd.py            → VMD 变分模态分解
        ↓
mupc_env/               → Gymnasium 环境：78/79维观测 + 2维动作 + 5场景奖励
  ├── core.py           → MupcEnv 主类
  ├── observation.py    → EnvState + build_observation + normalize_obs
  ├── rewards.py        → 5 场景奖励 + SCENE-01 子奖励
  ├── action_validator.py → 动作约束 ACT-01~05
  ├── state_builder.py  → 状态/奖励构建纯函数 (A-4 从 core.py 拆出)
  └── grid2op/          → Grid2Op + Pandapower 电压仿真
        ↓
train.py                → SB3 PPO/SAC 训练（主路径），NumPy PPO 后备
        ↓
export_onnx.py          → ONNX 导出
        ↓
MUPC AI Engine (Rust, RK3588 NPU)
```

## 核心规格

**观测空间（78维单模式，79维多模式，v2.14 对齐下游 AI 引擎）**：

```
[0..8]   D1 实时数据: SOC/光伏/负荷/电网功率/变压器负载/电池功率/三相电压 (9维)
[9..23]  D2 光伏预测 (15维，LSTM或Oracle)
[24..38] D2 负荷预测 (15维，LSTM或Oracle)
[39..41] D3 电价: 当前价/下时段价/时段ID (3维, peak/valley_price 仅日志使用)
[42..44] D4 需量: 当前需量/合同值/本月峰值
[45..46] D5 气象: 辐照/温度
[47]     D6 调度有功指令 (1维, dispatch_q_set 仅日志使用)
[48]     D7 实时模块Q裕度
[49..54] D8 季节one-hot: 灌溉季/炒茶季/空调季/常规季/保留/保留 (6维)
[55..56] D8 时段: 白天/夜间 (2维)
[57..60] D9 安全覆盖: active/p_ref/consecutive/ratio (4维)
[61..75] D10 分位数负荷预测: P3.3~P96.7 (15维)
[76]     D10 冲击负荷概率
[77]     D10 基荷 (50% 分位数)
[78]     mode_id (仅多模式训练, v2.14)
```

**动作空间（2维，v2.15 精简）**：`[p_ref, k_droop]`

- p_ref ∈ [-50, 50] kW（充电<0，放电>0），对齐下游 p_ref 符号约定
- k_droop ∈ [0, 30] kW/V（v3.1），对齐下游 parse_action_output clamp(0.0, 30.0)
- load_shedding/pv_limit 下沉至 strategy-engine, confidence 移至 ModelOutput 元数据 (v2.15)

Q_batt 由实时电压调节器闭环控制，不经过 RL 动作空间。

**5种场景模式**：


| 模式     | 优化目标                                           |
| -------- | -------------------------------------------------- |
| SCENE-01 | 台区季节性负荷模式 (MODE-01)：光伏消纳 + 防过载    |
| SCENE-B1 | 工商业模式-自主套利 (MODE-02)：峰谷价差 + 电池保护 |
| SCENE-B2 | 工商业模式-需量控制 (MODE-03)：减免需量罚金        |
| SCENE-B3 | 工商业模式-虚拟电厂 (MODE-04)：辅助服务 + 响应精度 |
| SCENE-B5 | 工商业模式-极致绿色 (MODE-05)：绿电消纳 + 碳减排   |

**关键物理约束**：

- SOC 硬限制：10%~90%（不可突破）
- 变压器容量：200 kVA，过载阈值 85%
- 动作约束 ACT-01/03/04/05（见 action_validator.py）

## 模块依赖

```
train.py
├── data_loader.py              (SmartDSLoader + ChinaDataLoader + UnifiedDataLoader)
├── config/                     (YAML 环境配置)
│   ├── config_manager.py       (YAML → dataclass, load_config/get_config)
│   └── mupc_env_config.yaml    (物理常数/动作空间/奖励权重, v3.0 版本指纹)
├── models/                     (ML 模型)
│   ├── lstm.py                 (LSTM + TCN + Attention + BiLSTM + QuantileLoss)
│   ├── error_correction.py     (BiLSTM 误差修正)
│   └── vmd.py                  (VMD 变分模态分解)
├── mupc_env/                   (Gymnasium 环境)
│   ├── core.py                 (MupcEnv 主类)
│   ├── observation.py          (EnvState + build_observation + normalize_obs)
│   ├── rewards.py              (5 场景奖励 + SCENE-01 子奖励)
│   ├── constants.py            (物理常数 + 归一化边界)
│   ├── voltage_sim.py          (VoltageSimulator 降级电压模型)
│   ├── state_builder.py        (build_env_state/build_reward_dict 纯函数)
│   ├── action_validator.py     (ACT-01~05 动作约束)
│   └── grid2op/                (Grid2Op + Pandapower 电压仿真)
├── stable_baselines3           (PPO/SAC，主路径)
│   └── _ppo_core.py           (NumPy PPO 后备)
└── export_onnx.py             (ONNX 导出)
```

**降级规则**：

- SB3 / Gymnasium / Torch 不可用 → `_ppo_core.py`（纯 NumPy PPO，v2.15 2 维 tanh 兼容）
- Gymnasium 不可用 → `_gym_stub.py`
- LSTM 未提供 → Oracle（真实值 + 噪声）

**数据加载入口**：

- `UnifiedDataLoader(lat, lon)` — 统一入口，自动检测并加载 SMART-DS 或中国合成数据
- `SmartDSLoader()` — SMART-DS 格式
- `ChinaDataLoader(region="Shanghai")` — 中国区域数据（支持 region 过滤）
- `ChinaSyntheticDataGenerator(lat, lon)` — 生成中国合成数据

## 环境参数


| 参数           | 值      |
| -------------- | ------- |
| 变压器容量     | 200 kVA |
| 电池容量       | 100 kWh |
| 最大充放电功率 | 50 kW   |
| SOC 硬限制     | 10%~90% |
| 过载阈值       | 85%     |
| 时间步长       | 15分钟  |
| 负荷峰值       | 60 kW   |
| 光伏容量       | 150 kW  |

## 代码规范

1. 安全相关代码标注 `SAFETY`
2. 动作 clamp 标注约束规则 ID (v2.15: ACT-01/02/03/04 + ACT-05)
3. 所有函数包含 Type Hints 和 Docstrings
4. `mupc_env/` 包不依赖 RL 框架，可独立运行

## 其他约束

所有输出都需要用中文输出
