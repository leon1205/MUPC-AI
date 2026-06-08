# MUPC 强化学习模型训练管线

MUPC 强化学习模型训练管线，在 x86 PC 上训练 RL 模型，导出 ONNX 交付给 MUPC AI 引擎（RK3588 NPU 部署）。

## 核心规格

| 项目 | 值 |
|------|-----|
| 变压器容量 | 500 kVA |
| 电池容量 | 200 kWh |
| 最大充放电功率 | 500 kW |
| SOC 硬限制 | 10%~90% |
| 过载阈值 | 85% |
| 时间步长 | 15 分钟 |

## 架构

```
SMART-DS CSV/Parquet
        ↓
data_loader.py       → 加载光伏/负荷/气象，合成 TOU电价/需量/调度
        ↓
lstm_model.py        → LSTM 预测模型（或 Oracle 后备）
        ↓
mupc_env.py          → Gymnasium 环境：58维观测 + 2维动作 + 5场景奖励
        ↓
train.py             → PPO/SAC 训练（或 NumPy PPO 后备）
        ↓
export_onnx.py       → ONNX 导出
        ↓
MUPC AI Engine (Rust, RK3588 NPU)
```

## 快速命令

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

## 观测空间（58维单模式，59维多模式）

```
[0..9]   D1 实时数据: SOC/光伏/负荷/电网功率/变压器负载/电池功率/三相电压/实时模块Q裕度
[10..24] D2 光伏预测 (15维，LSTM或Oracle)
[25..39] D2 负荷预测 (15维，LSTM或Oracle)
[41..43] D3 电价: 当前价/下时段价/时段ID
[44..46] D4 需量: 当前需量/合同值/本月峰值
[47..48] D5 气象: 辐照/温度
[49]     D6 调度指令
[50]     D7 Q裕度
[51..56] D7 季节one-hot: 灌溉季/炒茶季/空调季/常规季/保留/保留
[57]     D7 时段: 白天/夜间
[58]     mode_id (仅多模式训练)
```

## 动作空间（2维）

| 维度 | 范围 | 说明 |
|------|------|------|
| p_batt | [-500, 500] kW | 电池有功（RL 控制） |
| load_shedding | [0, 500] kW | 可中断负荷切除（RL 控制） |

Q_batt 由实时电压调节器闭环，不经过 RL。

## 5种场景模式

| 模式 | 优化目标 |
|------|----------|
| MODE-01 | 农网灌溉：光伏消纳 + 防过载 + 电压质量 |
| MODE-02 | 自主套利：峰谷价差 + 电池保护 |
| MODE-03 | 需量控制：减免需量罚金 |
| MODE-04 | 虚拟电厂：辅助服务 + 响应精度 |
| MODE-05 | 极致绿色：绿电消纳 + 碳减排 |

## SCENE-01 奖励函数特性

- **自适应损耗系数 α(s)** ∈ {3.0, 0.2, 1.0}
- **条件触发电压惩罚**：q_realtime_margin ≤ 10% 且越限 ≥ 2 步才触发
- **弃光电压前置**：v_avg ≥ 1.05 p.u. → R_pv = 0
- **功率变化率惩罚**：R_ramp = w5 · |ΔP_batt| / BATTERY_CAPACITY_KWH

## 关键物理约束

- SOC 硬限制：10%~90%（不可突破）
- 变压器容量：500 kVA，过载阈值 85%
- 动作约束 ACT-01/03/05（Q/功率圆/pv_limit 由实时控制处理）

## 项目结构

```
MUPC-AI2/
├── data_loader.py              # SMART-DS 数据加载 + 状态合成
├── mupc_env.py               # Gymnasium 环境 (58/59维, 2维动作)
├── lstm_model.py             # LSTM 预测模型 / Oracle 后备
├── train.py                  # PPO/SAC 训练主入口
├── action_validator.py        # 3条动作约束 (ACT-01/03/05)
├── export_onnx.py           # ONNX 导出
├── _ppo_core.py             # 纯 NumPy PPO 后备
├── _gym_stub.py            # Gymnasium 桩模块
├── train_single_modes.py    # 5场景独立模型训练
├── data/
│   ├── download_smart_ds.py # 数据集下载
│   └── smart_ds/            # SMART-DS 数据
├── checkpoints/              # 训练产出权重
├── exported_models/         # ONNX 模型
└── tensorboard_logs/       # 训练日志
```

## 降级规则

- SB3 不可用 → `_ppo_core.py`（纯 NumPy PPO）
- Gymnasium 不可用 → `_gym_stub.py`
- LSTM 未提供 → Oracle（真实值 + 噪声）

## 代码规范

- 安全相关代码标注 `SAFETY`
- 动作 clamp 标注约束规则 ID (ACT-01~05)
- 所有函数包含 Type Hints 和 Docstrings
- `mupc_env.py` 不依赖 RL 框架，可独立运行
