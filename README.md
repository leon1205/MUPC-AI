# MUPC 强化学习模型训练管线

MUPC 强化学习模型训练管线，在 x86 PC 上训练 RL 模型，导出 ONNX 交付给 MUPC AI 引擎（RK3588 NPU 部署）。

## 核心规格

| 项目 | 值 |
|------|-----|
| 变压器容量 | 200 kVA |
| 电池容量 | 100 kWh |
| 最大放电功率 | 50 kW |
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
mupc_env/            → Gymnasium 环境：78/79维观测 + 2维动作 + 5场景奖励 (v2.14)
│ └── grid2op_env/ → Grid2Op + Pandapower 三相潮流电压仿真（可选）
        ↓
train.py → SB3 PPO/SAC 训练（主路径），NumPy PPO 后备
        ↓
export_onnx.py       → ONNX 导出
        ↓
MUPC AI Engine (Rust, RK3588 NPU)
```

## 快速命令

```bash
# ============================================================
# 独立 LSTM 训练（不跑 RL）
# ============================================================
python train.py --train-lstm --data-source merged --lstm-params epochs=100

# 增强 LSTM 训练（epochs=200, hidden=128, layers=3, patience=30）
python train.py --train-lstm --data-source smartds --lstm-params hidden_dim=128,num_layers=3,epochs=200,patience=30

# ============================================================
# RL 训练（使用已有关 LSTM checkpoint）
# ============================================================
# 快速训练测试（5万步，约1分钟）
python train.py --mode MODE-01 --total-timesteps 50000 --no-lstm

# 完整训练（多模式单模型，一个模型覆盖全部5种场景）
python train.py --mode all --total-timesteps 175200

# 5个独立模型训练（推荐生产方案）
python train.py --mode single --total-timesteps 200000

# 使用指定 LSTM checkpoint +导出 ONNX
python train.py --mode MODE-01 --data-source smartds \
       --lstm-checkpoint checkpoints/lstm_checkpoint.pt \
       --total-timesteps 1000000 --export-onnx

# ============================================================
# LSTM 训练 + RL 训练（合并数据，1M步）
# ============================================================
python train.py --mode MODE-01 --data-source merged --train-lstm \
       --lstm-params hidden_dim=128,num_layers=3,epochs=200,patience=30 \
       --total-timesteps 1000000 --export-onnx

# ============================================================
# 课程学习训练（v2.6 新增）
# ============================================================
# Phase 1: MODE-02 (基础) → Phase 2: MODE-01 (进阶) → Phase 3: 混合模式 (对抗)
python train.py --curriculum --total-timesteps 175200

# ============================================================
# Grid2Op / VoltageSimulator 切换
# ============================================================
# Grid2Op 模式（三相潮流计算）+ 中国合成数据
# 重要：Grid2Op 模拟农网台区场景，应使用中国合成数据而非 SMART-DS
python train.py --mode MODE-01 --data-source china --total-timesteps 100000

# Grid2Op + 中国经纬度（上海）
python train.py --mode MODE-01 --data-source china --lat 31.23 --lon 121.47 --total-timesteps 100000

# VoltageSimulator 降级模式（简化 Q-V 耦合）
python train.py --mode MODE-01 --data-source china --total-timesteps 100000 --no-grid2op

# ============================================================
# 其他选项
# ============================================================
# 中国数据 + 低学习率衰减
python train.py --mode all --data-source china --lr-decay --total-timesteps 500000

# 自定义网络结构 +熵系数
python train.py --mode MODE-01 --total-timesteps 500000 \
       --net-arch pi=256,128,64 vf=256,128,64 --ent-coef 0.02

# 自定义评估频率
python train.py --mode MODE-01 --total-timesteps 500000 --eval-freq 5000

# ============================================================
# 环境自测与工具
# ============================================================
# 环境自测
python mupc_env.py

# 数据加载器自测
python data_loader.py

# 导出 ONNX
python export_onnx.py --checkpoint checkpoints/MODE-01_model.zip
python export_onnx.py --lstm checkpoints/lstm_checkpoint.pt

# 下载数据集
python data/download_smart_ds.py

# 生成中国合成数据（需指定经纬度）
python data_loader.py --generate --lat 31.23 --lon 121.47 --year 2023

# 统一数据加载（自动检测数据源，支持中国合成/SMART-DS）
python data_loader.py --unified --lat 31.23 --lon 121.47
```

## train.py 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `all` | 训练模式：`MODE-01`~`MODE-05` 单模式，或 `all` 多模式 |
| `--algo` | `ppo` | 算法：`ppo` / `sac` |
| `--total-timesteps` | 175200 | 总训练步数 |
| `--data-source` | `smartds` | 数据源：`smartds` / `china` / `merged` / `unified` |
| `--lstm-checkpoint` | `None` | 预训练 LSTM 模型路径 |
| `--lstm-params` | `hidden_dim=64,num_layers=2,epochs=100,patience=15` | LSTM 训练参数 |
| `--train-lstm` | `False` | 独立训练 LSTM（不跑 RL） |
| `--no-lstm` | `False` | 使用 Oracle 预测代替 LSTM |
| `--algo-backend` | `auto` | RL 后端 (v2.17 SB3 主路径): `auto` / `sb3` / `numpy` |
| `--no-grid2op` | `False` | 使用 VoltageSimulator 代替 Grid2Op |
| `--export-onnx` | `False` | 训练结束后导出 ONNX |
| `--lr-decay` | `False` | 启用学习率衰减 |
| `--ent-coef` | `0.01` | 熵系数 |
| `--net-arch` | `pi=128,128,vf=128,128` | 网络结构 |
| `--eval-freq` | `10000` | 评估频率（步） |
| `--reward-weights` | 按场景默认 | 自定义奖励权重 |
| `--seed` | `42` | 随机种子 |

## 观测空间（78维单模式，79维多模式，v2.14 对齐下游 AI 引擎）

```
[0..8]    D1 实时数据 (9标量)         SOC/光伏/负荷/电网功率/变压器负载/电池功率/三相电压 (q_margin 移至 D7)
[9..23]   D2 光伏预测 (15维)          LSTM或Oracle
[24..38]  D2 负荷预测 (15维)          LSTM或Oracle
[39..41]  D3 电价 (3字段)             当前价/下时段价/时段ID (peak/valley_price 仅日志, 不入向量)
[42..44]  D4 需量 (3字段)             当前需量/合同值/本月峰值
[45..46]  D5 气象 (2字段)             辐照/温度
[47]      D6 调度有功指令              dispatch_p_set (dispatch_q_set 仅日志, 不入向量)
[48]      D7 Q裕度                    q_realtime_margin (v2.14 移至 D7)
[49..54]  D8 季节one-hot (6维)        灌溉季/炒茶季/空调季/常规季/保留/保留
[55..56]  D8 时段one-hot (2维)        白天/夜间
[57..60]  D9 安全覆盖 (4字段)         active/p_ref/consecutive/ratio
[61..75]  D10 分位数负荷预测 (15维)   P3.3~P96.7 等距 15 步
[76]      D10 冲击负荷概率 (0~1)
[77]      D10 基荷 (50% 分位数)
[78]      (可选) mode_id              多模式训练追加 (多模式为第79位)
```

## 动作空间（2维，v2.15 精简）

| 维度 | 范围 | 说明 |
|------|------|------|
| p_ref | [-50, 50] kW | 电池有功基准点（RL 控制），匹配电池最大充放电功率 |
| k_droop | [0, 30] kW/V | 电压-有功下垂系数（RL 控制），下垂公式: P_output = p_ref - k_droop × ΔV |

Q_batt 由实时电压调节器闭环，不经过 RL。  
load_shedding / pv_limit 已下沉至 strategy-engine 本地策略（不在 RL 动作空间）。  
confidence 改为 ModelOutput 元数据（不在 RL 动作空间）。

> **配置说明**：以上参数由 `config/mupc_env_config.yaml` 统一管理，修改配置文件即可调整动作空间范围，无需改动代码。

## 5种场景模式

| 场景 | 优化目标 |
|------|----------|
| SCENE-01: 台区季节性负荷模式 (MODE-01) | 最大化光伏消纳 + 防止变压器过载 + 电池寿命保护 |
| SCENE-B1: 工商业模式-自主套利 (MODE-02) | 最大化峰谷电价差收益 + 最小化电池损耗 |
| SCENE-B2: 工商业模式-需量控制 (MODE-03) | 减免需量罚金 + 最小化舒适度损失 |
| SCENE-B3: 工商业模式-虚拟电厂 (MODE-04) | 最大化辅助服务收益 + 响应精度 |
| SCENE-B5: 工商业模式-极致绿色 (MODE-05) | 最大化绿电消纳比例 + 最小化碳排放 |

## SCENE-01 奖励函数特性

- **自适应损耗系数 α(s)** ∈ {3.0, 0.2, 1.0}
- **条件触发电压惩罚**：q_realtime_margin ≤ 10% 且越限 ≥ 2 步才触发
- **弃光电压前置**：v_avg ≥ 1.05 p.u. → R_pv = 0
- **功率变化率惩罚**：R_ramp = w5 · |ΔP_batt| / BATTERY_CAPACITY_KWH
- **电压变化斜率惩罚（v2.6 新增）**：R_slope = w6 · |ΔV|，迫使 AI 平滑调节

## 电压仿真引擎（v2.3 新增）

训练环境支持两种电压仿真引擎：

| 模式 | 参数 | 说明 |
|------|------|------|
| Grid2Op + Pandapower | 默认启用 | 三相潮流计算，精度高 |
| VoltageSimulator | `--no-grid2op` | 简化 Q-V 耦合灵敏度系数，降级方案 |

**Grid2Op 模式**：
- 后端优先级：lightsim2grid（C++）> PandaPowerBackend（Python）
- 三相电压由 Pandapower `run_pp()` 计算
- 每步仿真目标 ≤ 50ms（lightsim2grid 加速）

**数据源要求**：
- **重要**：Grid2Op 模拟农网台区场景，**必须使用中国合成数据**（`--data-source china`）
- SMART-DS 数据来自美国加州，与中国农网场景不匹配
- 如检测到 SMART-DS 数据，NumpyChronics 会输出 WARN 警告

**农业冲击负荷（季节性）**：
- 6-9月（灌溉季）：50% 概率触发，80~120kW，持续 2~4 小时
- 其他月份：20% 概率触发，30~80kW，持续 1~2 小时
- 居民负荷：叠加 ±10% 随机噪声模拟家用电器启停

**VoltageSimulator 降级模式**：
- 使用简化灵敏度系数：`V = 1.0 + k_p·P_net/S_base - k_q·Q/S_base + noise`
- k_p=0.05, k_q=0.03, S_base=500 kVA

## 关键物理约束

- SOC 硬限制：10%~90%（不可突破）
- 变压器容量：200 kVA，过载阈值 85%
- 动作约束 (v2.15): ACT-01 (Δp_ref ≤ 50kW/步) / ACT-02 (Δk_droop ≤ 10kW/V/步) / ACT-03 (p_ref ∈ [-50,50] kW) / ACT-04 (k_droop ∈ [0,30] kW/V) / ACT-05 (|p_ref| ≤ |dispatch_p|)

## 项目结构

```
MUPC-AI2/
├── data_loader.py              # SMART-DS 数据加载 + 状态合成
├── mupc_env.py               # 兼容重定向 (→ mupc_env/ 包)
├── mupc_env/                 # Gymnasium 环境 (78/79维, 2维动作, Grid2Op集成, v2.15 模块化)
│   ├── __init__.py           # 仅导出 MupcEnv
│   ├── constants.py          # 物理常数 + 归一化边界 + 权重配置
│   ├── voltage_sim.py        # VoltageSimulator (Grid2Op 优先, 自动降级)
│   ├── observation.py        # EnvState + build_observation + normalize_obs
│   ├── rewards.py            # 5 场景奖励 + 13 个 SCENE-01 子奖励
│   └── core.py               # MupcEnv 主类
├── lstm_model.py             # LSTM 预测模型 / Oracle 后备
├── train.py                  # SB3 PPO/SAC 主入口 (NumPy PPO 后备)
├── action_validator.py        # 动作约束 (ACT-01/02/03/04 + ACT-05, v2.15 4+1 条)
├── export_onnx.py           # ONNX 导出 (RL 策略 act_dim=2)
├── _ppo_core.py             # 纯 NumPy PPO 后备 (v2.15 act_dim=2)
├── _gym_stub.py            # Gymnasium 桩模块
├── grid2op_env/ # Grid2Op电压仿真引擎（v2.3 新增）
│   ├── __init__.py
│   ├── numpy_chronics.py   # NumpyChronics: data dict → Grid2Op 三相格式
│   ├── power_flow.py        # Grid2OpPowerFlow: Grid2Op 引擎封装
│   ├── network.py           # create_mupc_network(): Pandapower 拓扑
│   └── backend.py           # Backend 选择 (lightsim vs pandapower)
├── data/
│   ├── download_smart_ds.py # 数据集下载
│   └── smart_ds/            # SMART-DS 数据
├── checkpoints/              # 训练产出权重
├── exported_models/         # ONNX 模型
└── tensorboard_logs/ # 训练日志
```

## 降级规则 (v2.17 SB3 主路径)

- SB3 / Gymnasium / Torch 任一不可用 → `_ppo_core.py`（纯 NumPy PPO 后备）
- `--algo-backend=numpy` 强制 NumPy PPO；`--algo-backend=sb3` 强制 SB3（不可用时 WARN 降级）
- Gymnasium 不可用 → `_gym_stub.py`
- LSTM 未提供 → Oracle（真实值 + 噪声）
- Grid2Op不可用 → VoltageSimulator（简化电压模型）
- lightsim2grid 不可用 → PandaPowerBackend（Python）

## 性能说明

| 模式 | step() 耗时 |训练吞吐 |
|------|-------------|----------|
| VoltageSimulator 模式 | < 2ms | > 500 steps/s |
| Grid2Op + lightsim2grid | ≤ 50ms |视硬件，约 20~250 steps/s |
| Grid2Op + PandaPowerBackend | 10~50ms | 视硬件 |

Grid2Op 模式训练速度下降约 2~13 倍（相比 VoltageSimulator），但电压计算精度更高。

## 代码规范

- 安全相关代码标注 `SAFETY`
- 动作 clamp 标注约束规则 ID (v2.15: ACT-01/02/03/04 + ACT-05)
- 所有函数包含 Type Hints 和 Docstrings
- `mupc_env.py`/`mupc_env/` 不依赖 RL 框架，可独立运行