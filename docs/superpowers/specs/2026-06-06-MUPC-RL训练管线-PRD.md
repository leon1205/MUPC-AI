# MUPC 强化学习模型训练管线 — 产品需求文档 (PRD)

| 版本 | 日期 | 作者 | 状态 |
|------|------|------|------|
| v2.0 | 2026-06-06 | 需求分析师 | **[REVIEWED: PASS]** |

**对应部署端 PRD:** `docs/MUPC/05-MUPC-AI引擎-PRD.md` v2.2 (`[REVIEWED: PASS]`)

---

> **v2.0 变更说明：** 本版本完全对齐 MUPC AI 引擎 PRD v2.2 的完整规格。状态空间从 6 维扩展为 48/49 维（21 字段序列化），动作空间从 2 维扩展为 4 维，奖励函数从简化版升级为 5 种场景的完整公式。新增 LSTM 预测模型训练和动作约束校验功能。旧版 v1.0 基于 CLAUDE.md 的简化方案已废弃。

---

## 1. 产品概述

### 1.1 产品定位

MUPC 强化学习模型训练管线是一个运行在本地 x86 PC 上的 Python 工具链。它负责训练两个核心模型并通过 ONNX 交付给 MUPC AI 引擎（RK3588 NPU 部署）：

1. **LSTM 时序预测模型** — 光伏出力 / 负荷功率时序预测
2. **PPO/SAC 强化学习决策模型** — 4 维动作空间的多目标优化控制

本管线是 MUPC AI 引擎的**模型供给侧**。训练的模型经 ONNX 导出后，由部署端进行 INT8 量化（rknn-toolkit2）并在 RK3588 NPU 上执行推理。

### 1.2 与 MUPC AI 引擎的对齐关系

```
┌──────────────────────────────────────────────────────────────────┐
│  本训练管线 (x86 PC, Python)                                       │
│                                                                    │
│  SMART-DS 数据 → 环境仿真(21字段) → PPO/SAC训练 → ONNX导出        │
│                  + LSTM训练                                       │
│  输出: lstm_forecast.onnx + rl_policy.onnx                       │
└──────────────────────────────────┬───────────────────────────────┘
                                   │ ONNX 交付
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  MUPC AI 引擎 (RK3588, Rust)                                      │
│                                                                    │
│  DataFusionEngine(5数据源) → FusedSystemState(21字段)             │
│    → to_input_vector() [48维] → RKNN Runtime → ActionOutput(4维) │
│    → ActionValidator(5条约束) → strategy-engine                  │
└──────────────────────────────────────────────────────────────────┘
```

**训练环境与部署环境的差异**：

| 维度 | 训练环境 | 部署环境 |
|------|----------|----------|
| 数据来源 | SMART-DS 数据集 + 合成数据 | 5 个实时数据源 (intercore/物联平台/气象API/gateway/LSTM) |
| 时间步长 | 15 分钟 | 1 秒 |
| 数据获取 | 文件读取 | DataSourceAdapter trait (TCP/MQTT/HTTP/Event) |
| LSTM 预测 | 训练环境生成（作为状态输入给 RL） | RKNN Runtime 实时推理 |
| 推理框架 | 不涉及 | RKNN Runtime (NPU) |
| 动作校验 | 环境内部 clamp | ActionValidator (5条规则) |

> **关键原则：** 训练环境的观测空间、动作空间和奖励函数规格完全对齐部署端 PRD v2.2。差异仅在于数据获取方式。

### 1.3 目标平台

| 项目 | 要求 |
|------|------|
| 运行环境 | x86 PC（训练用） |
| 操作系统 | Windows / Linux / macOS |
| Python 版本 | 3.9+ |
| RL 框架 | Stable-Baselines3 (SB3) PPO / SAC，纯 NumPy PPO 后备 |
| 环境接口 | Gymnasium（带 `_gym_stub.py` 降级） |
| 深度学习框架 | PyTorch（LSTM + ONNX 导出） |
| 模型格式 | ONNX（训练产出）→ .rknn（部署端 INT8 量化） |
| 数据源 | SMART-DS 数据集（光伏 CSV + 负荷 per-unit + Parquet） |

### 1.4 核心价值

| 价值 | 说明 | 量化目标 |
|------|------|----------|
| 完整对齐部署规格 | 训练即部署：观测/动作/奖励与 MUPC AI 引擎完全一致 | 模型导出后直接可用，无需额外适配 |
| 多场景单一策略 | 一个 RL 模型覆盖 5 种预设运行场景 (MODE-01~05) | 场景切换无需换模型 |
| 安全优先 | 安全惩罚梯度 + SOC 硬约束 + 动作约束校验 | 过载事件减少 > 90%（相比固定策略） |
| 端到端链路 | 数据加载 → LSTM训练 → RL训练 → ONNX导出，单条命令 | 无中间手动步骤 |

---

## 2. 用户角色

| 角色 | 描述 | 核心诉求 |
|------|------|----------|
| **AI 训练工程师** | 负责训练、调参、评估模型性能 | 一键训练、清晰的命令行参数、TensorBoard 监控 |
| **AI 运维人员** | 负责模型部署和监控（详见 MUPC AI 引擎 PRD 1.5） | 获取 ONNX 模型文件、了解模型输入输出规格 |
| **策略管理员** | 负责配置奖励权重和运行场景 | 理解各场景奖励函数含义、自定义权重 |

---

## 3. 核心功能

### 功能列表总览

| 编号 | 功能 | 优先级 | 说明 |
|------|------|--------|------|
| F1 | SMART-DS 数据加载与合成 | P0 | 加载光伏/负荷数据 + 合成缺失的 21 字段数据 |
| F2 | MUPC 全状态环境仿真 | P0 | 48 维观测 + 4 维动作 + 5 种场景奖励的 Gymnasium 环境 |
| F3 | LSTM 时序预测模型训练 | P0 | 光伏/负荷预测，输出 15 分钟预测向量 |
| F4 | 多模式 RL 训练 | P0 | PPO/SAC 训练，多模式单模型，48/49 维输入 |
| F5 | 动作约束校验 | P0 | 5 条约束规则，环境内 clamp |
| F6 | 模型导出 | P0 | LSTM + RL 策略网络 → ONNX，含 onnxruntime 验证 |
| F7 | 训练监控 | P1 | TensorBoard + CSV 日志，21 字段各自可追踪 |

---

### 3.2 F1：数据加载与状态合成

#### 功能描述

`data_loader.py` 负责加载 SMART-DS 真实数据，并**合成部署环境中由实时数据源提供的缺失字段**。

**从 SMART-DS 直接获取**：

| 字段 | 来源 | 说明 |
|------|------|------|
| pv_power | 光伏 CSV `kW Generated` 列 × 0.2 (缩放到 200kW) | D1 |
| load_power | 负荷 per-unit CSV × `LOAD_PEAK_KW` (400kW) | D1 |
| solar_irradiance | 光伏 CSV `PoA Irradiance` 列 | D5 |
| temperature | 光伏 CSV `Temperature` 列 | D5 |

**训练环境合成生成**：

| 字段 | 合成方法 | 说明 |
|------|----------|------|
| battery_soc | 环境仿真内部状态 | D1，reset 随机 [0.2, 0.8] |
| grid_power | 计算: `P_load - P_pv + P_batt` | D1 |
| transformer_load | 计算: `S_transformer / 500` | D1 |
| battery_power | 上一周期动作 A1 的值 | D1 |
| voltage_phase_a/b/c | **电压仿真模型**（见下方） | D1，v2.2 新增 |
| pv_forecast_15min | **LSTM 模型预测输出**（或 Oracle 后备） | D2 |
| load_forecast_15min | **LSTM 模型预测输出**（或 Oracle 后备） | D2 |
| current_electricity_price | **TOU 电价生成器** | D3 |
| next_period_price | 下一时段电价查表 | D3 |
| price_tariff_id | 时段枚举 {0=谷,1=平,2=峰,3=尖峰} | D3 |
| current_demand | 最近 15 分钟平均负荷功率 | D4 |
| contract_demand | 配置文件常量（默认 300 kW） | D4 |
| peak_demand_this_month | 滑动窗口跟踪本月峰值 | D4 |
| dispatch_p_set | 大部分时间 None；虚拟电厂模式合成 | D6 |
| dispatch_q_set | 大部分时间 None；虚拟电厂模式合成 | D6 |

**电压仿真模型**：

训练环境无法获取真实三相电压。采用简化线路模型模拟电压随功率变化：

```
V_phase = 1.0 + k_p * (P_pv - P_load + P_batt) / S_base
               - k_q * Q_batt / S_base
               + noise(σ=0.005)

其中:
  k_p = 0.05   (有功对电压的灵敏度)
  k_q = 0.03   (无功对电压的灵敏度，反映 Q-V 耦合)
  S_base = 500  (kVA)
  noise 模拟测量噪声
  clamp(V_phase, 0.85, 1.15)
```

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F1-01 | 加载全部 SMART-DS 光伏和负荷数据，打印统计摘要 | `python data_loader.py` 无错误 |
| F1-02 | TOU 电价按时段输出：谷 0.4/平 0.8/峰 1.2/尖峰 1.5 元/kWh | 单元测试 |
| F1-03 | 训练/验证集按时间 8:2 切分（不打乱顺序） | 单元测试 |
| F1-04 | 电压仿真值域在 [0.85, 1.15] p.u. 内 | 单元测试 |
| F1-05 | 合成数据覆盖全部 21 个状态字段 | 单元测试 |

---

### 3.3 F2：全状态环境仿真

#### 功能描述

`mupc_env.py` 实现基于 MUPC AI 引擎 PRD v2.2 完整规格的 Gymnasium 环境。

**观测空间（48 维序列化向量）**：

遵循 MUPC AI 引擎设计文档 `to_input_vector()` 布局：

```
索引      内容                      来源类别
[0..9]    D1 实时数据 (9标量)       SMART-DS + 环境仿真
[9..24]   D2 pv_forecast (15维)    LSTM 输出
[24..39]  D2 load_forecast (15维)  LSTM 输出
[39..42]  D3 电价 (3字段)          TOU 合成
[42..45]  D4 需量 (3字段)          合成
[45..47]  D5 气象 (2字段)          SMART-DS
[47]      D6 dispatch_p_set (1维)  合成 (None→0.0)
```

多模式训练时追加 mode_id 为第 49 维。

**动作空间（4 维）**：

| 维度 | 字段名 | 训练值域 | 单位 | 说明 |
|------|--------|----------|------|------|
| 0 | p_batt_set | [-1, 1] → [-500, 500] kW | kW | 电池有功功率设定值 |
| 1 | q_batt_set | [-1, 1] → [-300, 300] kVar | kVar | 无功功率设定值 |
| 2 | load_shedding | [0, 1] → [0, 500] kW | kW | 可中断负荷切除量 |
| 3 | pv_limit | [0, 1] → [0, 1] 比例 | - | 光伏限功率比例 |

> 训练时动作归一化到 [-1,1] 或 [0,1]，环境内部映射到物理值。这与部署端 ActionOutput 物理值一一对应。

**核心物理方程**：

```
P_batt = p_batt_set_norm * 500  (kW)
Q_batt = q_batt_set_norm * 300  (kVar)
P_load_eff = P_load - load_shedding  (kW, 切除后有效负荷)
P_pv_eff = P_pv * pv_limit  (kW, 限功率后有效光伏)

SOC_raw = SOC_t + (-P_batt * dt) / BATTERY_CAPACITY_KWH
SOC_{t+1} = clamp(SOC_raw, 0.10, 0.90)  // SAFETY: 硬约束

grid_power = P_load_eff - P_pv_eff + P_batt
Q_load = P_load_eff * tan(acos(0.90))  (功率因数 0.90)
S_transformer = sqrt(grid_power² + (Q_load - Q_batt)²)
load_rate = S_transformer / TRANSFORMER_KVA (500)

电压仿真: voltage_phase_{a,b,c} = f(P, Q) 见 3.2 节
需量: current_demand = max(最近4步 P_load_eff 的滑动平均值, 前值)
```

**环境常量**：

| 参数 | 值 | 常量名 |
|------|-----|--------|
| 变压器容量 | 500 kVA | `TRANSFORMER_KVA` |
| 电池容量 | 200 kWh | `BATTERY_CAPACITY_KWH` |
| 最大充放电功率 | 500 kW (p_batt 范围上限) | `P_BATT_MAX` |
| 最大无功功率 | 300 kVar (q_batt 范围上限) | `Q_BATT_MAX` |
| SOC 硬限制 | 10% ~ 90% | `SOC_MIN`, `SOC_MAX` |
| 过载阈值 | 85% | `OVERLOAD_THRESHOLD` |
| 时间步长 | 15 分钟 (0.25 h) | `DT` |
| 合同需量 | 300 kW (可配置) | `CONTRACT_DEMAND` |
| 碳排放因子 | 0.581 kg CO2/kWh (可配置) | `GRID_EMISSION_FACTOR` |

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F2-01 | `mupc_env.py` 可独立运行随机动作循环 100 步 | `python mupc_env.py` 无错误 |
| F2-02 | `observation_space.shape = (48,)` | 单元测试 |
| F2-03 | `action_space.shape = (4,)` | 单元测试 |
| F2-04 | SOC 硬约束不可突破 | 单元测试：连续充电 1000 步，验证 SOC ≤ 0.90 |
| F2-05 | info dict 包含全部奖励分量原始值 + SOC + load_rate | 单元测试 |
| F2-06 | 电压仿真三相电压在 [0.85, 1.15] 内 | 单元测试 |
| F2-07 | 兼容 gymnasium.Env 和 _gym_stub 双重接口 | 集成测试 |
| F2-08 | terminal_observation 包含完整 48 维观测 | 单元测试 |

---

### 3.4 F3：LSTM 时序预测模型

#### 功能描述

`lstm_model.py` 训练 LSTM 模型用于光伏/负荷的 15 分钟超前预测。预测输出作为 RL 环境状态空间 D2 的输入。

**模型规格**：

| 项目 | 规格 |
|------|------|
| 输入窗口 | 过去 60 分钟 (4 步 × 15分钟) |
| 输出窗口 | 未来 15 分钟 (1 步 × 15分钟)（默认，可配置至 30 分钟） |
| 输入特征 | [pv_power, load_power, solar_irradiance, temperature, hour_sin, hour_cos] (6 维) |
| 输出 | [pv_forecast_1..15, load_forecast_1..15] (30 维，15+15) |
| 模型架构 | 2 层 LSTM (hidden=64) + Linear head |
| 精度要求 | 光伏 MAPE ≤ 10%，负荷 MAPE ≤ 15% |
| 训练数据 | SMART-DS 光伏 + 负荷数据，按时间 8:2 切分 |

**输出序列化**：LSTM 输出 15 个时间步的预测值。当部署端配置为 30 分钟时，超出部分补零填充。

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F3-01 | LSTM 模型训练完成，loss 收敛 | 检查训练 loss 曲线 |
| F3-02 | 光伏预测 MAPE ≤ 10%（测试集） | 回测计算 |
| F3-03 | 负荷预测 MAPE ≤ 15%（测试集） | 回测计算 |
| F3-04 | 预测输出形状 (1, 30) = 15 pv + 15 load | 单元测试 |
| F3-05 | ONNX 导出 + checker 验证通过 | `python export_onnx.py --lstm` |

---

### 3.5 F4：多模式 RL 训练

#### 功能描述

`train.py` 是训练主入口，支持 PPO/SAC 算法和多模式单模型训练。

**命令行参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `all` | 训练模式：`MODE-01`~`MODE-05` 单模式，或 `all` 多模式 |
| `--algo` | `ppo` | 算法：`ppo` / `sac` |
| `--total-timesteps` | 175200 | 总训练步数 |
| `--reward-weights` | 按场景默认 | 自定义权重 `w1=0.6,w2=0.3` |
| `--seed` | 42 | 随机种子 |
| `--lstm-model` | `None` | 预训练 LSTM 模型路径（不指定则使用 Oracle 后备） |
| `--no-lstm` | False | 使用 Oracle 预测（真实未来值 + 噪声）代替 LSTM |

**5 种场景与奖励函数**（与 MUPC AI 引擎 PRD 第 6 章完全对齐）：

| 场景 ID | 命令行 | 优化目标 | 公式 |
|---------|--------|----------|------|
| MODE-01 | `MODE-01` | 农网灌溉：最大化光伏消纳 + 防止过载 | `R = w1·R_pv_consumption - w2·P_battery_degradation - w3·P_transformer_overload` |
| MODE-02 | `MODE-02` | 自主套利：最大化峰谷价差 + 最小化电池损耗 | `R = w1·R_price_spread - w2·P_battery_degradation` |
| MODE-03 | `MODE-03` | 需量控制：减免需量罚金 | `R = w1·R_demand_penalty_avoidance - w2·P_comfort_loss` |
| MODE-04 | `MODE-04` | 虚拟电厂：辅助服务收益 + 响应精度 | `R = w1·R_ancillary_service + w2·R_response_accuracy - w3·P_deadline_deviation` |
| MODE-05 | `MODE-05` | 极致绿色：最大化绿电消纳 + 最小化碳排放 | `R = w1·R_green_consumption + w2·R_carbon_reduction` |

> 各场景奖励公式的完整定义见 MUPC AI 引擎 PRD 第 6.2~6.6 节。本环境在 `mupc_env.py` 中逐项实现。

**多模式训练策略**：
- `--mode all`（默认）：每个 episode 随机选择一种场景，mode_id 编码追加到观测向量（49 维），训练单一模型覆盖全部 5 种场景
- `--mode MODE-01`：单场景训练，观测为 48 维

**PPO 网络结构**：

```
Input(48 or 49) → Linear(128) → ReLU → Linear(128) → ReLU
                       ├── actor:  Linear(4)  → Tanh (A1,A2) / Sigmoid (A3,A4)
                       └── critic: Linear(1)
```

**SAC 网络结构**：同上，actor 输出 mean + log_std。

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F4-01 | `python train.py --total_timesteps 50000` 完成不崩溃 | 集成测试 |
| F4-02 | 支持 PPO 和 SAC，分别生成 checkpoint | 集成测试 |
| F4-03 | `--mode all` 时每个 episode 随机切换场景 | 检查 CSV 日志 mode 分布 |
| F4-04 | `--mode MODE-01` 时所有 episode 使用同一种奖励函数 | 检查 CSV 日志 |
| F4-05 | SB3 不可用时自动切换 `_ppo_core.py` | 卸载 SB3 后测试 |
| F4-06 | Ctrl+C 中断保存 checkpoint 不丢失进度 | 手动测试 |
| F4-07 | TensorBoard 中可监控所有奖励分量 + 4 个动作维度的均值 | 手动检查 |

---

### 3.6 F5：动作约束校验

#### 功能描述

遵循 MUPC AI 引擎 PRD 第 5.4 节的 5 条约束规则，在环境 `step()` 中对 RL 输出的动作进行校验和 clamp。

| 规则 ID | 约束条件 | 训练环境实现 |
|---------|----------|-------------|
| ACT-01 | Δp_batt ≤ 50 kW/步 | 计算变化率，超标则 clamp |
| ACT-02 | Δq_batt ≤ 30 kVar/步 | 计算变化率，超标则 clamp |
| ACT-03 | √(p_batt²+q_batt²) ≤ 500 kVA | 超标则等比例缩放回功率圆内 |
| ACT-04 | pv_limit ≥ 0.1（防逆流时除外） | clamp 到 [0.1, 1.0] |
| ACT-05 | dispatch_p 有效时 \|p_batt\| ≤ \|dispatch_p\| | 有调度时 clamp |

> 部署端 ActionValidator 在 Rust 端同样实现这 5 条规则。训练时在环境内执行校验，让 RL agent 在与部署相同的约束下学习。

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F5-01 | 5 条约束均实现 | 单元测试：逐条触发违规并验证 clamp 结果 |
| F5-02 | 约束违反时 info dict 中记录 `constraint_violated=True` | 单元测试 |
| F5-03 | 约束校验耗时 < 0.5ms | 性能测试 |

---

### 3.7 F6：模型导出

#### 功能描述

`export_onnx.py` 导出两种模型为 ONNX：

1. **LSTM 预测模型**：`lstm_forecast.onnx`，输入 (1, 4, 6) (batch, seq_len, features)，输出 (1, 30)
2. **RL 策略网络**：`rl_policy.onnx`，输入 (1, 48) 或 (1, 49)，输出 (1, 4)

导出流程：
1. 加载 PyTorch checkpoint / NumPy PPO weights
2. 构建等效 PyTorch 模型
3. `torch.onnx.export()`
4. `onnx.checker.check_model()` 验证
5. onnxruntime 推理验证（误差 < 1e-5）

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F6-01 | RL 策略 ONNX 输入 (1, 48) 输出 (1, 4) | 检查 ONNX spec |
| F6-02 | LSTM ONNX 输入 (1, 4, 6) 输出 (1, 30) | 检查 ONNX spec |
| F6-03 | ONNX 推理与 PyTorch 推理误差 < 1e-5 | 单元测试 |
| F6-04 | 导出文件名含时间戳 | 检查文件名 |
| F6-05 | 无 SB3 checkpoints 时自动从 npz 导出 | 测试 npz 路径 |

---

### 3.8 F7：训练监控

| 指标 | 输出位置 |
|------|----------|
| episode 奖励（总和 + 各分量） | TensorBoard + CSV |
| 4 个动作维度的均值/最大值 | TensorBoard |
| SOC 均值、负载率均值、过载次数 | TensorBoard + 控制台 |
| 当前场景 ID | CSV |
| 训练 loss（actor/critic） | TensorBoard |
| 学习率 | TensorBoard |

控制台每隔 10000 步打印摘要。

---

## 4. 非功能性需求

### 4.1 性能

| 指标 | 要求 |
|------|------|
| 环境 step() 耗时（不含推理） | < 2ms |
| 训练吞吐（SB3 PPO, CPU） | > 500 steps/s |
| 峰值内存 | < 8GB |

### 4.2 模型规格

| 指标 | 要求 |
|------|------|
| RL 策略 ONNX 大小 | < 10MB |
| LSTM ONNX 大小 | < 5MB |
| RL 网络参数量 | < 100k |

### 4.3 兼容性

| 指标 | 要求 |
|------|------|
| Python 最低版本 | 3.9 |
| 第三方包上限 | 不超过 6 个 (SB3, gymnasium, torch, onnx, onnxruntime, numpy) |
| 降级方案 | SB3 → NumPy PPO; Gymnasium → _gym_stub; torch → 仅 numpy 推理 |

### 4.4 代码质量

- 所有公共函数包含 Type Hints 和 Docstrings
- `mupc_env.py` 不依赖 RL 框架，可独立运行
- SOC guard 代码标注 `SAFETY`
- 所有动作 clamp 标注约束规则 ID (ACT-01~05)

---

## 5. 边界条件与异常流程

### 5.1 数据异常

| 场景 | 处理 |
|------|------|
| SMART-DS 目录不存在 | 打印提示运行 `data/download_smart_ds.py`，退出 |
| 部分光伏 CSV 缺失 | 跳过缺失文件，打印 WARN，继续加载 |
| CSV 含 NaN/Inf | 跳过该行 |
| 数据不足一整年 | 打印实际天数，调整 total_timesteps 建议 |

### 5.2 训练异常

| 场景 | 处理 |
|------|------|
| SB3 不可用 | 自动切换 `_ppo_core.py` NumPy PPO |
| Gymnasium 不可用 | 自动切换 `_gym_stub.py` |
| GPU 不可用 | 自动回退 CPU |
| LSTM 模型未提供且无 Oracle | 打印 ERROR，退出（D2 预测数据是必需的） |
| Ctrl+C | 保存 checkpoint |
| loss 爆炸 (> 1e6) | 停止训练，保存崩溃前 checkpoint |

### 5.3 环境边界

| 场景 | 处理 |
|------|------|
| SOC 达 90% 仍收到充电指令 | p_batt clamp 为 0，info 标记 `soc_clipped` |
| SOC 达 10% 仍收到放电指令 | p_batt clamp 为 0，info 标记 `soc_clipped` |
| load_rate > 150% | 安全惩罚急剧增长，但仍允许 step |
| 电压超出 [0.85, 1.15] | clamp 到边界值 |

### 5.4 导出异常

| 场景 | 处理 |
|------|------|
| checkpoint 不存在 | 打印 ERROR 退出 |
| ONNX checker 不通过 | 打印详细错误，不输出文件 |
| onnxruntime 不可用 | 仅 checker 验证，跳过推理比对 |

---

## 6. 文件结构

```
MUPC-AI2/
├── data_loader.py              # F1: 数据加载 + 状态合成
├── mupc_env.py                 # F2: Gymnasium 环境 (48/49 维, 4 维动作)
├── lstm_model.py               # F3: LSTM 训练
├── train.py                    # F4: RL 训练主入口
├── action_validator.py         # F5: 动作约束校验
├── export_onnx.py              # F6: ONNX 导出
├── _ppo_core.py                # 纯 NumPy PPO 后备
├── _gym_stub.py                # Gymnasium 最小替代
├── data/
│   ├── download_smart_ds.py    # 数据集下载 (已存在)
│   └── smart_ds/               # SMART-DS 数据 (已下载)
├── checkpoints/
├── exported_models/
└── tensorboard_logs/
```

---

## 附录 A：待澄清问题

| 序号 | 问题 | 优先级 | 影响 |
|------|------|--------|------|
| 1 | dispatch_q_set 是否需要加入输入向量？设计文档 to_input_vector() 仅包含 dispatch_p_set（47 号索引），不含 dispatch_q_set | 高 | 影响输入向量维度 |
| 2 | VPP 场景的 capacity_price 和 mileage_price 从何获取？训练时使用固定假设值？ | 中 | 影响 MODE-04 奖励函数实现 |
| 3 | 负荷预测区分基荷/可调负荷/冲击负荷三类，训练数据如何标注？SMART-DS 不含此分类 | 高 | 影响 LSTM 训练数据准备 |

---
