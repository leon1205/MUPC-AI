# MUPC 强化学习模型训练管线 — 产品需求文档 (PRD)

| 版本 | 日期 | 作者 | 状态 |
|------|------|------|------|
| v2.13 | 2026-06-14 | 需求分析师 | **[REVIEWED: PASS]** |
| v2.12 | 2026-06-14 | 需求分析师 | **[REVIEWED: PASS]** |
| v2.11 | 2026-06-13 | 需求分析师 | **[REVIEWED: PASS]** |
| v2.10 | 2026-06-13 | 需求分析师 | **[REVIEWED: PASS]** |
| v2.9 | 2026-06-13 | 需求分析师 | **[REVIEWED: PASS]** |
| v2.8 | 2026-06-11 | 架构师 | **[REVIEWED: PASS]** |
| v2.7 | 2026-06-11 | 架构师 | **[REVIEWED: PASS]** |
| v2.6 | 2026-06-11 | 需求分析师 | **[REVIEWED: PASS]** |

**对应部署端 PRD:** `docs/MUPC/05-MUPC-AI引擎-PRD.md` v2.12 (`[REVIEWED: PASS]`)（符号链接 → MUPC2/superpowers/specs/modules/）

---

> **v2.13 变更说明（对齐部署端 v2.12）：** SCENE-01 奖励函数三项改进（R-01~R-03 已实现）。R-01：各奖励子项标准化到 [-1,1] 区间（r_pv_norm、p_batt_deg_norm、p_overload_norm、r_pq_norm、p_ramp_norm、p_voltage_slope_norm、r_smooth_norm、r_safety_override_norm），统一量纲加速 RL 收敛。R-02：引入塑造奖励 overload_warning（负载率 >85% 开始预警）和 soc_warning（SOC 接近 15%/85% 边界时预警）。R-03：新增 SOC 均衡奖励 R_soc_balance = -λ×|SOC-0.5|，鼓励 SOC 保持在 50% 附近。新增 w9/w10/w11 权重。训练管线对齐下游 v2.12 PRD。

> **v2.12 变更说明（对齐部署端 v2.10）：** 观测空间从 58/59 维扩展为 61/62 维（新增 D9 安全覆盖状态 3 字段：safety_override_active, safety_override_p_ref, safety_override_reason_code）。SCENE-01 奖励函数新增 R_safety_override 惩罚项（w8 加权）：safety_override_active=True 时根据触发原因惩罚（voltage_violation=-50/q_exhausted=-30/emergency=-100/generic=-20），激励 AI 学习避免触发覆盖的策略。训练管线对齐下游 v2.10 PRD。

> **v2.11 变更说明（对齐部署端 v2.8）：** SCENE-01 奖励函数重构（R_PQ_coordination + R_smooth）。移除"电压硬惩罚"，引入 P-Q 协同度奖励（Q 有裕度时省电奖励/Q 饱和时正确出手奖励）。弃光场景差异化（v_avg≥1.05 时检查 p_ref 方向，充电消纳=正确/放电=惩罚）。新增下垂系数平滑惩罚 R_smooth（防止 k_droop 震荡）。新增 w7 光滑惩罚权重。训练管线对齐下游 v2.8 PRD。

> **v2.10 变更说明（v2.7 双参数动作空间合并）：** 新增下垂模式（`config.dual_control.enabled=true`），RL 输出 4 维动作（A1: P_ref, A2: k_droop, A3: load_shedding, A4: pv_limit），执行器按 P_output = P_ref + k_droop × ΔV 计算最终功率。NumPy PPO 支持 dual_mode，5 维策略输出（p_ref, k_droop, load_shedding, pv_limit, confidence），buffer 使用 `env.action_space.shape[0]` 截断至 4 维。DualActionValidator 实现 ACT-DUAL-01~05 约束规则（p_ref 斜率限制、k_droop 范围、|p_ref| ≤ |dispatch_p|、pv_limit ≥ 0.1 防逆流）。

> **v2.9 变更说明（v2.7/v2.8/v2.9 合并）：** 动态阻抗扰动（VoltageSimulator 每步对 k_p/k_q 加 ±10% 随机扰动）+ 3/5 次谐波注入（3%/2% 幅值）。通信延迟模拟（mupc_env FIFO 动作缓冲区 1~3 步延迟）。MODE-01 奖励函数新增主动弃光专项奖励（v_avg>=1.05 时，pv_limit 越低奖励越高，解决弃光悖论）。config 新增 CommConfig / voltage_simulator 谐波/阻抗漂移参数。

> **v2.8 变更说明：** 修复 SOC 递推充放电效率因子（battery_charge/discharge_efficiency=0.90）。ONNX 导出包含完整归一化层（_normalize Bake 进模型）。PPO log_prob 统一在 pre-activation 空间计算（修复概率基准分布不匹配）。LSTM 预测输出加 ReLU 非负约束（PV/load>=0）。Rollout bootstrap 边界条件修复。grid2op_env 电池容量 200kWh→100kWh（运行时从配置读取）。VoltageSimulator K_Q 物理含义澄清。

> **v2.7 变更说明：** ONNX 导出 act_dim 2→3（添加 a3 sigmoid pv_limit）。NumPy PPO 3 维动作全面支持 + tanh/sigmoid Jacobian 修正。NumPy PPO 反向传播硬编码 2 层→动态循环（支持任意 --net-arch）。V_DEAD→VOLTAGE_DEADBAND，window=4→DEMAND_WINDOW_STEPS 从配置读取。SMART-DS .npz 缓存机制。观测归一化逐元素 _minmax()→批量 _norm_slice() 向量化。

> **v2.6 变更说明：** 对齐部署端 PRD v2.6。动作空间从 2 维恢复为 3 维（新增 pv_limit 光伏限功率比例，由 RL 主动弃光）。物理参数全面修正：电池最大充放电功率 500kW→50kW，电池容量 200kWh→100kWh，变压器容量 500kVA→200kVA，ACT-03 功率圆上限 500kVA→200kVA，load_shedding 上限 500kW→60kW。SCENE-01 新增 w6 电压变化斜率惩罚 R_slope = w6·|ΔV|。ACT-04（pv_limit ∈ [0,1]）约束规则恢复。
>
> **v2.5 变更说明：** 对齐部署端 PRD v2.5。状态空间从 56/57 维扩展为 58/59 维（新增 D5 气象 2 维 + 修正 D2 光伏/负荷预测偏移）。观测空间索引修正 [24..39] → [25..39]。
>
> **v2.4 变更说明：** 对齐部署端 PRD v2.4。动作空间从 4 维缩减为 2 维（分层控制架构：Q 控制交由实时控制核心闭环，RL 仅输出 P_batt + Load_shedding）。SCENE-01 奖励函数新增电压死区（±5%，越限连续 2 步触发）和 R_ramp 功率变化率惩罚（归一化到 C-rate）。MODE-01 权重表新增 w4（电压质量）、w5（功率变化率）。ACT-02/ACT-04 约束规则移除（由实时控制处理）。
>
> **v2.3 变更说明：**集成 Grid2Op + Pandapower电压仿真替换（2026-06-09）。将 VoltageSimulator 替换为 Grid2Op 物理仿真引擎，三相电压基于真实潮流计算。原有 PRD v2.2 规格（58/59维观测、2维动作、5场景奖励）全部保留，新增电压仿真引擎切换开关 `use_grid2op`。性能目标：每步仿真 ≤ 50ms（lightsim2grid 加速）。
>
> **v2.2 变更说明：** 对齐部署端 PRD v2.5。状态空间从 48/49 维扩展为 56/57 维（新增 D7: q_realtime_margin + season_encoding + time_period_encoding）。SCENE-01 奖励函数新增自适应损耗系数 α(s) ∈ {3.0, 0.2, 1.0}、条件触发电压惩罚（仅当 q_realtime_margin ≤ 10% 且越限 ≥2 步）和弃光电压前置条件（v_avg ≥ 1.05 → R_pv = 0）。观测空间维度更新。
>
> **v2.1 变更说明：** 对齐部署端 PRD v2.4。动作空间从 4 维缩减为 2 维（分层控制架构：Q 控制交由实时控制核心闭环，RL 仅输出 P_batt + Load_shedding）。SCENE-01 奖励函数新增电压死区（±5%，越限连续 2 步触发）和 R_ramp 功率变化率惩罚（归一化到 C-rate）。MODE-01 权重表新增 w4（电压质量）、w5（功率变化率）。ACT-02/ACT-04 约束规则移除（由实时控制处理）。

---

## 1. 产品概述

### 1.1 产品定位

MUPC 强化学习模型训练管线是一个运行在本地 x86 PC 上的 Python 工具链。它负责训练两个核心模型并通过 ONNX 交付给 MUPC AI 引擎（RK3588 NPU 部署）：

1. **LSTM 时序预测模型** — 光伏出力 / 负荷功率时序预测
2. **PPO/SAC 强化学习决策模型** — 3 维动作空间的多目标优化控制

本管线是 MUPC AI 引擎的**模型供给侧**。训练的模型经 ONNX 导出后，由部署端进行 INT8 量化（rknn-toolkit2）并在 RK3588 NPU 上执行推理。

### 1.2 与 MUPC AI 引擎的对齐关系

```
┌──────────────────────────────────────────────────────────────────┐
│  本训练管线 (x86 PC, Python)                                       │
│ │
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
│    → to_input_vector() [58维] → RKNN Runtime → ActionOutput(3维) │
│    → ActionValidator(4条约束) → strategy-engine                  │
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
| 电压仿真 | Grid2Op + Pandapower 三相潮流（可切换简化模型） | 实时数据 |

> **关键原则：** 训练环境的观测空间、动作空间和奖励函数规格完全对齐部署端 PRD v2.6。差异仅在于数据获取方式和电压仿真的实现方式。

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
| 电压仿真 | Grid2Op + Pandapower 三相潮流（可切换 VoltageSimulator） |

### 1.4 核心价值

| 价值 | 说明 | 量化目标 |
|------|------|----------|
| 完整对齐部署规格 | 训练即部署：观测/动作/奖励与 MUPC AI 引擎完全一致 | 模型导出后直接可用，无需额外适配 |
| 多场景单一策略 | 一个 RL 模型覆盖 5 种预设运行场景 (MODE-01~05) | 场景切换无需换模型 |
| 安全优先 | 安全惩罚梯度 + SOC 硬约束 + 动作约束校验 | 过载事件减少 > 90%（相比固定策略） |
| 端到端链路 | 数据加载 → LSTM训练 → RL训练 → ONNX导出，单条命令 | 无中间手动步骤 |
| 精确电压仿真 | Grid2Op + Pandapower 三相潮流计算 | 三相电压误差 ≤ 2%，每步 ≤ 50ms |

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
| F2 | MUPC 全状态环境仿真 | P0 | 58 维观测 + 2 维动作 + 5 种场景奖励的 Gymnasium 环境 |
| F3 | LSTM 时序预测模型训练 | P0 | 光伏/负荷预测，输出 15 分钟预测向量 |
| F4 | 多模式 RL 训练 | P0 | PPO/SAC 训练，多模式单模型，58/59 维输入 |
| F5 | 动作约束校验 | P0 | 3 条约束规则（ACT-01/03/05），环境内 clamp |
| F6 | 模型导出 | P0 | LSTM + RL 策略网络 → ONNX，含 onnxruntime 验证 |
| F7 | 训练监控 | P1 | TensorBoard + CSV 日志，21 字段各自可追踪 |
| F8 | Grid2Op 电压仿真 | P0 | 三相潮流计算（可切换简化 VoltageSimulator） |

---

### 3.1 F1：数据加载与状态合成

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
| voltage_phase_a/b/c | **Grid2Op 潮流计算** 或 VoltageSimulator | D1，v2.3 Grid2Op 新增 |
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

**电压仿真模型（v2.3 新增）**：

训练环境支持两种电压仿真引擎：

1. **Grid2Op + Pandapower**（默认）：基于真实三相潮流计算，三相电压由 Pandapower `runpp_3ph` 计算
2. **VoltageSimulator**（降级）：简化 Q-V 耦合灵敏度系数模型，当 Grid2Op 不可用或用户指定时启用

```
Grid2Op 模式:
  V_phase = f(P_pv, P_load, P_batt, Q_batt)  ← Pandapower runpp_3ph 三相潮流
 精度: 三相独立计算，不平衡度自然呈现
  误差: ≤ 2%（与 Pandapower 标准对比）

VoltageSimulator 模式（降级）:
  V_phase = 1.0 + k_p * (P_pv - P_load + P_batt) / S_base
               - k_q * Q_batt / S_base
               + noise(σ=0.005)
  k_p = 0.05, k_q = 0.03, S_base = 200 kVA
```

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F1-01 | 加载全部 SMART-DS 光伏和负荷数据，打印统计摘要 | `python data_loader.py` 无错误 |
| F1-02 | TOU 电价按时段输出：谷 0.4/平 0.8/峰 1.2/尖峰 1.5 元/kWh | 单元测试 |
| F1-03 | 训练/验证集按时间 8:2 切分（不打乱顺序） | 单元测试 |
| F1-04 | Grid2Op 模式下三相电压在 [0.85, 1.15] p.u. 内 | 单元测试 |
| F1-05 | VoltageSimulator 降级模式下三相电压在 [0.85, 1.15] p.u. 内 | 单元测试 |
| F1-06 | 合成数据覆盖全部 21 个状态字段 | 单元测试 |

---

### 3.2 F2：全状态环境仿真

#### 功能描述

`mupc_env.py` 实现基于 MUPC AI 引擎 PRD v2.5 完整规格的 Gymnasium 环境，支持 Grid2Op 电压仿真引擎。

**观测空间（58 维序列化向量，多模式 59 维）**：

遵循 MUPC AI 引擎设计文档 `to_input_vector()` 布局：

```
索引      内容                      来源类别
[0..9]    D1 实时数据 (10标量)      SMART-DS + 环境仿真（含 q_realtime_margin）
[10..24]  D2 pv_forecast (15维)     LSTM 输出
[25..39]  D2 load_forecast (15维)   LSTM 输出
[41..43]  D3 电价 (3字段)           TOU 合成
[44..46]  D4 需量 (3字段)           合成
[47..48]  D5 气象 (2字段)           SMART-DS
[49]      D6 dispatch_p_set (1维)      合成 (None→0.0)
[50]      D7 q_realtime_margin       实时控制模块计算
[51..56]  D7 season_encoding (6维)   合成（季节 one-hot）
[57]      D7 time_period_encoding    合成（时段 one-hot，白天/夜间）
```

多模式训练时追加 mode_id 为第 59 维。

**动作空间（3 维）**：

| 维度 | 字段名 | 训练值域 | 单位 | 说明 |
|------|--------|----------|------|------|
| 0 | p_batt_set | [-1, 1] → [-50, 50] kW | kW | 电池有功功率设定值（RL 控制） |
| 1 | load_shedding | [0, 1] → [0, 60] kW | kW | 可中断负荷切除量（RL 控制） |
| 2 | pv_limit | [0, 1] → [0, 1] | — | 光伏有功限功率比例（v2.6 新增，主动弃光） |

> **v2.6 分层控制架构：** Q 控制（q_batt_set）由实时电压调节器闭环调节，不经过 RL。pv_limit 由 RL 主动控制（主动弃光），与 Q 调节互补。RL 专注能量管理（P_batt + Load_shedding + Pv_limit），避免 ms 级 Q 控制与 min 级 P 控制的时间尺度冲突。

**下垂模式（4维，v2.7新增）**

当 `config.dual_control.enabled=true` 时启用，RL 输出 4 维动作：

| 维度 | 字段 | 范围 | 说明 |
|------|------|------|------|
| A1 | P_ref | [-50, 50] kW | 有功功率基准点 |
| A2 | k_droop | [-100, 100] kW/V | 电压-有功下垂系数 |
| A3 | load_shedding | [0, 60] kW | 可中断负荷切除量 |
| A4 | pv_limit | [0.1, 1] | 光伏限功率比例（防逆流） |

执行器根据下垂公式计算最终功率：P_output = P_ref + k_droop × ΔV

**配置参数：**
- `dual_control.enabled`: 启用/禁用下垂模式
- `dual_control.k_droop_min/max`: 下垂系数范围 [-100, 100]
- `dual_control.p_ref_ramp_limit_kw`: P_ref 变化率限制 (50.0 kW/步)

**核心物理方程**：

```
P_batt = p_batt_set_norm * 50  (kW)
P_load_eff = P_load - load_shedding  (kW, 切除后有效负荷)
P_pv_eff = P_pv * pv_limit  (光伏限功率后出力，v2.6 新增 pv_limit)

SOC_raw = SOC_t + (-P_batt * dt) / BATTERY_CAPACITY_KWH
SOC_{t+1} = clamp(SOC_raw, 0.10, 0.90)  // SAFETY: 硬约束

grid_power = P_load_eff - P_pv_eff + P_batt
Q_load = P_load_eff * tan(acos(0.90))  (功率因数 0.90)
Q_batt 由实时控制核心闭环调节（电压死区 ±5%）
S_transformer = sqrt(grid_power² + (Q_load - Q_batt)²)
load_rate = S_transformer / TRANSFORMER_KVA (200)

电压仿真（Grid2Op 模式）: voltage_phase_{a,b,c} = Grid2Op runpp_3ph 三相潮流
电压仿真（VoltageSimulator 模式）: voltage_phase_{a,b,c} = f(P, Q) 简化模型
需量: current_demand = max(最近4步 P_load_eff 的滑动平均值, 前值)
```

**环境常量**：

| 参数 | 值 | 常量名 |
|------|-----|--------|
| 变压器容量 | 200 kVA | `TRANSFORMER_KVA` |
| 电池容量 | 100 kWh | `BATTERY_CAPACITY_KWH` |
| 最大充放电功率 | 50 kW (p_batt 范围上限) | `P_BATT_MAX` |
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
| F2-02 | `observation_space.shape = (58,)` | 单元测试 |
| F2-03 | `action_space.shape = (3,)` | 单元测试 |
| F2-04 | SOC 硬约束不可突破 | 单元测试：连续充电 1000 步，验证 SOC ≤ 0.90 |
| F2-05 | info dict 包含全部奖励分量原始值 + SOC + load_rate | 单元测试 |
| F2-06 | Grid2Op 模式下三相电压在 [0.85, 1.15] 内 | 单元测试 |
| F2-07 | VoltageSimulator 降级模式下三相电压在 [0.85, 1.15] 内 | 单元测试 |
| F2-08 | 兼容 gymnasium.Env 和 _gym_stub 双重接口 | 集成测试 |
| F2-09 | terminal_observation 包含完整 59 维观测 | 单元测试 |
| F2-10 | `use_grid2op=True` 时使用 Grid2Op电压仿真 | 集成测试 |
| F2-11 | `use_grid2op=False` 时降级到 VoltageSimulator | 集成测试 |

---

### 3.3 F3：LSTM 时序预测模型

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

### 3.4 F4：多模式 RL 训练

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
| `--use-grid2op` | True | 使用 Grid2Op 电压仿真（False降级到 VoltageSimulator） |
| `--data-source` | `smartds` | 数据源：`smartds` / `china` / `merged` / `unified` |
| `--train-lstm` | False | 独立训练 LSTM（不跑 RL） |
| `--lstm-params` | `hidden_dim=64,num_layers=2,epochs=100,patience=15` | LSTM训练参数 |
| `--export-onnx` | False | 训练结束后导出 ONNX |

**5 种场景与奖励函数**（与 MUPC AI 引擎 PRD 第 6 章完全对齐）：

| 场景 | 命令行 | 优化目标 | 公式 |
|---------|--------|----------|------|
| SCENE-01: 台区季节性负荷模式 (MODE-01) | `MODE-01` | 最大化光伏消纳 + 防止过载 + 电压质量 + 功率变化率 | `R = w1·R_pv_consumption - w2·P_battery_degradation - w3·P_transformer_overload - w4·P_voltage_deviation - w5·R_ramp` |
| SCENE-B1: 工商业模式-自主套利 (MODE-02) | `MODE-02` | 最大化峰谷价差 + 最小化电池损耗 | `R = w1·R_price_spread - w2·P_battery_degradation` |
| SCENE-B2: 工商业模式-需量控制 (MODE-03) | `MODE-03` | 减免需量罚金 | `R = w1·R_demand_penalty_avoidance - w2·P_comfort_loss` |
| SCENE-B3: 工商业模式-虚拟电厂 (MODE-04) | `MODE-04` | 辅助服务收益 + 响应精度 | `R = w1·R_ancillary_service + w2·R_response_accuracy - w3·P_deadline_deviation` |
| SCENE-B5: 工商业模式-极致绿色 (MODE-05) | `MODE-05` | 最大化绿电消纳 + 最小化碳排放 | `R = w1·R_green_consumption + w2·R_carbon_reduction` |

> 各场景奖励公式的完整定义见 MUPC AI 引擎 PRD 第 6.2~6.6 节。本环境在 `mupc_env.py` 中逐项实现。

**多模式训练策略**：
- `--mode all`（默认）：每个 episode 随机选择一种场景，mode_id 编码追加到观测向量（59 维），训练单一模型覆盖全部 5 种场景
- `--mode MODE-01`：单场景训练，观测为 58 维

**PPO 网络结构**：

```
Input(58 or 59) → Linear(128) → ReLU → Linear(128) → ReLU
                       ├── actor:  Linear(2) → Tanh (A1) / Sigmoid (A2)
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
| F4-07 | TensorBoard 中可监控所有奖励分量 + 3 个动作维度的均值 | 手动检查 |
| F4-08 | `--use-grid2op=False` 时降级到 VoltageSimulator | 集成测试 |

---

### 3.5 F5：动作约束校验

#### 功能描述

遵循 MUPC AI 引擎 PRD 第 5.4 节的 4 条约束规则，在环境 `step()` 中对 RL 输出的动作进行校验和 clamp。

> **v2.6 变更：** ACT-04（pv_limit ∈ [0,1]）约束规则恢复。pv_limit 由 RL 主动控制（主动弃光），不再由实时控制核心单独处理。Q 控制（q_batt_set）仍由实时电压调节器闭环调节，不经过 RL。

| 规则 ID | 约束条件 | 训练环境实现 |
|---------|----------|-------------|
| ACT-01 | Δp_batt ≤ 50 kW/步 | 计算变化率，超标则 clamp |
| ACT-03 | √(p_batt²+q_batt²) ≤ 200 kVA（q 取实时值） | 超标则等比例缩放 P |
| ACT-04 | pv_limit ∈ [0, 1] | 超出边界则 clamp |
| ACT-05 | dispatch_p 有效时 \|p_batt\| ≤ \|dispatch_p\| | 有调度时 clamp |

> 部署端 ActionValidator 在 Rust 端同样实现这 4 条规则（ACT-02 由实时控制处理）。训练时在环境内执行校验，让 RL agent 在与部署相同的约束下学习。

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F5-01 | 4 条约束均实现 | 单元测试：逐条触发违规并验证 clamp 结果 |
| F5-02 | 约束违反时 info dict 中记录 `constraint_violated=True` | 单元测试 |
| F5-03 | 约束校验耗时 < 0.5ms | 性能测试 |

---

### 3.6 F6：模型导出

#### 功能描述

`export_onnx.py` 导出两种模型为 ONNX：

1. **LSTM 预测模型**：`lstm_forecast.onnx`，输入 (1, 4, 6) (batch, seq_len, features)，输出 (1, 30)
2. **RL 策略网络**：`rl_policy.onnx`，输入 (1, 58) 或 (1, 59)，输出 (1, 3)

导出流程：
1. 加载 PyTorch checkpoint / NumPy PPO weights
2. 构建等效 PyTorch 模型
3. `torch.onnx.export()`
4. `onnx.checker.check_model()` 验证
5. onnxruntime 推理验证（误差 < 1e-5）

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F6-01 | RL 策略 ONNX 输入 (1, 58) 输出 (1, 3) | 检查 ONNX spec |
| F6-02 | LSTM ONNX 输入 (1, 4, 6) 输出 (1, 30) | 检查 ONNX spec |
| F6-03 | ONNX 推理与 PyTorch 推理误差 < 1e-5 | 单元测试 |
| F6-04 | 导出文件名含时间戳 | 检查文件名 |
| F6-05 | 无 SB3 checkpoints 时自动从 npz 导出 | 测试 npz 路径 |

---

### 3.7 F7：训练监控

| 指标 | 输出位置 |
|------|----------|
| episode 奖励（总和 + 各分量） | TensorBoard + CSV |
| 3 个动作维度的均值/最大值 | TensorBoard |
| SOC 均值、负载率均值、过载次数 | TensorBoard + 控制台 |
| 当前场景 ID | CSV |
| 训练 loss（actor/critic） | TensorBoard |
| 学习率 | TensorBoard |

控制台每隔 10000 步打印摘要。

---

### 3.8 F8：Grid2Op 电压仿真引擎

#### 功能描述

Grid2Op + Pandapower 电压仿真引擎作为 `VoltageSimulator` 的替代方案，提供更精确的三相电压计算。

**核心组件**：

| 组件 | 文件 | 职责 |
|------|------|------|
| `NumpyChronics` | `grid2op_env/numpy_chronics.py` | 将 SmartDSLoader 的 data dict 转换为 Grid2Op 三相格式 |
| `Grid2OpPowerFlow` | `grid2op_env/power_flow.py` | Grid2Op 引擎封装，提供同步 SOC 和获取三相电压的接口 |
| `create_mupc_network` | `grid2op_env/network.py` | Pandapower 网络拓扑（农网台区 3 总线模型） |

**技术规格**：

| 项目 | 规格 |
|------|------|
| 潮流计算 | Pandapower `runpp_3ph` 三相潮流 |
| 后端优先级 | lightsim2grid（C++ 加速）> PandaPowerBackend（Python） |
| 网络拓扑 | 3 总线：高压电网 → 配电变压器(200kVA) → 低压母线 → 末端节点 |
| 元件 | 变压器、线路（LGJ-70, 1.5km）、居民负荷、农业冲击负荷、光伏(150kW)、储能(100kWh) |
| SOC 同步 | 双向同步：step入口 Grid2Op→MupcEnv，step 出口 MupcEnv→Grid2Op |
| 不收敛处理 | 回退到上一时刻安全电压，`has_illegal=True` 标记 |

**电压仿真切换**：

```bash
# Grid2Op 模式（默认）
python train.py --mode MODE-01 --total-timesteps 1000000

# VoltageSimulator 降级模式
python train.py --mode MODE-01 --total-timesteps 1000000 --no-grid2op
```

**性能目标**：

| 指标 | 要求 |
|------|------|
| 每步仿真耗时 | ≤ 50ms（lightsim2grid 加速后端） |
| 三相电压误差 | ≤ 2%（与 Pandapower `runpp_3ph` 标准对比） |
| 训练吞吐量下降 | ≤ 20%（相比 VoltageSimulator） |

#### 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| F8-01 | Grid2Op 初始化成功，三相电压输出正常 | `python mupc_env.py` |
| F8-02 | `has_illegal=True` 时电压回退到上一时刻安全值 | 单元测试 |
| F8-03 | SOC 双向同步误差 ≤ 0.1% | 连续 100 步测试 |
| F8-04 | Grid2Op 不可用时自动降级到 VoltageSimulator | 卸载 grid2op 后测试 |
| F8-05 | lightsim2grid 不可用时降级到 PandaPowerBackend | 卸载 lightsim2grid 后测试 |

---

## 4. 非功能性需求

### 4.1 性能

| 指标 | 要求 |
|------|------|
| 环境 step() 耗时（不含推理，VoltageSimulator 模式） | < 2ms |
| 环境 step() 耗时（Grid2Op 模式，lightsim2grid 后端） | ≤ 50ms |
| 训练吞吐（SB3 PPO, CPU, VoltageSimulator 模式） | > 500 steps/s |
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
| 第三方包 | SB3, gymnasium, torch, onnx, onnxruntime, numpy, grid2op, pandapower |
| 降级方案 | SB3 → NumPy PPO; Gymnasium → _gym_stub; Grid2Op → VoltageSimulator; lightsim2grid → PandaPowerBackend |
| Checkpoint 兼容 | 替换前后 checkpoint 可互相加载 |

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
| Grid2Op 不可用 | 自动降级到 VoltageSimulator（`use_grid2op=False`） |
| lightsim2grid 不可用 | 降级到 PandaPowerBackend |
| 潮流计算不收敛 | 回退到上一时刻安全电压，`has_illegal=True` |
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
| 电压超出 [0.85, 1.15] | Grid2Op 模式：潮流不收敛标记；VoltageSimulator 模式：clamp 到边界值 |

### 5.4 导出异常

| 场景 | 处理 |
|------|------|
| checkpoint 不存在 | 打印 ERROR 退出 |
| ONNX checker 不通过 | 打印详细错误，不输出文件 |
| onnxruntime 不可用 | 仅 checker 验证，跳过推理比对 |

---

## 6. 风险评估

### 6.1 Grid2Op 相关风险

| 风险编号 | 风险描述 | 影响等级 | 应对策略 |
|----------|----------|----------|----------|
| R-01 | Grid2Op 与 Gymnasium 接口冲突（环境基类不兼容） | 高 | 采用组合模式：MupcEnv 内部持有 Grid2Op 实例 |
| R-02 | 潮流计算耗时过高，导致训练速度下降 5~10 倍 | 高 | 优先使用 lightsim2grid C++ 后端；必要时降级为单相潮流 |
| R-03 | 三相潮流收敛失败（孤岛/过载导致潮流不收敛） | 中 | 添加潮流收敛检测，不收敛时回退到上一时刻电压值 |
| R-04 | Grid2Op Chronics 数据格式与现有 `data` dict 不兼容 | 高 | 编写 `NumpyChronics` 自定义类，将 `data` dict 转换为 Grid2Op 三相格式 |
| R-05 | 动作空间映射错误（2维动作未正确映射到 Grid2Op storage） | 高 | 添加单元测试，验证 `action → storage_p` 映射一致性 |
| R-06 | SOC 状态在 Grid2Op 与 MupcEnv 之间不同步 | 高 | 在 `step()` 开始时同步 Grid2Op storage SOC 到 `self._soc`，结束时反向同步 |
| R-07 | 现有奖励函数依赖的观测字段在 Grid2Op 中不可用 | 中 | Grid2Op 仅提供三相电压和变压器负载率，其他字段继续从 data dict 读取 |
| R-08 | 依赖库版本冲突（pandapower vs grid2op） | 低 | 使用虚拟环境隔离；自动降级到 VoltageSimulator |

### 6.2 一般训练风险

| 风险编号 | 风险描述 | 影响等级 | 应对策略 |
|----------|----------|----------|----------|
| R-09 | LSTM训练收敛失败 | 中 | 使用现有 checkpoint 或 Oracle 后备 |
| R-10 | RL 训练不收敛 | 中 | 检查奖励权重、超参数、网络结构 |
| R-11 | ONNX 导出失败 | 低 | 检查模型结构和权重完整性 |

---

## 7. 文件结构

```
MUPC-AI2/
├── data_loader.py              # F1: 数据加载 + 状态合成
├── mupc_env.py # F2: Gymnasium 环境 (58/59维, 2维动作, Grid2Op集成)
├── lstm_model.py             # F3: LSTM 训练
├── train.py                  # F4: RL 训练主入口
├── action_validator.py       # F5: 动作约束校验
├── export_onnx.py # F6: ONNX 导出
├── _ppo_core.py              # 纯 NumPy PPO 后备
├── _gym_stub.py # Gymnasium 最小替代
├── grid2op_env/             # F8: Grid2Op 电压仿真引擎
│   ├── __init__.py
│   ├── numpy_chronics.py     # NumpyChronics: data dict → Grid2Op 格式
│   ├── power_flow.py # Grid2OpPowerFlow: Grid2Op 引擎封装
│   ├── network.py           # create_mupc_network(): Pandapower 拓扑
│   └── backend.py           # Backend 选择 (lightsim vs pandapower)
├── data/
│   ├── download_smart_ds.py # 数据集下载
│   └── smart_ds/ # SMART-DS 数据
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

## v2.6 修订记录

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 动作空间 2 维→3 维（恢复 pv_limit） | 1.1/3.2/3.5/3.6/3.7/6 | RL 主动控制光伏限功率比例（主动弃光） |
| 2 | 物理参数全面修正 | 3.2/环境常量表 | 电池最大充放电功率 500→50kW，电池容量 200→100kWh，变压器容量 500→200kVA |
| 3 | ACT-03 功率圆上限修正 | 3.5/ACT-03 | S_MAX: 500kVA→200kVA（匹配变压器容量） |
| 4 | ACT-04 约束规则恢复 | 3.5 | pv_limit ∈ [0,1] 约束规则重新加入 |
| 5 | SCENE-01 新增 w6 电压斜率惩罚 | 3.5 | R_slope = w6·|ΔV|，迫使 AI 平滑调节 |
| 6 | 更新部署端 PRD 版本引用 | 文档头部 | v2.5 → v2.6 |
| 7 | 动作值域修正 | 3.2 | p_batt [-500,500]→[-50,50]，load_shedding [0,500]→[0,60] |

**修订依据：** 对齐 MUPC AI 引擎 PRD v2.6，物理参数与实际设备规格一致

## v2.5 修订记录

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 状态空间 56/57 维→58/59 维 | 3.2 | 新增 D5 气象 2 维（辐照/温度） |
| 2 | 观测空间索引修正 | 3.2 | D2 光伏预测 [10..24]，负荷预测 [25..39]（原 [24..39] 偏移1位） |
| 3 | 更新部署端 PRD 版本引用 | 文档头部 | v2.4 → v2.5 |

**修订依据：** MUPC AI 引擎 PRD v2.5 状态空间扩展

## v2.4 修订记录

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 动作空间 4 维→2 维 | 1.1/3.2/3.5/3.6/3.7/6 | 分层控制架构：Q 控制交由实时控制核心闭环，RL 仅输出 P_batt + Load_shedding |
| 2 | SCENE-01 奖励函数更新 | 3.5 | 新增 w4（电压质量惩罚）、w5（功率变化率惩罚 R_ramp）、电压死区（±5%，越限连续 2 步触发） |
| 3 | ACT-02/ACT-04 约束移除 | 3.6 | Q 变化率和光伏限功率由实时控制处理，训练环境移除对应约束 |
| 4 | 更新部署端 PRD 版本引用 | 文档头部 | v2.3 → v2.4 |

**修订依据：** 对齐部署端 PRD v2.4 分层控制架构

## v2.3 修订记录

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 集成 Grid2Op 电压仿真替换 | 全文 | 将 VoltageSimulator 替换为 Grid2Op + Pandapower 三相潮流计算 |
| 2 | 新增 F8 功能（Grid2Op 电压仿真引擎） | 3.8 | 新增 Grid2Op 核心组件和技术规格 |
| 3 | 新增 R-01~R-08 风险评估 | 6.1 | Grid2Op相关的8 个风险及应对策略 |
| 4 | 新增 `--use-grid2op` / `--no-grid2op` 参数 | 3.4 | 电压仿真引擎切换开关 |
| 5 | 新增 F8 验收标准 | 3.8 | Grid2Op 初始化、SOC 同步、不收敛处理等 |
| 6 | 更新文件结构 | 7 | 新增 grid2op_env/ 目录 |
| 7 | 更新版本引用 | 文档头部 | v2.2 → v2.3 |

**修订依据：** Grid2Op + Pandapower 电压仿真替换 PRD（2026-06-09）已通过 code review，集成到训练管线 PRD

## v2.2 修订记录

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 状态空间 48/49 维→56/57 维 | 1.1/3.3/3.5 | 新增 D7 字段: q_realtime_margin (1维) + season_encoding (6维) + time_period_encoding |
| 2 | SCENE-01 奖励函数重写 | 3.5 | 新增 α(s) 自适应损耗系数、条件触发电压惩罚、弃光电压前置条件 |
| 3 | 更新部署端 PRD 版本引用 | 文档头部 | v2.4 → v2.5 |
| 4 | 观测空间维度更新 | 3.3 | 单模式 56 维，多模式 57 维 |

**修订依据：** MUPC AI 引擎 PRD v2.5 状态空间扩展 + SCENE-01 奖励函数重写

## v2.1 修订记录

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 动作空间 4 维→2 维 | 1.1/3.3/3.5/3.6/3.7/6 | 分层控制架构：Q 控制交由实时控制核心闭环，RL 仅输出 P_batt + Load_shedding |
| 2 | SCENE-01 奖励函数更新 | 3.5 | 新增 w4（电压质量惩罚）、w5（功率变化率惩罚 R_ramp）、电压死区（±5%，越限连续 2 步触发） |
| 3 | ACT-02/ACT-04 约束移除 | 3.6 | Q 变化率和光伏限功率由实时控制处理，训练环境移除对应约束 |
| 4 | 更新部署端 PRD 版本引用 | 文档头部 | v2.2 → v2.4 |
| 5 | SceneWeights 表更新 | 隐含 | MODE-01 权重数量从 3 增至 5 |

**修订依据：** MUPC AI 引擎 PRD v2.4 分层控制架构 + 电压死区 + 变化率惩罚