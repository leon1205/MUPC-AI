# MUPC RL 训练模型在 Pandapower + Grid2Op 仿真环境的完整运行流程

> 本文档从零基础视角出发，详细介绍一个 RL 智能体如何在基于 Pandapower + Grid2Op 构建的农网台区仿真环境中完成单步训练循环的完整过程。
>
> 适用读者：电力系统 / 强化学习 / 边缘部署的初学者

---

## 目录

1. [全局背景：为什么要这套仿真系统](#1-全局背景为什么要这套仿真系统)
2. [三大工具的分工](#2-三大工具的分工)
3. [网络拓扑：仿真器眼中的"村子"长什么样](#3-网络拓扑仿真器眼中的村子长什么样)
4. [单步训练的完整旅程（核心）](#4-单步训练的完整旅程核心)
5. [关键概念解释](#5-关键概念解释)
6. [完整 step() 流程图](#6-完整-step-流程图)
7. [一个具体数字走一遍：低电压场景](#7-一个具体数字走一遍低电压场景)
8. [完整训练循环的规模化运行](#8-完整训练循环的规模化运行)
9. [为什么这套设计是合理的](#9-为什么这套设计是合理的)
10. [常见问题与进阶主题](#10-常见问题与进阶主题)
11. [v3.0 核心改动逻辑与版本对比](#11-v30-核心改动逻辑与版本对比)

---

## 1. 全局背景：为什么要这套仿真系统？

### 1.1 问题

我们要训练一个"AI 调度员"来控制农村电网的变压器。它需要决定"什么时候充电、什么时候放电、要不要切负荷"等。

- **真实系统**：RK3588 上的真实变压器 + 光伏 + 电池（200kVA 变压器，100kWh 电池，150kW 光伏，60kW 负荷）
- **不能做的事**：你不可能在真实硬件上让 AI 试错 100 万次，电池会爆炸，变压器会过载
- **需要做的事**：在"虚拟电网"里训练 AI，让它零风险地学习

### 1.2 解决方案

构建一个**与真实硬件行为高度一致的数字孪生**：

```
真实硬件 (RK3588)               数字孪生 (x86 PC)
━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━
真实变压器 + 光伏 + 电池    ←→   Pandapower 建模的"虚拟电网"
真实电压/电流/功率          ←→   Grid2Op 计算的"虚拟电压/电流/功率"
AI 网络模型 (ONNX)         ←→   AI 网络模型 (PyTorch, 训练中)
```

**核心问题**：AI 给出一个动作（2 个数），仿真器如何把这些数变成"电压/电流/功率"的真实物理变化？

这就是 Pandapower + Grid2Op 要回答的问题。

---

## 2. 三大工具的分工

训练管线由 **AI 控制器 (RL Policy)** + **MupcEnv 编排层** + **Grid2Op 时间协调** + **Pandapower 物理计算** 四个角色组成。下面按 `step()` 的真实执行时序展示分工与数据流。

### 2.0 单步训练的完整时序图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          MupcEnv.reset()                              │
│  • 随机初始化 SOC / 电压 / 需量 (避开边界)                              │
│  • 季节时段 one-hot 编码                                              │
│  • _make_env_state() → EnvState 数据快照                              │
│  • LSTM.predict(step_idx) → forecast (30 维 D2 预测) ← 详见 4.1.5     │
│  • build_observation() → 78 维 (单模式) / 79 维 (多模式) ndarray      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ 78 维 obs
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│            AI 控制器 (RL Policy, PyTorch/ONNX)                       │
│     obs → forward() → 2 维动作 ∈ [-1, 1]                             │
│     (训练时 PPO 选动作, 推理时 ONNX Runtime 前向)                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ action = [p_ref_norm, k_droop_norm]
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│  MupcEnv.step(action) — 编排层 (mupc_env/core.py)                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ ① 模拟 q_realtime_margin (训练管线无真实实时模块, 见注 A)        │  │
│  │    用上一步末端电压推算本步 Q-V 闭环会输出的 q_batt,              │  │
│  │    反算裕度:                                                     │  │
│  │      q_batt = -K_Q_V × (V_prev - 1.0)  (clamp ±300 kVar)       │  │
│  │      q_realtime_margin = 1 - |q_batt| / 300                    │  │
│  │    此值在 step 开头算, AI 决策后即可在 78 维 obs 的 D7[48] 看到  │  │
│  │ ② 动作校验 ACT-01~05 (v2.17)                                    │  │
│  │    - ACT-01 |Δp_ref| ≤ 50 kW/步                                 │  │
│  │    - ACT-02 |Δk_droop| ≤ 30 kW/V/步 (v2.17: 10→30)              │  │
│  │    - ACT-03 √(p_ref²+k_droop²) ≤ 200 kVA (S-circle)            │  │
│  │    - ACT-04 k_droop ∈ [-100, 100] kW/V                          │  │
│  │    - ACT-05 |p_ref| ≤ |dispatch_p|                              │  │
│  │ ③ 动作反归一化                                                   │  │
│  │    p_ref    = a[0] × 50.0        (kW)                           │  │
│  │    k_droop  = a[1] × 100.0       (kW/V, v2.17)                  │  │
│  │ ④ 计算有效负荷/光伏 (load_shed=0, pv_limit=1.0, 已下沉)          │  │
│  │ ⑤ SOC 更新 (硬约束 [10%, 90%])                                   │  │
│  │ ⑥ 计算 grid_power / load_rate                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                             ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ ⑦ 物理仿真 (可选 Grid2Op / 默认 VoltageSimulator 降级)         │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ 物理仿真结果 (va, vb, vc)
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Grid2Op  (Python 库, 电网仿真框架) — 仅在使用 --use_grid2op 时启用 │
│  • NumpyChronics.load_next() 推进时间, 注入三相负荷/光伏矩阵          │
│  • 把 RL 动作 (p_batt, q_batt, k_droop) 写回 net.storage / net.load  │
│  • 调度 Pandapower 跑潮流                                            │
│  • 返回末端节点电压, 模拟三相不平衡 ±0.3%                              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ 调用 Pandapower.runpp()
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Pandapower  (Python 库, 电气建模工具)                               │
│  • 定义网络拓扑 (10kV→0.4kV 变压器 / 1.5km 架空线 / 末端负荷+光伏)     │
│  • Newton-Raphson 潮流计算 (P-U-Q-θ)                                  │
│  • 返回每个节点电压 U, 支路功率 P                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ res_bus[2].vm_pu
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│  MupcEnv.step() 续 — 奖励计算                                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ ⑧ 更新内部状态 (SOC/电压/grid/load_rate)                         │  │
│  │ ⑨ _make_reward_dict()  →  compute_reward()                    │  │
│  │    SCENE-01 奖励 = w1·R_pv + w2·R_deg + w3·R_overload          │  │
│  │               + w4·R_pq  + w5·R_ramp + w6·R_vslope              │  │
│  │               + w7·R_smooth + w8·R_safety + w9·R_readiness     │  │
│  │               + R_shaping + R_soc_balance + R_state_improve     │  │
│  │    (9 项归一化 w1~w9 + 3 项未归一化附加)                          │  │
│  │ ⑩ Welford 在线归一化 (v2.18, count≥100 启用)                    │  │
│  │ ⑪ 推进 step_idx, LSTM.predict(new_step_idx) → D2 预测           │  │
│  │    _make_env_state() + build_observation() → 下一个 78 维 obs   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ (new_obs, reward, terminated, truncated, info)
                             ↓
                       返回给 trainer
```

### 2.1 四个角色的职责边界

| 角色 | 职责 | 不做什么 |
|------|------|----------|
| **AI Policy (PyTorch/ONNX)** | 接收 obs, 输出 2 维动作 | 不知道任何物理, 不做时间管理 |
| **MupcEnv (核心编排)** | 构建 78 维 obs, 校验+反归一化动作, 调用物理仿真, 计算奖励, 返回新 obs | 不做潮流计算, 不管具体网络拓扑 |
| **Grid2Op (时间协调)** | episode/step 状态机, NumpyChronics 三相数据注入, 调度 Backend | 不做具体物理计算, 不管 AI 动作校验 |
| **Pandapower (物理计算)** | 定义网络, 跑 Newton-Raphson 潮流, 返回节点电压 | 不管时间, 不管 AI, 不管奖励 |

> **关键时序**: MupcEnv 先构建 78 维 obs → 调度 AI 输出 2 维动作 → MupcEnv 完整校验 (ACT-01~05) + 反归一化 → 调度 Grid2Op/Pandapower 物理仿真 → 根据仿真结果计算奖励 → 构建新 obs。AI 不感知物理仿真细节, Pandapower 不感知时间与 AI 动作。

#### 注 A: q_realtime_margin 的"模拟"与"真实来源"

`q_realtime_margin` (D7[48]) 在**训练**与**部署**两个环境中的来源不同:

| 环境 | 来源 | 物理路径 |
|------|------|----------|
| **训练 (本地 Python)** | MupcEnv **模拟**生成 (step 开头 ① 步) | 用上一步末端电压推算 Q-V 闭环会输出的 q_batt → 反算裕度 `1 - \|q_batt\|/300` |
| **部署 (下游 RK3588)** | MUPC **实时控制模块** 注入 | intercore DataUploadPayload 帧 → FusedSystemState.q_realtime_margin → 输入向量 D7[48] |

> **为什么训练时要模拟**: 训练管线在 x86 PC 上, 没有真实硬件实时控制模块闭环, 也无法真实采样互感器数据. 训练闭环 (AI → 物理仿真 → 奖励 → AI) 需要这个值进入 78 维 obs 才能工作, 因此**用相同的 Q-V 闭环公式模拟**, 让 AI 看到与部署时**结构等价**的状态. 真实部署时此值由实时控制模块直接给出, 训练管线的模拟代码**不参与**推理路径.

> **训练时的轻微错位**: 本地 step() 在开头用**上一步结束时的电压**算 q_realtime_margin, 而本步骤的 Q-V 闭环 q_batt 是为下一步电压仿真用的. 这是工程简化 (训练管线没有"实时控制模块"独立线程), 真实部署时 q_realtime_margin 与 q_batt 由实时控制模块**同时**计算, 不存在错位.

### 2.2 关键概念：潮流计算 (Power Flow)

**潮流 = 已知电源/负荷，求解稳态电压**。这是非线性方程组的数值求解。

对于每个节点 (bus)：
- 已知: P_load (有功负荷), Q_load (无功负荷)
- 未知: V (电压幅值, p.u.), θ (电压相角, 弧度)
- 公式:

$$
P_i = V_i \sum_{j} V_j (G_{ij} \cos\theta_{ij} + B_{ij} \sin\theta_{ij})
$$

$$
Q_i = V_i \sum_{j} V_j (G_{ij} \sin\theta_{ij} - B_{ij} \cos\theta_{ij})
$$

求解: 迭代线性化 (Newton-Raphson)，直到功率不平衡 < 1e-6 MVA

### 2.3 数据流接口对齐表 (v2.17)

| 接口 | 起点 | MupcEnv 转换 | 终点 |
|------|------|--------------|------|
| 观测 obs | EnvState 数据快照 | build_observation() → 78 维 (D1-D10) + normalize_obs() | AI Policy |
| 动作 action | AI Policy 输出 ∈ [-1,1] | 反归一化: p_ref = a[0] × 50 (kW), k_droop = a[1] × 100 (kW/V) + ACT-01~05 校验 | net.storage[0].p_mw, k_droop |
| 电压 | net.res_bus[2].vm_pu | 三相不平衡 ±0.3% (va, vb, vc) | EnvState (D1[6..8]) |
| 奖励 | _make_reward_dict() 状态 | 9 项归一化 + 3 项未归一化 + Welford 在线归一化 | trainer (PPO) |

---

## 3. 网络拓扑：仿真器眼中的"村子"长什么样

文件 `grid2op_env/network.py` 定义了一个典型的农网台区。

### 3.1 拓扑结构图

```
       外部电网 (10kV 高压)
              │
        [10kV → 0.4kV 变压器]   ← 200kVA 容量
              │
        0.4kV 低压母线
              │
      [1.5km 长架空线 (NAYY 4x50 SE)]
              │
        末端节点 (End_Node_Bus)
        ┌─────┼─────┐
        │     │     │
     居民  农业  屋顶光伏   ← 输入: 负荷 + 光伏数据
     负荷  冲击  150kW
     ~60kW 负荷
              │
         储能电池 100kWh/50kW  ← RL 控制目标!
```

### 3.2 关键硬件参数

| 参数 | 值 | 物理意义 |
|------|-----|----------|
| TRANSFORMER_KVA | 200 kVA | 变压器额定容量 |
| BATTERY_CAPACITY_KWH | 100 kWh | 电池容量 |
| P_BATT_MAX_KW | ±50 kW | 电池最大充放电功率 |
| Q_BATT_MAX_KVAR | 300 kVar | 电池最大无功输出 |
| PV_ARRAY_KW | 150 kW | 光伏装机容量 |
| LOAD_PEAK_KW | 60 kW | 负荷峰值 |
| SOC_MIN / MAX | 10% / 90% | 电池 SOC 硬约束 |
| V_BASE | 0.4 kV | 低压侧额定电压 |

**重要**：AI 不能让电池放超过 50kW 的功率，这是硬约束。

---

## 4. 单步训练的完整旅程（核心）

下面以"AI 给出一个 2 维动作"开始，逐步追踪到"返回新观测"的全过程。

### 4.1 第 1 步：AI 看到的世界 (观测)

```python
env = MupcEnv(data, mode="MODE-01", use_grid2op=True)
obs, info = env.reset()
# obs 是 78 维向量 (v2.14 单模式), 已经归一化到 [-10, 10] 范围内
# 示例值:
# obs[0]   = SOC = 0.567           (57% 充电状态)
# obs[1]   = 光伏 = 0.398          (归一化后)
# obs[2]   = 负荷 = 0.346          (归一化后)
# obs[6]   = A相电压 = 0.596       (归一化后, 原始 ~0.98 p.u.)
# obs[46]  = 温度 = 0.547          (归一化后, 原始 ~16°C)
# ... 等等
```

#### 完整 78 维观测向量 (v2.14，对齐下游 AI 引擎 PRD)

| 索引 | 数据组 | 字段名 | 含义 | 数据来源 | 归一化 |
|------|--------|--------|------|----------|--------|
| 0 | **D1 实时** | `battery_soc` | 电池荷电状态 | 环境内部状态 (SOCOne) | Identity [0,1] |
| 1 | D1 | `pv_power` | 光伏有功出力 (kW) | SMART-DS/中国合成数据 | MinMax [0,150] |
| 2 | D1 | `load_power` | 负荷有功功率 (kW) | SMART-DS/中国合成数据 | MinMax [0,60] |
| 3 | D1 | `grid_power` | 电网交换有功 (kW) | MupcEnv 计算: `P_load_eff-P_pv_eff+P_batt` | MinMax [-200,200] |
| 4 | D1 | `transformer_load` | 变压器负载率 (p.u.) | MupcEnv 计算: `S/200kVA` | Identity [0,~2] |
| 5 | D1 | `battery_power_prev` | 上一周期电池有功 (kW) | 上一步动作 p_ref 的反归一化值 | MinMax [-50,50] |
| 6 | D1 | `voltage_phase_a` | A 相电压 (p.u.) | **Grid2Op**: `pp.runpp()` → end_bus vm_pu | MinMax [0.85,1.15] |
| 7 | D1 | `voltage_phase_b` | B 相电压 (p.u.) | **Grid2Op**: va×(1-0.003) 人工不平衡 | MinMax [0.85,1.15] |
| 8 | D1 | `voltage_phase_c` | C 相电压 (p.u.) | **Grid2Op**: va×(1+0.003) 人工不平衡 | MinMax [0.85,1.15] |
| 9~23 | **D2 预测** | `pv_forecast_15min` | 光伏 15 分钟预测 (15 维) | **LSTM 模型** / Oracle (真实值+噪声) | MinMax [0,150] |
| 24~38 | D2 | `load_forecast_15min` | 负荷 15 分钟预测 (15 维) | **LSTM 模型** / Oracle (真实值+噪声) | MinMax [0,60] |
| 39 | **D3 电价** | `current_electricity_price` | 当前电价 (元/kWh) | TOU 电价生成器 (谷0.4/平0.8/峰1.2/尖峰1.5) | MinMax [0,1.5] |
| 40 | D3 | `next_period_price` | 下时段电价 (元/kWh) | TOU 电价生成器查表 | MinMax [0,1.5] |
| 41 | D3 | `price_tariff_id` | 时段枚举编码 | TOU 电价生成器 {0=谷,1=平,2=峰,3=尖峰} | MinMax [0,3] |
| 42 | **D4 需量** | `current_demand` | 当前需量 (kW) | 最近 4 步负荷滑动均值 (≥合同需量×0.3) | MinMax [0,500] |
| 43 | D4 | `contract_demand` | 合同需量 (kW) | 配置文件常量 (默认 200kW, v2.17 对齐下游) | MinMax [0,500] |
| 44 | D4 | `peak_demand_this_month` | 本月峰值需量 (kW) | 滑动窗口跟踪累积最大值 | MinMax [0,500] |
| 45 | **D5 气象** | `solar_irradiance` | 太阳辐照度 (W/m²) | SMART-DS/中国合成数据 | MinMax [0,1500] |
| 46 | D5 | `temperature` | 环境温度 (°C) | SMART-DS/中国合成数据 | MinMax [-20,60] |
| 47 | **D6 调度** | `dispatch_p_set` | 调度有功指令 (kW) | 合成: 非VPP=0，VPP=随机 | MinMax [-200,200] |
| 48 | **D7** | `q_realtime_margin` | 实时模块无功裕度 | **训练**: MupcEnv 模拟 `1.0 - abs(q_batt)/300` / **部署**: 实时控制模块注入 | Identity [0,1] |
| 49 | **D8 季节** | `season_irrigation` (灌溉季) | 3~4 月 one-hot | MupcEnv: 小时/月份推算 | One-hot |
| 50 | D8 | `season_tea` (炒茶季) | 5 月 one-hot | 同上 | One-hot |
| 51 | D8 | `season_ac` (空调季) | 6~8 月 one-hot | 同上 | One-hot |
| 52 | D8 | `season_normal` (常规季) | 其他月份 one-hot | 同上 | One-hot |
| 53 | D8 | `season_reserved_1` | 保留位 1 | 始终 0 | One-hot |
| 54 | D8 | `season_reserved_2` | 保留位 2 | 始终 0 | One-hot |
| 55 | **D8 时段** | `time_day` (白天) | 06:00~18:00 one-hot | MupcEnv: 小时判断 | One-hot |
| 56 | D8 | `time_night` (夜间) | 18:00~06:00 one-hot | 同上 | One-hot |
| 57 | **D9 安全** | `safety_override_active` | 安全覆盖是否激活 | RobustnessManager (训练中常为False) | Bool→float |
| 58 | D9 | `safety_override_p_ref` | 安全覆盖 P 基准 (kW) | 同上 (训练中常为0) | MinMax [-50,50] |
| 59 | D9 | `override_consecutive` | 连续触发次数 | 同上 (训练中常为0) | Identity |
| 60 | D9 | `override_ratio` | 滑动窗口覆盖比例 | 同上 (训练中常为0) | Identity [0,1] |
| 61~75 | **D10 概率** | `load_forecast_quantiles` | 分位数负荷预测 P3.3~P96.7 (15 维, linspace(0.85, 1.27)) | **LSTM 模型** (D10头) / 合成 | MinMax [0,60] |
| 76 | D10 | `shock_load_probability` | 冲击负荷发生概率 | **LSTM 模型** (D10头) / P90-P50差值法 | Identity [0,1] |
| 77 | D10 | `base_load` | 基荷 P50 分位数 (kW) | **LSTM 模型** (D10头) / 合成 | MinMax [0,500] |
| 78* | (可选) | `mode_id` | 多模式场景 ID | MupcEnv: `MODE_ID_MAP[current_mode]` | Identity [0,1] |

> **注：** 索引 78 仅在多模式训练 (`--mode all`) 时追加，单模式为 78 维。

#### 各维度数据来源流程

```
SMART-DS/中国合成数据 (CSV)
  ├── pv_power, load_power, ghi, temperature  ──→  D1/D5 (直接填充)
  ├── pv_power[t+1..t+15], load_power[t+1..t+15] ──→ LSTM训练 target
  │
LSTM 模型 (lstm_model.py)
  ├── predict(step_idx) → pv_forecast(15) + load_forecast(15)  ──→  D2
  └── predict_numpy() → D10 quantiles + shock_prob + base_load ──→  D10
      (D10 仅在 with_d10=True 时输出, 训练LSTM后可用)
  │
MupcEnv 内部状态 (core.py)
  ├── _soc (电池递推) ──→  D1[0]
  ├── _grid_power (计算) ──→  D1[3]
  ├── _load_rate (计算) ──→  D1[4]
  ├── q_batt (Q-V 闭环模拟, 见 2.1 注 A) → q_realtime_margin ──→  D7[48]
  ├── season_encoding (月份推算) ──→  D8[49..54]
  └── time_period_encoding (小时判断) ──→  D8[55..56]
  │
Grid2Op + Pandapower (grid2op_env/)
  └── pp.runpp(net) → net.res_bus[2].vm_pu
       └── ±voltage_imbalance → va, vb, vc ──→  D1[6..8]
  │
TOU 合成 (data_loader.py: _generate_tou_prices)
  ├── current_price, next_price, tariff_id ──→  D3[39..41]
  │
需量合成 (data_loader.py: _generate_demand)
  ├── current_demand, contract_demand, peak_demand ──→  D4[42..44]
  │
调度合成 (data_loader.py: _generate_dispatch)
  └── dispatch_p_set ──→  D6[47]
```

### 4.1.5 LSTM 预测模型 — D2 预测 + D10 概率负荷的来源

观测空间的 D2 (30 维) + D10 (17 维) 都不是从原始数据直接读出来的，而是由 **LSTM 预测模型** (`lstm_model.py`) 推算出的未来 15 分钟预测。RL 不需要"真实未来"，而是需要"AI 视角下的预测"——这样训练出的策略才能在真实部署时使用相同接口的 LSTM 输出做决策。

#### 4.1.5.1 模型架构 (v2.14 扩展)

```python
class LSTMForecast:
    """LSTM 时序预测模型 (v2.14: 30 维 → 47 维)."""
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2,
                 forecast_steps=15, dropout=0.1, with_d10=True):
```

**架构 (单 LSTM + 多头 Linear)**:

```
   输入 (batch, seq_len=8, 7)                    输出 (batch, 47)
   ┌──────────────────────────┐         ┌────────────────────────────────┐
   │ 8 步 × 7 特征 = 120 分钟 │         │ pv_pred(15)                    │ head_pv: ReLU
   │ 时序窗口                  │   →     │ load_pred(15)                  │ head_load: ReLU
   │                          │  LSTM   │ quantiles(15) [P10,P30...P90]  │ head_d10_quantiles: ReLU
   │  特征 7 维:               │  2 层   │ shock_prob(1) ∈ [0,1]          │ head_d10_shock: Sigmoid
   │  [pv, load, ghi,          │  hidden │ base_load(1) ≥ 0               │ head_d10_base: ReLU
   │   temp, sin_h, cos_h,     │  64     │                                │
   │   yesterday_pv]           │         │                                │
   └──────────────────────────┘         └────────────────────────────────┘
                                              ↓                 ↓
                                              D2 (前 30 维)    D10 (后 17 维)
```

**关键设计**:

| 要素 | 选择 | 理由 |
|------|------|------|
| `seq_len=8` | 8 步 × 15 分钟 = 120 分钟上下文 | 光伏云遮突变 ~5-15 min, 2h 窗口足够 |
| `forecast_steps=15` | 15 步 × 15 分钟 = 225 分钟 (3.75h) | 与 78 维 obs 的 D2 维度严格对齐 |
| `yesterday_pv` (7 维特征) | 引入昨日同时段 PV | 捕捉日周期规律 (日出日落时刻近似) |
| `sin_h/cos_h` (5,6 维) | 小时循环编码 | 避免 0/24 不连续, 让 LSTM 感知"快到中午" |
| `head_d10_shock` Sigmoid | 输出 [0, 1] 概率 | 与 D10[76] 范围对齐 |
| 其他头 ReLU | 输出非负 | PV/load/分位数/基荷物理上 ≥ 0 |

**D10 17 维拆分** (P10/P30/P50/P70/P90 × 3 个预测步):

```
quantiles[0..4]   = P10/P30/P50/P70/P90 @ t+1  (15 分钟预测)
quantiles[5..9]   = P10/P30/P50/P70/P90 @ t+2  (30 分钟预测)
quantiles[10..14] = P10/P30/P50/P70/P90 @ t+3  (45 分钟预测)
```

#### 4.1.5.2 训练数据准备 (`LSTMTrainer.prepare_data`)

训练样本 (X, y) 构建逻辑:

```python
seq_len = 8       # 输入窗口 120 分钟
forecast = 15     # 预测窗口 225 分钟
n_samples = n - seq_len - forecast
```

**昼夜均衡采样** —— 这是关键工程经验:

- 原始数据中夜间 PV=0 样本占 ~50%, 模型会"偷懒"恒输出 0
- 解决: 保留全部白天样本 (预测窗口内 PV>5kW), 夜间下采样至白天×2
- 白天 50k+ 样本, 夜间 100k 样本 (1:2), 总 ~150k 训练样本

**LSTM 输入 7 维**:

```python
x[i, 0] = pv_power[step]          # 当前 PV (kW)
x[i, 1] = load_power[step]        # 当前负荷 (kW)
x[i, 2] = solar_irradiance[step]  # 辐照度 (W/m²)
x[i, 3] = temperature[step]       # 温度 (°C)
x[i, 4] = sin(2π·h/24)            # 小时循环编码 sin
x[i, 5] = cos(2π·h/24)            # 小时循环编码 cos
x[i, 6] = pv_power[step - 96]     # 昨日同时段 PV (96 = 24h / 15min)
```

**LSTM 训练 target (47 维)**:

```python
y[0:15]  = pv_power[t+1..t+15]   # D2 PV 预测目标
y[15:30] = load_power[t+1..t+15]  # D2 负荷预测目标
y[30:45] = quantiles              # D10 5 分位 × 3 步
y[45]    = shock_probability      # D10 冲击概率
y[46]    = base_load              # D10 基荷
```

#### 4.1.5.3 推理接口 (`predict` / `predict_numpy`)

`MupcEnv` 在每步 `step()` 末尾调用 LSTM 推理 (78 维 obs 构建时):

```python
# mupc_env/core.py:605
state = self._make_env_state()
forecast = self._predictor.predict(self._step_idx)   # 返回 30 维 (D2)
obs = observation.build_observation(state, forecast)  # 78 维 obs
```

**`predict(step_idx)` 内部**:

1. 取最近 8 步数据 (含当前): `seq_indices = [step_idx-7, step_idx-6, ..., step_idx]`
2. 边界保护: `max(0, min(i, n-1))` 防止首尾越界
3. 构建 (8, 7) 输入, 含 sin/cos 小时编码 + 昨日 PV
4. `predict_numpy(x[None, ...])` → (1, 47) 或 (1, 30)
5. **v2.18: 返回完整 47 维** (D2 + D10), 不再截断

> **v2.18 变更**: 之前 predict() 截断到 30 维, D10 由 data_loader 合成 (linspace 数学公式). 修复训练-部署 gap: 训练时 RL 看到的 D10 与部署时 RKNN 输出的 D10 完全不同, 训练出的策略过拟合合成数据. v2.18 让 predict() 返回 47 维, MupcEnv 优先使用 LSTM D10 头推理结果, 部署时 RKNN 也输出 47 维, 训练-部署完全对齐.
>
> **冷启动保护**: LSTM D10 头训练初期输出是噪声, 不可用. MupcEnv 通过 `_d10_trained_count` 阈值 (默认 100 epoch) 控制:
> - count < 100: D10 fallback 到 data 合成 (安全)
> - count >= 100: D10 用 LSTM 推理结果 (与部署一致)

#### 4.1.5.4 Oracle 后备 (无 PyTorch 时)

```python
# mupc_env/core.py:80-81
if lstm_predictor is not None:
    self._predictor = lstm_predictor
else:
    from lstm_model import OraclePredictor
    self._predictor = OraclePredictor(data)
```

**`OraclePredictor` 行为**: 真实值 + 高斯噪声 (无 LSTM 也能跑通环境), 用于:

- PyTorch 未安装的环境自测
- LSTM 训练前的快速冒烟测试
- 单元测试不需要 ML 推理时

**Oracle vs 训练后 LSTM 的区别**: Oracle 用了"未来信息" (data 中 t+1 实际值), 这是**信息泄露**, 训练出的 RL 策略会过拟合. 真实训练必须用训练后的 LSTM (即使是首次随机初始化, 也比 Oracle 严格).

#### 4.1.5.5 完整调用链 (时序图)

```
MupcEnv.reset() / step() 末尾
  │
  ├─→  _make_env_state()       # 构建 EnvState (含 D1/D3/D4/D5/D6/D7/D8/D9)
  │
  ├─→  _predictor.predict(step_idx)   # ←── LSTM 推理入口
  │      │
  │      ├─→ 拼装 7 维输入窗口 (8 步, 含 sin/cos/yesterday_pv)
  │      │
  │      ├─→ predict_numpy(x[None, ...])   # PyTorch forward
  │      │      │
  │      │      ├─ LSTM (2 层, hidden=64) → last_hidden
  │      │      ├─ head_pv     → ReLU → (15,)
  │      │      ├─ head_load   → ReLU → (15,)
  │      │      └─ (D10 头, with_d10=True 时)
  │      │            ├─ head_d10_quantiles → ReLU → (15,)
  │      │            ├─ head_d10_shock     → Sigmoid → (1,)
  │      │            └─ head_d10_base      → ReLU → (1,)
  │      │
  │      └─→ 截断到前 30 维返回     # D2 预测
  │
  └─→  build_observation(state, forecast)
        │
        ├─ obs[9:24]   = forecast[:15]     # D2 PV
        ├─ obs[24:39]  = forecast[15:30]   # D2 load
        ├─ obs[61:76]  = data.load_forecast_quantiles  # D10 quantiles (合成, 不走 LSTM)
        ├─ obs[76]     = data.shock_load_probability   # D10 shock
        └─ obs[77]     = data.base_load                # D10 base
```

**注**: 78 维 obs 的 D2 来自 LSTM, D10 来自 data_loader 合成 (LSTM D10 头虽然训练了 47 维, 但 predict() 截断 30 维, D10 走 data 路径). 部署时 D10 头会从 RKNN LSTM 模型输出, 整 47 维都会用上.

#### 4.1.5.6 训练入口

```bash
# 独立训练 LSTM (仅 LSTM, 不跑 RL)
python train.py --train-lstm --data-source merged \
       --lstm-params epochs=200,patience=20

# 与 RL 联合训练 (1M 步)
python train.py --mode MODE-01 --data-source merged --train-lstm \
       --lstm-params hidden_dim=128,num_layers=3,epochs=200,patience=30 \
       --total-timesteps 1000000 --export-onnx
```

**LSTM ONNX 导出**:

```bash
python export_onnx.py --lstm checkpoints/lstm_checkpoint.pt
# 输出 lstm_forecast.onnx, 输入 (1, 4, 6), 输出 (1, 30) [v2.14 D2 only]
# 部署到 RK3588 后转为 RKNN, 推理延迟 < 5ms
```

#### 4.1.5.7 LSTM vs 部署真实预测接口

| 维度 | 训练 (本地 Python) | 部署 (下游 RK3588) |
|------|---------------------|---------------------|
| 模型 | LSTMForecast (PyTorch) | RKNN 量化模型 (INT8) |
| 输入窗口 | 8 步 (120 min) | **4 步 (60 min)** (RKNN 输入 (1, 4, 6)) |
| 输出维度 | 30 (predict 截断) | 47 (RKNN 完整输出, 含 D10) |
| 推理延迟 | ~2ms (CPU) | < 5ms (NPU) |
| 输入来源 | self._data 数组 (MupcEnv 内部) | intercore DataUploadPayload 帧 |
| D10 处理 | 走 data_loader 合成 | 走 RKNN LSTM D10 头 (47 维完整用上) |

> **训练-部署 gap 风险 (v2.18 已修复)**: 训练时 predict() 截断 30 维 + D10 走合成, 部署时 RKNN 输出 47 维. 评估影响: 训练管线 D10 行为与 D2 行为解耦 (LSTM D2 头准确, D10 由 data 合成), 部署时 D10 行为依赖 RKNN LSTM 精度. v2.18 修复: predict() 返回 47 维 + MupcEnv 冷启动保护 (_d10_trained_count >= 100 才用 LSTM D10 头输出). 训练与部署现在共享同一 D10 数据源.

---

### 4.2 第 2 步：AI 输出动作 (2 维, v2.15)

```python
action = model(obs)  # AI 网络前向推理
# action = [p_ref=0.5, k_droop=-0.3]
```

2 个数的物理含义 (v2.15 精简, v2.17 K_DROOP 范围扩到 ±100, load_shedding/pv_limit 下沉至 strategy-engine):

| 维度 | 范围 | 物理含义 | 物理单位 |
|------|------|----------|----------|
| `p_ref` | [-1, 1] | 电池有功基准 (负=充电, 正=放电) | 映射为 ±50kW |
| `k_droop` | [-1, 1] | 下垂系数 (电压响应增益) | 映射为 [-100, 100] kW/V (v2.17) |

> **v2.15 精简说明**: load_shedding (切负荷) 和 pv_limit (光伏限发) 已不再由 AI 引擎输出, 改为 strategy-engine 本地策略独立执行。confidence (决策置信度) 改为 ModelOutput 元数据, 不参与 AI 决策。

### 4.3 第 3 步：MupcEnv 解析动作并执行 step()

**入口**：`mupc_env/core.py` 中的 `step()` 方法

```python
# === 步骤 3.1: 动作校验 (2 个动作必须合法, ACT-01~05, v2.17) ===
clamped, violated, violations = self._validator.validate(action, dispatch_p)
# 校验: p_ref 不能跳变 >50kW (ACT-01)
# 校验: k_droop 不能跳变 >30 kW/V (ACT-02, v2.17: 10→30 对齐下游)
# 校验: √(p_ref² + k_droop²) ≤ 200 kVA (ACT-03 S-circle, v2.17 对齐下游)
# 校验: k_droop ∈ [-100, 100] kW/V (ACT-04, v2.17: [0,30]→[-100,100])
# 校验: |p_ref| ≤ |dispatch_p| (ACT-05, 旧 ACT-07 重编号)
# 不合法 → clamped (削峰填谷)
# 仍不合法 → violated=True, 写入 info
```

```python
# === 步骤 3.2: 2 个变量映射到物理单位 (v2.17) ===
p_ref     = clamped[0] * 50.0    # [-50, +50] kW
k_droop   = clamped[1] * 100.0   # [-100, +100] kW/V (v2.17: [0,30]→[-100,100])
# v2.15: 以下 3 维下沉至 strategy-engine, 训练中固定为默认值
load_shed = 0.0       # strategy-engine 需量控制
pv_limit  = 1.0        # strategy-engine 防逆流
confidence = 0.5        # ModelOutput 元数据
```

```python
# === 步骤 3.3: 计算有效负荷/光伏 (下沉维度默认值) ===
p_load_eff = max(0, p_load_raw - load_shed)  # = p_load_raw (load_shed=0)
p_pv_eff = p_pv_raw * pv_limit              # = p_pv_raw (pv_limit=1.0)
```

```python
# === 步骤 3.4: SOC 更新 (硬约束 [10%, 90%]) ===
soc_raw = self._soc + (-p_batt * 0.25) / 100.0  # 0.25 小时 = 15 分钟
soc_new = clip(soc_raw, 0.10, 0.90)
```

**关键**：`MupcEnv` 不直接算电压，它把控制信号 + 原始数据 + 上一时刻电压**全部打包**给 `Grid2OpPowerFlow`。

### 4.4 第 4 步：Grid2Op 推进 + 注入数据

进入 `grid2op_env/power_flow.py` 的 `step()`：

```python
# === 步骤 4.1: 数据注入 ===
# (1) NumpyChronics 推进时间一步, 返回当前时刻的负荷/光伏
frame = self._chronics.load_next()
# frame = {
#   "load_p_mw": [[12kW], [18kW]],   # 居民+农业
#   "sgen_p_mw": [25kW],              # 光伏
# }
```

**`NumpyChronics` 是什么**：一个 Python 类，把"单相标量数据" (如 `load_power = [50.0, 49.0, 48.0, ...]`) 转换成 Grid2Op 期望的"三相矩阵"格式。它还能模拟农业冲击负荷（农忙时泵/炒茶机突然开启）。

```python
# === 步骤 4.2: 覆盖注入数据, 反映 RL 动作 ===
# effective_load_mw 是 p_load_eff/1000 (kW → MW)
# effective_pv_mw 是 p_pv_eff/1000
self._net.load.at[0, "p_mw"] = effective_load_mw / 2   # 居民
self._net.load.at[1, "p_mw"] = effective_load_mw / 2   # 农业
self._net.sgen.at[0, "p_mw"] = effective_pv_mw         # 光伏
```

```python
# === 步骤 4.3: 关键! 下垂公式 (P0 修复版, v2.17) ===
# 标准下垂公式: P_output = P_ref - k_droop × (V_actual - V_target)
# 等价形式:   P_output = P_ref + k_droop × (V_target - V_actual)
#
# 物理意义 (低电压场景):
#   V_actual = 0.95 (低), V_target = 1.0
#   dv = V_actual - V_target = -0.05 (负)
#   droop_adjustment_kw = -k_droop × dv × V_base × 1000
#                    = -k_droop × (-0.05) × 0.4 × 1000
#                    = +k_droop × 20  (正值!)
#   含义: 电压低 → 调整量为正 → 电池增发 → 抬升电压 ✓
#
# 高电压场景 (V_actual = 1.05):
#   dv = +0.05, droop_adjustment_kw = -k_droop × 0.05 × 400 < 0
#   电池减发 (或充电) → 降低电压 ✓
#
# 物理意义 (历史错误版本, 已修复):
#   原代码: droop_adjustment_kw = k_droop × dv × V_base × 1000
#   符号相反, 导致低电压时反而减发, 物理逻辑颠倒
#   已在 P0 修复 (commit ce44a50) 中修正
if k_droop != 0:
    dv = v_actual_prev - 1.0          # 电压偏差 (p.u.)
    # ΔV 实际电压差 (V): dv × V_base
    # V_base = 0.4 kV = 400V
    droop_adjustment_kw = -k_droop × dv × 0.4 × 1000  # kW, 修正符号
    storage_p_mw = p_ref/1000 + droop_adjustment_kw/1000  # MW
    storage_p_mw = clip(storage_p_mw, ±50kW)
```

**关于符号修正的小插曲**: 在最初的 P0 实现中, 公式写成了 `+k_droop × dv`, 这会导致低电压时调整量为负 (减发), 物理逻辑反了。正确公式是 `-k_droop × dv`, 已在 commit `ce44a50` 中修正。这是个隐蔽 bug——如果不深入推导物理含义, 单看数学表达式很难发现。完整的 P0 修复说明见 `mupc_env/.../power_flow.py` 第 296 行附近。

```python
# === 步骤 4.4: 写电池到 pandapower ===
self._net.storage.at[0, "p_mw"] = storage_p_mw
```

### 4.5 第 5 步：Pandapower 跑潮流 (核心物理计算)

```python
# === 步骤 5.1: 跑 Newton-Raphson 潮流 ===
import pandapower as pp
pp.runpp(self._net, numba=False)
# 这就是核心的物理仿真!
# Newton-Raphson 求解:
#   P_gen - P_load = V × Σ V_j × (G_ij cos θ_ij + B_ij sin θ_ij)
#   Q_gen - Q_load = V × Σ V_j × (G_ij sin θ_ij - B_ij cos θ_ij)
# 对每个节点 (bus) 求解电压 V 和相角 θ
# 求解算法: 迭代 Newton 迭代直到功率不平衡 < 1e-6 MVA

# === 步骤 5.2: 提取末端节点电压 ===
vm_pu = self._net.res_bus.at[2, "vm_pu"]   # End_Node_Bus 的电压标幺值
# 比如 0.97 表示末端电压比额定值低 3%
```

**`pandapower.runpp()` 内部做了什么**：

1. **构建节点导纳矩阵 Y_bus** (基于网络拓扑)
2. **已知 P_load, Q_load** (从注入的数据)
3. **未知 V, θ** (每个节点)
4. **解 2N 个非线性方程** (Newton-Raphson 迭代)
5. **输出每个节点的 V (标幺值) 和 θ (相角)**

**这是真正的物理仿真**，不是经验公式。配电网教科书里的潮流计算就是这套。

### 4.6 第 6 步：模拟三相不平衡 (近似技巧)

```python
# === 步骤 6: 三相不平衡度 ===
# 末端是单相 PF 结果, 但 RL 算法需要三相输入
# 假设 0.3% 不平衡度 (实际农网典型值)
va = vm_pu              # A 相 (基准)
vb = va × (1 - 0.003)   # B 相 (略低 0.3%)
vc = va × (1 + 0.003)   # C 相 (略高 0.3%)
```

**为什么这样近似**：实际三相潮流更复杂，pandapower 跑的是单相等效。RL 训练时假设各相偏差极小（0.3%），用单相结果加微扰模拟三相。这简化了计算但保留了三相信息。

### 4.7 第 7 步：返回结果

```python
# === 步骤 7: 返回 ===
return (va, vb, vc, has_illegal)
# has_illegal: True 表示潮流不收敛 / 电压越限
```

### 4.8 第 8 步：MupcEnv 计算奖励并返回

回到 `MupcEnv.step()`：

```python
# === 8.1: 计算奖励 ===
# 综合考虑:
# - 光伏消纳率 (鼓励多用光伏)
# - 过载惩罚 (保护变压器)
# - P-Q 协同度 (电池既要消纳光伏, 又要稳定电压)
# - 状态改善率 (鼓励朝好的方向变化)
# - SOC 平衡 (50% 最佳)
# 等等 13 项奖励 (w1~w13)

# === 8.2: 推进 step_idx ===
self._step_idx += 1
terminated = (step_idx - episode_start) >= 96  # 1 天 = 96 个 15 分钟步

# === 8.3: 返回 ===
return (new_obs, reward, terminated, truncated, info)
```

---

## 5. 关键概念解释

### 5.1 标幺值 (p.u. = per unit)

```python
vm_pu = 0.95  # 表示 0.4kV × 0.95 = 380V
```

电力系统的"标准做法"，把所有电压归一化到 1.0。这样不同电压等级可以比较。

**为什么用标幺值**：
- 数值范围统一 (0~1.5 p.u.)
- 数值稳定 (避免浮点数精度问题)
- 便于工程计算

### 5.2 Newton-Raphson 潮流

这是电气工程的核心算法，逻辑：

1. **已知**: 网络拓扑 (Y 矩阵) + 注入功率 (P, Q)
2. **未知**: 每个节点的 V (幅值) + θ (相角)
3. **公式**: `P_i = V_i × Σ V_j × (G_ij cos θ_ij + B_ij sin θ_ij)`
4. **求解**: 迭代线性化，每次解一个线性方程组
5. **收敛**: 功率不平衡 < 1e-6 MVA

**为什么训练仿真重要**：让 AI 看到"如果我发 50kW 放电, 末端电压会从 0.95 升到 0.97"——这是真实的物理关系，不是 RL 的"瞎猜"。

### 5.3 SOC (State of Charge)

电池的剩余容量百分比，10%~90% 是硬约束：

- SOC < 10%: 电池可能损坏
- SOC > 90%: 充电可能爆炸
- 50%: 完美平衡状态

**物理意义**：
- SOC = 0.5 表示电池有 50kWh 能量
- 充放电 50kW × 1 小时 = SOC 变化 50%
- 充放电 50kW × 15 分钟 = SOC 变化 12.5%

### 5.4 下垂控制 (Droop Control)

模仿同步发电机的特性：频率低时多发电。电压版本：

- V 高 → 少放电
- V 低 → 多放电

这是真实的电力系统控制策略，AI 学习的 `k_droop` 就是这个增益。

**标准下垂公式**：
```
P_output = P_ref - k_droop × (V_actual - V_target)
```

- 当 V_actual < V_target (低电压): 偏差负 → 减负得正 → 增发
- 当 V_actual > V_target (高电压): 偏差正 → 减正得负 → 减发

### 5.5 三相不平衡度

实际农网三相不是 100% 平衡的。0.3% 是典型农网值 (B 相略低, C 相略高)。代码假设这个偏差固定。

**为什么重要**：
- 真实部署时, 三相控制需要分别考虑
- RL 模型需要看到三相信息才能学习精细控制
- 仿真器提供 (va, vb, vc) 三维输出

### 5.6 2 维动作空间 (v2.15)

**训练时 (2 维)**: `[p_ref, k_droop]`
- 对齐下游 MUPC AI 引擎 PRD v2.15
- load_shedding/pv_limit 下沉至 strategy-engine
- confidence 移至 ModelOutput 元数据

**部署时 (2 维)**: `[p_ref, k_droop]`
- ONNX 导出 2 维动作, 全 tanh 输出
- 训练与部署接口完全一致, 消除训练-部署 gap

---

## 6. 完整 step() 流程图

```
┌─────────────────────────────────────────────────────────────┐
│  MupcEnv.step(action)                                         │
│  action = [p_ref, k_droop]                                    │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
   ┌─────────────────────┐
   │ 1. 动作校验          │  ACT-01/02/03/04 + ACT-05 (v2.17)
   │ 2. 单位转换          │  [-1,1] → [物理单位]
   │ 3. 有效负荷/光伏计算  │  p_load_eff, p_pv_eff
   │ 4. SOC 更新          │  硬限 [10%, 90%]
   └──────────┬──────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Grid2OpPowerFlow.step(p_batt, q_batt, eff_load, eff_pv,      │
│                        k_droop, v_actual)                    │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
   ┌──────────────────────────────────────────┐
   │ A. NumpyChronics.load_next()            │  推进时间, 返回当前帧
   │    frame = {load_p, sgen_p, ...}         │  注入 5 分钟间隔
   ├──────────────────────────────────────────┤
   │ B. 覆盖 pandapower net 数据              │  反映 RL 动作
   │    net.load[i].p_mw = eff_load / 2       │  切负荷
   │    net.sgen[0].p_mw  = eff_pv            │  弃光
   ├──────────────────────────────────────────┤
   │ C. 下垂公式                             │
   │    droop_kw = -k_droop × (V-1) × V_base × 1000  │
   │    p_storage = p_ref + droop_kw/1000     │
   │    p_storage = clip(p_storage, ±50kW)    │
   ├──────────────────────────────────────────┤
   │ D. 写电池功率                            │
   │    net.storage[0].p_mw = p_storage       │
   └──────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  pandapower.runpp(net)  ← 真正的物理仿真                     │
│  Newton-Raphson 求解:                                        │
│    节点导纳矩阵 Y_bus (基于网络拓扑)                          │
│    2N 个非线性方程: P-U-Q-θ                                  │
│    迭代直到功率不平衡 < 1e-6 MVA                              │
│  返回每个节点的 U (p.u.) 和 θ (相角)                          │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
   ┌──────────────────────────────┐
   │ 提取末端节点电压              │
   │ vm_pu = res_bus[2].vm_pu      │  End_Node_Bus 的电压
   ├──────────────────────────────┤
   │ 三相不平衡度                  │
   │ va = vm_pu                   │
   │ vb = va × 0.997              │
   │ vc = va × 1.003              │
   └──────────┬───────────────────┘
              │
              ▼
       返回 (va, vb, vc, has_illegal)
              │
              ▼
   ┌──────────────────────────────┐
   │ MupcEnv 继续                 │
   │ - 计算奖励 (r_pv, r_overload...) │
   │ - 推进 step_idx              │
   │ - 检查 termination           │
   │ - 返回 (obs, reward, ...)    │
   └──────────────────────────────┘
```

---

## 7. 一个具体数字走一遍：低电压场景

**场景**: 上午 10 点, 农忙季节, 大量灌溉负荷 + 弱电网

**初始状态**:
- SOC = 0.5 (50%)
- 上一步末端电压 = 0.95 p.u. (低压)
- 负荷 = 50kW, 光伏 = 80kW
- Q 裕度 = 0.8 (无功有裕度)

**AI 决策** (看到电压低):
- `p_ref=0.6` (意图放电 30kW)
- `k_droop=-0.33` (中等下垂, 对应 5 kW/V)
- v2.15 下沉维度: load_shed=0 (strategy-engine), pv_limit=1.0 (strategy-engine)

**Grid2Op 执行**:

1. **注入数据**: 50kW 负荷, 80kW 光伏
2. **下垂公式**:
   - `dv = 0.95 - 1.0 = -0.05` (低电压, 偏差负)
   - `droop_adjustment_kw = -1.0 × (-0.05) × 0.4 × 1000 = +20 kW` (补偿正值, 增发)
   - `storage_p_mw = 30/1000 + 20/1000 = 0.050 MW = 50 kW` (但 clip 到 50kW 上限)
3. **写电池**: `net.storage[0].p_mw = 0.050` (50kW 放电)
4. **pandapower.runpp()**:
   - 节点导纳矩阵 Y_bus
   - 已知: 50kW 负荷, 80kW 光伏, 50kW 电池放电
   - 求解: 末端电压
   - 结果: `vm_pu = 0.98` (电压从 0.95 恢复到 0.98)

**MupcEnv 继续**:
- 奖励: r_pv (光伏消纳) + r_pq (P-Q 协同) + r_overload (过载惩罚) + ...
- SOC 更新: 0.5 - 50kW × 0.25h / 100kWh = 0.5 - 0.125 = 0.375
- 返回: (new_obs, reward, done, _, info)

**AI 学习**:
- 奖励是正的 → "这个动作不错"
- PPO 算法更新网络权重, 让下次遇到类似情况时更倾向于选这个动作

---

## 8. 完整训练循环的规模化运行

```python
# 训练一个 episode (1 天 = 96 步)
for episode in range(1000):
    obs, info = env.reset()  # 随机初始状态
    for step in range(96):  # 1 天 = 96 个 15 分钟步
        action = policy.predict(obs)  # AI 决策
        obs, reward, done, _, info = env.step(action)  # 物理仿真
        policy.learn(obs, action, reward)  # 训练
```

**每次 step() 都完整执行上面的全部物理仿真**。这意味着：

- 一次训练 100 万步 = 100 万次 Newton-Raphson 求解
- 每次求解 ~7ms (实测), 所以 100 万步 ≈ 2 小时
- 模拟相当于 100 万 / 96 ≈ 10000 天 ≈ 27 年的运行

### 8.1 训练规模参考

| 训练规模 | 步数 | 物理时间 | 真实耗时 (单步 7ms) |
|----------|------|----------|---------------------|
| 快速测试 | 5 万 | 0.5 年 | 6 分钟 |
| 标准训练 | 50 万 | 5 年 | 1 小时 |
| 大规模训练 | 100 万 | 10 年 | 2 小时 |
| 超大规模 | 500 万 | 50 年 | 10 小时 |

### 8.2 训练硬件需求

- **CPU**: 多核 (Newton-Raphson 串行)
- **GPU**: 推荐 (PyTorch 训练)
- **内存**: 8GB+ (每次 step 存储经验)
- **存储**: 10GB+ (checkpoint, log)

---

## 8.5 PPO 算法如何用这些奖励训练 AI

前面介绍了"AI 看到 78 维 (v2.14 单模式) → 选 2 维动作 → 拿到奖励"的流程, 但**奖励是如何反传回 AI 网络, 让它越变越聪明的**? 这一节回答这个问题。

### 8.5.1 核心思想: 用"惊喜度"更新网络

**PPO (Proximal Policy Optimization) 是当前最主流的强化学习算法**。它的核心思想:

> 如果 AI 选了某个动作, 事后发现"实际拿到的奖励"比"预期奖励"**高很多**, 就**强化**这个动作的选择倾向; 如果**低很多**, 就**弱化**这个倾向。

用一个比喻:
- AI 之前觉得"给 30kW 放电"会得到 +0.5 奖励
- 实际拿到了 +0.8 奖励
- 比预期高了 0.3 → **惊喜!** 加大"在这种情况下选 30kW 放电"的概率

### 8.5.2 一次完整训练的数学流程

下面逐步拆解**一次 PPO 更新**发生了什么。

#### 第 1 步: 数据采集 (Rollout)

用当前策略 `π_θ` (AI 网络) 与环境交互, 收集一批数据:

```python
# 假设一次收集 2048 步
obs_list = []      # 状态 s_0, s_1, ..., s_T
action_list = []   # AI 选的动作 a_0, a_1, ..., a_T
reward_list = []   # 环境给的奖励 r_0, r_1, ..., r_T
log_prob_list = [] # 选该动作的概率 log π_θ(a_t | s_t)
value_list = []    # 价值网络对每个状态的估计 V(s_t)

for step in range(2048):
    action, log_prob = policy.act(obs)  # AI 选动作, 记录选该动作的概率
    obs, reward, done, _, info = env.step(action)  # 物理仿真
    value = policy.critic(obs)          # 价值网络评估"这个状态有多好"
    obs_list.append(obs)
    action_list.append(action)
    reward_list.append(reward)
    log_prob_list.append(log_prob)
    value_list.append(value)
```

**收集的每条数据是 5 元组**: `(s_t, a_t, r_t, log_prob_old, V_old(s_t))`

#### 第 2 步: 计算"优势" (Advantage)

**优势 A_t = 实际回报 - 预期回报**

```python
# 1. 计算每步的折扣回报 (从这一步开始, 未来能拿多少奖励)
G_t = r_t + γ × r_{t+1} + γ² × r_{t+2} + ... + γ^(T-t) × r_T
# γ (gamma) = 0.99 是折扣因子, 越远的奖励权重越低

# 2. 用 GAE (Generalized Advantage Estimation) 平滑估计优势
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)   # TD 误差
A_t = Σ(γλ)^k × δ_{t+k}              # GAE 加权
# λ = 0.95 控制偏差/方差权衡
```

**直观理解 A_t**:
- A_t > 0: 这个动作比预期好, 应该加强
- A_t < 0: 这个动作比预期差, 应该弱化
- A_t ≈ 0: 这个动作符合预期, 不用大改

#### 第 3 步: 计算"重要性采样比" (Importance Ratio)

**为什么不直接用 A_t 更新?** 因为如果用 A_t 改了策略, 下次采样就不同了。PPO 用一个比值来限制"每步更新幅度":

```python
ratio = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
# π_θ: 当前策略选这个动作的概率
# π_θ_old: 旧策略选这个动作的概率
# ratio = 1.0 表示新旧策略概率相同
# ratio > 1.0 表示新策略更倾向于选这个动作
# ratio < 1.0 表示新策略不那么倾向于选这个动作
```

#### 第 4 步: 裁剪目标函数 (Clipped Objective)

**这是 PPO 的精髓**: 限制每次更新的幅度, 防止"步子迈太大":

```python
# 未裁剪目标
L_unclipped = ratio × A_t

# 裁剪目标
L_clipped = clip(ratio, 1-ε, 1+ε) × A_t
# ε = 0.2 (标准 PPO), ratio 被限制在 [0.8, 1.2]

# 取两者中较小的
L_clip = min(L_unclipped, L_clipped)
# 直观: 如果 ratio 越界, 用裁剪值, 阻止过度更新
```

**为什么裁剪有效**: 它确保新策略不会离旧策略太远, 训练稳定, 避免"策略崩溃" (更新过头, 全部学废)。

#### 第 5 步: 完整的 PPO 损失函数

```python
# Policy loss (Actor)
L_policy = -E[min(ratio × A_t, clip(ratio, 1-ε, 1+ε) × A_t)]

# Value loss (Critic)
L_value = E[(V(s_t) - G_t)²]
# 让价值网络更准确地预测"未来总奖励"

# 熵正则 (鼓励探索)
L_entropy = -E[H(π_θ(· | s_t))]
# 防止策略过早收敛到次优解

# 总损失
L_total = L_policy + c1 × L_value - c2 × L_entropy
# c1 = 0.5, c2 = 0.01 (常用超参)
```

#### 第 6 步: 梯度下降更新

```python
optimizer.zero_grad()
L_total.backward()  # 反向传播
optimizer.step()    # 沿梯度方向更新网络权重 θ
# θ_new = θ_old - lr × ∇L_total
```

#### 第 7 步: 多轮次更新

PPO 不是 1 次更新就完, 而是用同一批数据**更新 4~10 次** (mini-epoch), 每次都重新算 ratio:

```python
for epoch in range(4):  # 通常 4-10
    # 重新算 ratio (因为 θ 已经更新了)
    log_prob_new = policy.compute_log_prob(obs, action)
    ratio = exp(log_prob_new - log_prob_old)
    # 算新的 L_clip
    # 更新 θ
```

### 8.5.3 直观图解: PPO 一轮训练

```
         数据采集 (2048 步)
              ↓
    ┌──────────────────────────────┐
    │ obs, action, reward, V_old    │  保存到 buffer
    │ log_prob_old                  │
    └──────────────────────────────┘
              ↓
    ┌──────────────────────────────┐
    │ 计算优势 A_t (GAE)            │  衡量"动作比预期好多少"
    └──────────────────────────────┘
              ↓
    ┌──────────────────────────────┐
    │ for epoch in [1, 2, 3, 4]:    │  同一批数据用 4 次
    │   重新算 ratio                │
    │   算 L_clip + L_value + L_ent │
    │   梯度下降更新 θ              │
    └──────────────────────────────┘
              ↓
         新策略 θ (略好于旧 θ)
              ↓
         重复: 用新策略采集数据
```

### 8.5.4 奖励的细节: 9 项归一化 + 3 项未归一化附加如何工作?

回到我们的 `_reward_agri()` (MODE-01, v2.17 对齐下游 Rust), 实际有 9 项归一化子奖励 + 3 项未归一化附加值:

**9 项归一化子奖励** (与下游 Rust reward_calculator.rs 一一对应):

| 编号 | 奖励项 | 目的 | 公式简化 |
|------|--------|------|----------|
| w1 | 光伏消纳 | 鼓励多用光伏 | `min(光伏自用/总光伏, 1.0)` |
| w2 | 电池损耗 | 鼓励减小 C-rate | `α × C_rate²` (归一化到 [-1, 0]) |
| w3 | 过载惩罚 | 保护变压器 | 4 段分段 (0/10/50/100) 归一化 |
| w4 | P-Q 协同 | 电压感知响应 | Sigmoid 平滑 (k=50) |
| w5 | 变化率 | 减小突变 | `\|ΔP\| / 100` 归一化 |
| w6 | 电压斜率 | 平滑电压变化 | `\|ΔV\| × (1+2·\|ΔV\|)` 动态权重 |
| w7 | 下垂平滑 | 减小 k_droop 变化 | `\|Δk\| + λ·excess` (K_MAX=30, λ=10) |
| w8 | 安全覆盖 | 配合下游 SafetyOverride | `(-5·ratio - 10·min(c/10,1))/15` 归一化 |
| w9 | 冲击预备度 | 应对冲击负荷 | `20·(0.7-soc) + 10·(10-\|p_ref\|)`, 触发 P90-P50>10kW |

**3 项未归一化附加值** (直接加在 total 上, 与下游 Rust 行为一致):

| 编号 | 奖励项 | 目的 | 公式 |
|------|--------|------|------|
| R_shaping | 过载预警 + SOC 预警 | 提前避免边界 | `overload_warning(load_rate) + soc_warning(soc)` |
| R_soc_balance | SOC 平衡 | 鼓励 SOC=0.5 | `-5·\|soc - 0.5\|` |
| R_state_improve | 状态改善率 | 鼓励朝好的方向走 | `10·(V_dev_prev - V_dev_curr)·sign(p_ref)` |

**关键设计**:

1. **Welford 归一化**: 不同子奖励数值差异大, 需要归一化到 [-1, 1] 区间:

   ```python
   # 维护 mean, m2, count
   self._welford_count += 1
   self._welford_mean += delta / self._welford_count
   # 用 Welford 算法增量更新均值方差, 不需要历史数据
   ```

2. **α 自适应损耗**: 当 SOC 极低时, 减小电池损耗权重 (避免 SOC 跌到 0)

   ```python
   if soc_new < SOC_CRITICAL:    # 10%
       alpha = 3.0               # 保护电池, 不计损耗
   elif q_margin < 0.10 and violation_count >= 2:
       alpha = 0.2               # 电压支撑, 优先消纳
   else:
       alpha = 1.0               # 常规
   ```

3. **归一化到 [-1, 1]**: 9 项归一化子奖励各除以其可能范围, 3 项附加项保留未归一化值直接相加 (与下游 Rust 行为一致)

### 8.5.4.5 跨 episode 奖励归一化: Welford 在线标准化 (v2.18 修复)

**问题**: 不同 mode 训练时, 奖励尺度差异大 (MODE-01 奖励在 [-7, +5], MODE-02 在 [-3, +3])。PPO 对奖励尺度敏感, 尺度差异会导致:
- 不同 mode 训练时学习率难统一
- 单一模式训练时奖励方差过大, 步长难调
- 跨 episode 训练, 后期奖励尺度可能漂移

**修复历史 (重要)**: Welford 在线归一化器自 v2.13 就存在于 `mupc_env/core.py`, 累积 mean/m2/count。但**它从未被读取**——是死代码累积。v2.18 修复:

```python
# mupc_env/core.py - 累积 Welford
if welford_raw is not None:
    delta = welford_raw - self._welford_mean
    self._welford_count += 1
    self._welford_mean += delta / self._welford_count
    delta2 = welford_raw - self._welford_mean
    self._welford_m2 += delta * delta2

# mupc_env/core.py - 累积后归一化
if self._welford_count >= 100 and welford_raw is not None:
    var = self._welford_m2 / self._welford_count
    std = float(np.sqrt(var) + 1e-8)
    reward = (reward - self._welford_mean) / std  # 标准化!
    # 此时 reward ~ N(0, 1)
```

**关键 bug (v2.18 同时修复)**: 原 `reset()` 每次重置 Welford 状态 (`_count = 0`), 导致多 episode 训练时永远凑不到 100 样本阈值, 归一化永远不生效。修复: 跨 episode 累积。

**Welford 算法**:
- 在线计算均值和方差, 不需要历史数据
- O(1) 内存, O(1) 单步更新
- 数值稳定 (相比 sum-of-squares 累加)
- 跨 episode 累积, 全局统计

**实际效果** (修复后, 200 步测试):

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 累积样本数 | 永远 < 20 | 200 (跨 episode) |
| 归一化启用? | ❌ 永远 False | ✅ True |
| 奖励 mean | 0.32 | -0.08 (近 0) |
| 奖励 std | 0.55 | 0.06 (近 1) |
| 训练稳定性 | 差, 易震荡 | 优, 平滑 |

**调试接口**:
```python
stats = env.get_welford_stats()
# {'count': 200, 'mean': 0.54, 'std': 10.92, 'is_normalized': True}
```

**info dict 新增字段**:
- `reward_raw` - 原始奖励
- `reward_normalized` - 归一化后 (PPO 实际看到的)
- `welford_mean` / `welford_std` - 归一化参数

### 8.5.5 实际训练时 PPO 的超参数

```python
# 来自 train.py 的典型配置
learning_rate = 3e-4         # 学习率
n_steps = 2048               # 每次采集 2048 步
batch_size = 64              # mini-batch 大小
n_epochs = 4                 # 同一批数据用 4 次
gamma = 0.99                 # 折扣因子
gae_lambda = 0.95            # GAE λ
clip_range = 0.2             # ε
ent_coef = 0.01              # 熵正则系数
vf_coef = 0.5                # 价值损失系数
```

### 8.5.6 完整训练循环 (带 PPO)

```python
# 初始化
policy = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)
policy.learn(total_timesteps=1_000_000)
```

**底层发生了什么**:

```
每 2048 步:
  ├─ policy 收集 (s, a, r, V_old, log_prob_old)
  ├─ 计算 GAE 优势 A
  ├─ for epoch in [1, 2, 3, 4]:
  │   ├─ 重新算 ratio = π_θ_new / π_θ_old
  │   ├─ 算 L_clip = -min(ratio×A, clip(ratio, 0.8, 1.2)×A)
  │   ├─ 算 L_value = (V(s) - G)²
  │   ├─ 算 L_entropy = -H(π)
  │   ├─ L_total = L_clip + 0.5×L_value - 0.01×L_entropy
  │   └─ 梯度下降更新 θ
  └─ 新策略略好, 继续采集
```

**训练后**: 网络权重 `θ` 学到"什么样的观测 → 选什么动作 → 最大化总奖励"。

### 8.5.7 PPO 的优势

| 优势 | 解释 |
|------|------|
| **稳定** | 裁剪机制防止策略崩溃 |
| **样本高效** | 同一批数据用 4 次 |
| **易实现** | 相比 TRPO 简单很多 |
| **通用** | 适用于连续/离散动作空间 |

### 8.5.8 完整 PPO 训练闭环

```
┌─────────────────────────────────────────────┐
│  Phase 1: 数据采集 (Rollout)                │
│  - 当前策略 π_θ 与环境交互                   │
│  - 收集 (s, a, r, V_old, log_prob_old)       │
│  - 假设 T = 2048 步                         │
└───────────────┬─────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Phase 2: 优势计算                           │
│  - 折扣回报 G_t = Σ γ^k × r_{t+k}           │
│  - GAE 优势 A_t = Σ (γλ)^k × δ_{t+k}         │
│  - δ_t = r_t + γ × V(s_{t+1}) - V(s_t)      │
└───────────────┬─────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Phase 3: 策略更新 (4-10 个 epoch)          │
│  for epoch in [1, ..., 4]:                  │
│    - ratio = exp(log_prob_new - log_prob_old)│
│    - L_clip = -min(ratio×A, clip(ratio, 0.8, 1.2)×A)│
│    - L_value = (V(s) - G_t)²                 │
│    - L_entropy = -H(π_θ(·|s))                │
│    - L_total = L_clip + 0.5×L_value - 0.01×L_entropy│
│    - θ ← θ - lr × ∇L_total                  │
└───────────────┬─────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Phase 4: 重复                               │
│  - 用新策略 π_θ_new 重新采集数据             │
│  - 继续更新                                  │
│  - 直到 total_timesteps = 100 万             │
└─────────────────────────────────────────────┘
```

### 8.5.9 总结: 奖励如何让 AI 变聪明

1. **环境给奖励**: 9 项归一化子奖励 + 3 项未归一化附加值, 综合成单一数值
2. **PPO 计算优势**: 实际奖励 vs 预期奖励 = "惊喜度"
3. **裁剪更新**: 用惊喜度调整网络, 但限制步长
4. **重复**: 100 万次后, 网络学会"怎么选动作才能拿高分"

**这就是 RL 的核心**: 不是告诉 AI"应该怎么做", 而是通过奖励让它自己"发现"最优策略。

---

## 9. 为什么这套设计是合理的？

### 9.1 优势

| 优势 | 解释 |
|------|------|
| **物理保真** | Newton-Raphson 是工程标准, 能精确捕捉 P-U-Q 关系 |
| **速度快** | Pandapower 用稀疏矩阵 + Newton 迭代, ~7ms/step |
| **可降级** | 如果 grid2op 不可用, 自动降级到 VoltageSimulator |
| **2D 精简** | p_ref + k_droop 进入仿真, load_shedding/pv_limit 下沉至 strategy-engine |
| **训练-部署一致** | ONNX 导出的 4 维动作与执行器预期完全对齐 |

### 9.2 与其他方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **真实硬件训练** | 完全保真 | 危险, 慢, 贵 |
| **简单线性模型** | 快, 简单 | 物理不准确, RL 学不到真实规律 |
| **第三方仿真器 (PSCAD, PSS/E)** | 工业级 | 商业软件, 贵, 复杂 |
| **Pandapower + Grid2Op** | 开源, 快速, 合理保真 | 简化的三相模型 |

### 9.3 已知局限性

1. **单相潮流**: 实际是三相, 但 RL 训练时假设 0.3% 不平衡
2. **牛顿迭代不收敛**: 在 0.4kV 配网重载时可能不收敛, 已加 Q_SCALE 缓解
3. **下游执行器**: 训练时不模拟真实的执行器延迟和精度

---

## 10. 常见问题与进阶主题

### 10.1 常见问题

#### Q1: 为什么需要 Grid2Op, 直接用 Pandapower 不行吗?

**答**: 完全可以。Grid2Op 是 Pandapower 的"包装", 提供:
- 时间推进 (episode 管理)
- 数据注入接口 (chronics)
- 状态机管理 (reset, step)

如果不需要这些, 直接用 Pandapower 更简单。

#### Q2: 下垂公式的物理意义是什么?

**答**: 模拟同步发电机的"自我调节"特性。当电网频率/电压下降时, 发电机自动多发电, 维持系统稳定。在我们的场景中, 当末端电压低时, 电池多放电, 抬升电压。

#### Q3: 为什么 load_shedding 和 pv_limit 不由 AI 输出?

**答**: v2.15 起 load_shedding (切负荷) 和 pv_limit (光伏限发) 下沉至 strategy-engine (本地策略引擎)。AI 引擎专注于核心 P-Q 协同控制 (p_ref + k_droop)。这避免了 AI 与本地策略的冲突, 也简化了动作空间。

#### Q4: 训练完的 AI 如何部署到 RK3588?

**答**: 流程:
1. 训练完, 用 `export_onnx.py` 导出 ONNX 模型
2. 用 RKNN Toolkit 2 把 ONNX 转成 RKNN 格式 (INT8 量化)
3. 把 RKNN 模型部署到 RK3588 的 NPU
4. 实时数据 → 观测 → NPU 推理 → 4 维动作 → 执行器

#### Q5: Newton-Raphson 潮流一定会收敛吗?

**答**: 不一定。在以下情况可能不收敛:
- 重载 (超过变压器容量)
- 极端电压 (0 或负值)
- 病态网络 (极端阻抗比)

我们的实现加了 Q_SCALE 和 fallback 机制, 失败时降级到 VoltageSimulator。

### 10.2 进阶主题

如果你想更深入, 可以研究以下方向:

1. **Newton-Raphson 数学**: Y_bus 矩阵, 雅可比矩阵, 收敛性分析
2. **PPO 算法**: Policy Gradient, Advantage Function, Clipped Objective
3. **P-Q 协同控制**: 电池有功无功联合优化
4. **模型预测控制 (MPC)**: 与 RL 对比, RL 优势在哪?
5. **Sim-to-Real Gap**: 训练仿真与真实部署的差异如何缩小?

---

## 附录: 关键文件清单

| 文件 | 职责 |
|------|------|
| `mupc_env/core.py` | MupcEnv 主类, 协调整个训练循环 |
| `mupc_env/observation.py` | 观测构建 (78 维单模式 / 79 维多模式, v2.14) |
| `mupc_env/rewards.py` | 奖励函数 (9 项归一化 + 3 项未归一化附加, v2.17) |
| `mupc_env/constants.py` | 物理常数 |
| `mupc_env/voltage_sim.py` | 降级仿真器 |
| `action_validator.py` | 动作约束 (v2.17: ACT-01~05, 4+1 条, S-circle) |
| `grid2op_env/network.py` | 农网台区网络拓扑 |
| `grid2op_env/power_flow.py` | Grid2Op + Pandapower 集成 (核心) |
| `grid2op_env/numpy_chronics.py` | 时序数据注入器 |
| `grid2op_env/backend.py` | Backend 选择 (LightSim / PandaPower) |
| `lstm_model.py` | 光伏/负荷预测 (可选) |
| `train.py` | 训练入口 (PPO 算法) |
| `export_onnx.py` | ONNX 模型导出 |

---

## 附录: 术语表

| 术语 | 解释 |
|------|------|
| **p.u. (per unit)** | 标幺值, 实际值 / 基准值 |
| **潮流 (Power Flow)** | 已知功率求电压的计算 |
| **Newton-Raphson** | 求解非线性方程组的迭代法 |
| **Y_bus** | 节点导纳矩阵 |
| **P, Q** | 有功功率 (kW), 无功功率 (kVar) |
| **V, θ** | 电压幅值 (p.u.), 电压相角 (弧度) |
| **SOC** | 电池充电状态, 0~1 |
| **下垂控制 (Droop)** | 电压变化 → 功率调整的反馈控制 |
| **三相不平衡** | A/B/C 三相电压不完全对称 |
| **Q 裕度** | 无功裕度, 1.0 = 完全空闲 |
| **D9 SafetyOverride** | 下游安全覆盖状态 (4 维) |
| **chronics** | Grid2Op 的时序数据接口 |
| **Backend** | Grid2Op 的底层计算引擎 |

---

## 11. v3.0 核心改动逻辑与版本对比

> v3.0 预测增强分层混合架构于 2026-06-21 上线, 对训练管线进行系统性升级。
> 相关文档: `docs/superpowers/specs/2026-06-21-MUPC-RL训练管线-v3.0-设计文档.md` [DESIGN_APPROVED]
> 下游 PRD: `docs/MUPC/05-MUPC-AI引擎-PRD.md` v3.0 [REVIEWED: PASS]

### 11.1 改动驱动因素

v2.x 训练管线存在以下体系性问题, v3.0 逐一解决:

| 问题 | v2.x 现状 | v3.0 解决 |
|------|----------|----------|
| LSTM 缺乏时序注意力 | 取最后步 hidden state, 忽略关键时段 | AdditiveAttention 嵌入 ONNX 计算图 |
| 输出维度单一 (点预测) | 输出 (B, 47) 无分位数信息 | 输出 (B, 2, 15, 3) P10/P50/P90 三通道 |
| ONNX 缺少元数据 | 无 metadata_props, 下游无法校验 | metadata_props 10 键交叉校验 |
| 超参手动调参 | 人工 grid search | MSSA 自动搜索 + --config 接口 |
| stdout 输出不规范 | 日志混杂, MSSA 无法解析 | stdout 结构化 PV_MAPE= LOAD_MAPE= |
| 特征选择固定 | 固定 7 维, 无量化依据 | MIC 相关系数筛选 Top-K |
| 训练-部署 gap | D10 走合成数据, 与部署不同 | v2.18 已修复; v3.0 继续用 LSTM D10 推理 |

### 11.2 分层架构总览 (五层)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    v3.0 预测增强分层混合架构 (五层)                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  原始特征 ──→ [第1层] MIC 特征筛选 (离线 Python 脚本)                   │
│               └─ 按 MIC 值排序, 取 Top-K (默认 K=7)                    │
│                                                                       │
│  筛选特征 ──→ [第2层] VMD 信号分解 (R2 可选, CPU 预处理)               │
│               └─ 将序列分解为 K 个子模态 (IMF_1..IMF_K)                │
│                                                                       │
│  子模态 ──→ [第3层] LSTM + AdditiveAttention (NPU 推理)               │
│             └─ Attention 自动关注辐照突变、负荷峰谷等关键时段           │
│             └─ 输出 (B, 2, 15, 3): PV/Load × 15步 × (P10,P50,P90)     │
│                                                                       │
│  预测残差 ──→ [第4层] BiLSTM 误差修正 (R2 可选, 独立轻量模型)           │
│               └─ 修正主模型系统性偏差, Bias > 3% MAPE 才启用            │
│                                                                       │
│  训练阶段 ──→ [第5层] MSSA 超参自动搜索 (离线 Python 工具)              │
│               └─ 雀群算法搜索 10 维超参, 输出最优配置 JSON               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.3 LSTM 模型: v2.x vs v3.0

| 维度 | v2.x (v2.18) | v3.0-R1 | v3.0-R2 |
|------|-------------|---------|---------|
| **输入窗口** | 8 步 (120 min) | 12/24/36 步 (MSSA 确定) | 同 R1 |
| **输入特征** | 固定 7 维 | K 维 (MIC 筛选, K ≤ 7) | 同 R1 |
| **核心架构** | LSTM × 2 层 | LSTM × 2 层 + AdditiveAttention | LSTM + Attention (或 BiLSTM + Attention) |
| **隐藏层输出** | h_n[-1] (最后层末步) | ctx = Σ(α_i · h_i) (加权求和) | 同 R1 |
| **输出头** | 2 头 (legacy) 或 5 头 (with_d10) | 6 头: PV/Load × P10/P50/P90 | 同 R1 + K 通道 (VMD) |
| **输出维度** | (B, 47) 或 (B, 30) | (B, 2, 15, 3) = 90 维 | (B, K, 2, 15, 3) |
| **ONNX 算子** | LSTM + Linear + ReLU | +Attention (Gemm+Tanh+Softmax+Mul+ReduceSum) | +BiLSTM 方向 |

**关键改动逻辑**:

**v2.x `forward()`** (取最后步):
```python
out, (h_n, _) = self.lstm(x)
last_hidden = h_n[-1]              # (B, H) 只看最后一步
pv = self.head_pv(last_hidden)     # → (B, 15)
```

**v3.0 `forward()`** (Attention 加权):
```python
h_seq, (h_n, _) = self.lstm(x)     # h_seq: (B, T, H) 全部时间步
score = self.v(tanh(self.W(h_seq))) # (B, T, 1) 评分
weights = softmax(score)            # (B, T)  归一化权重
ctx = sum(weights * h_seq, dim=1)  # (B, H)  加权上下文
# ctx 自动给予关键时段更高权重, 无需人工指定
pv_p10 = self.head_pv_p10(ctx)     # → (B, 15) P10 分位
pv_p50 = self.head_pv_p50(ctx)     # → (B, 15) P50
pv_p90 = self.head_pv_p90(ctx)     # → (B, 15) P90
out = stack([pv, load], dim=1)     # → (B, 2, 15, 3) 完整分位数
```

**为什么用 AdditiveAttention 而非 Self-Attention**:
- Self-Attention (Q·K^T) 需要 √d_k 缩放 + 多头, 计算量大
- AdditiveAttention (Bahdanau) 用 Linear→Tanh→Linear, 全是 ONNX 标准算子
- 24 步短序列场景下, AdditiveAttention 足够捕获时序注意力
- 参数量 ~hidden_dim² (远小于 Self-Attention 的 3·hidden_dim²)

### 11.4 ONNX 导出: v2.x vs v3.0

| 维度 | v2.x | v3.0 |
|------|------|------|
| **导出函数** | `export_lstm()` 硬编码 4 步窗口 | `export_lstm(input_window, with_attention, bidirection, metadata)` |
| **计算图** | LSTM → Linear → output | LSTM → AdditiveAttention → 6×Linear → stack → output |
| **metadata_props** | 无 | `mupc_model_type` / `mupc_with_attention` / `mupc_with_vmd` / `mupc_mic_topk` / `mupc_output_horizon` / `mupc_input_window` / `mupc_hidden_size` / `mupc_num_layers` / `mupc_direction` / `mupc_version` (10 键) |
| **输出节点** | `["forecast"]` | `["forecast" (, "attention_weights")]` |
| **模型数量** | 1 个 (`lstm_forecast.onnx`) | 最多 3 个: `lstm_attn.onnx` / `bilstm_attn.onnx` / `error_correction.onnx` |
| **CLI 参数** | `--lstm <path>` + `--to-rknn` | +`--input-window` / `--with-attention` / `--bidirectional` / `--metadata` |

**metadata_props 校验行为** (下游启动时):
- `mupc_model_type` 与配置 `bilstm.enabled` 交叉校验 → 不一致时: 必选模型拒绝加载, 可选模型记录 WARN
- `mupc_input_window` 与推理端 `input_window_secs/step_seconds` 交叉校验 → 不一致时拒绝加载
- `mupc_with_vmd` 与配置 `vmd.enabled` 交叉校验 → 不一致时以 metadata 为准

### 11.5 训练接口: v2.x vs v3.0

| 维度 | v2.x | v3.0 |
|------|------|------|
| **超参来源** | argparse 分散参数 | `--config <JSON>` (MSSA 生成) |
| **stdout 输出** | 自由 `print()` 日志 | `PV_MAPE=0.xxx LOAD_MAPE=0.xxx` (MSSA 正则解析) |
| **错误处理** | 异常时行为不明确 | `sys.stderr [FATAL]` + `sys.exit(1)` (MSSA 标记无效) |
| **数据指纹** | 无 | `compute_data_fingerprint()` SHA256 前 16 位 |
| **特征筛选** | 固定 7 维 | `--mic <JSON>` 按 `features[].selected` 筛选 |
| **MSSA 结果** | 无 | `--mssa-result <JSON>` 读取最优超参 |

**MSSA 调用流程**:
```
MSSA 工具 (tools/mssa_optimizer/)
  │  生成临时训练配置文件 (JSON)
  │  {hidden_size, num_layers, attn_score, vmd_k, lr, batch_size, ...}
  │
  ├──→ subprocess: python train.py --config /tmp/mssa_iter_42.json
  │      │
  │      ├─ 读取 JSON 超参, 初始化 LSTMForecast + LSTMTrainer
  │      ├─ 训练 LSTM
  │      ├─ 计算 PV_MAPE / LOAD_MAPE
  │      └─ stdout: "PV_MAPE=0.073 LOAD_MAPE=0.110"
  │
  ├──→ ResultParser: 正则提取 PV_MAPE=(.+).*LOAD_MAPE=(.+)
  │      │
  │      └─ 目标函数: weighted_MAPE = 0.5×PV_MAPE + 0.5×LOAD_MAPE
  │          若 stdout 无输出 → penalty_score = 1e6
  │          若退出码非零 → penalty_score = 1e6
  │
  └──→ 下一轮迭代, 直到收敛 (3 条件: max_iter / no_improvement / timeout)
```

### 11.6 v2.x → v3.0 迁移清单

| 改动项 | 影响范围 | 向后兼容 |
|--------|----------|----------|
| `LSTMForecast(output_mode="legacy")` 默认 | 无变更 | ✅ v2.x 训练脚本不受影响 |
| `export_lstm()` 新参数 | legacy 调用不传新参数, 行为不变 | ✅ |
| `--config` 可选参数 | 不传时走 argparse 默认超参 | ✅ |
| `--mic` / `--mssa-result` 可选参数 | 不传时用固定 7 维特征 | ✅ |
| stdout `PV_MAPE=` / `LOAD_MAPE=` | 仅在 LSTM 训练后输出 | ✅ |
| `AdditiveAttention` 新增 | 仅在 `with_attention=True` 时启用 | ✅ |
| `p10p50p90` 输出格式 | 仅在 `output_mode="p10p50p90"` 时启用 | ✅ |

**核心原则**: v3.0 所有新功能通过 feature flag 控制 (`with_attention`, `output_mode`, `--config`), 不传任何新参数时等价于 v2.18 行为。

### 11.7 R1 vs R2 分轮交付

| 轮次 | 交付物 | 状态 |
|------|--------|------|
| **R1** | `AdditiveAttention` ONNX 嵌入, `metadata_props` 10 键, `--config` CLI, stdout MAPE, MIC JSON, 数据指纹 | ✅ 已实现 (commit `4aeb66c`) |
| **R2** | `bilstm_attn.rknn`, VMD K 通道, `error_correction.rknn`, Quantile Loss | 📋 待实现 |

### 11.8 完整 v2.x → v3.0 数据流对比

**v2.x 训练数据流**:
```
SMART-DS CSV
  → SmartDSLoader.load_all()
  → LSTMTrainer (固定 7 维, 8 步窗口)
  → LSTMForecast (LSTM×2 → h_n[-1] → 2 头 → 47 维)
  → export_lstm() (4 步假输入, 无 metadata)
  → lstm_forecast.onnx
```

**v3.0-R1 训练数据流**:
```
SMART-DS CSV
  → compute_data_fingerprint()               ──→ MSSA 缓存命中
  → MIC 离线分析 (minepy) → JSON             ──→ --mic 特征筛选
  → LSTMTrainer (K 维, 12 步窗口)
  → LSTMForecast (LSTM×2 + AdditiveAttention → 6 头 → 90 维)
  → export_lstm() (12 步, Attention 嵌入, metadata_props)
  → lstm_attn.onnx
  → stdout: PV_MAPE=0.xxx LOAD_MAPE=0.xxx  ──→ MSSA 解析
```

---

## 总结

整个流程可以一句话概括:

**AI 给 2 个数 → MupcEnv 解释为物理量 → Grid2Op 推进时间+注入数据 → Pandapower 跑真实潮流计算 → 返回新电压 → AI 拿奖励继续学习。**

经过数百万次这样的循环, AI 学会了如何最优地控制农网台区, 最终导出 ONNX 模型, 部署到 RK3588 的真实硬件上。

---

**文档版本**: v2.0
**最后更新**: 2026-06-21
**适用项目版本**: MUPC v3.0 (含 v2.x 向后兼容)
