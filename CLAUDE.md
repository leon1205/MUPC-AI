# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

为 MUPC（多功能电力电子变换器）训练基于强化学习的边缘 AI 模型，目标在 RK3588 NPU 上实现毫秒级控制。

**当前状态：** 项目处于早期阶段。Python RL 训练管线（`train.py`、`mupc_env.py` 等）尚未实现，文档体系与数据集已就绪。本仓库负责两件事：(1) 训练出可在 RK3588 上部署的 RL 模型（Python 端），(2) MUPC 通信管理模块的 AI 引擎 Rust 实现（部署端，Rust crate `mupc-ai-engine`）。

- **训练端（本仓库）**: Python 3.9+，SB3 PPO/SAC，Gymnasium 环境，ONNX 导出
- **部署端（目标平台）**: Rust + Tokio，RK3588 NPU，openEuler，RKNN Runtime (INT8)
- **数据源**: SMART-DS 数据集（光伏 CSV + 负荷 per-unit profile，已下载至 `data/`）
- **核心文档**: PRD — `docs/MUPC/05-MUPC-AI引擎-PRD.md`，设计文档 — `docs/MUPC/05-MUPC-AI引擎-设计文档.md`，项目主文档 — `docs/MUPC/PROJECT-MUPC-项目设计主文档.md`

## 现有文件清单

```
data/download_smart_ds.py     # SMART-DS 数据集下载脚本（已执行，数据已就绪）
data/smart_ds/solar/          # 光伏 CSV（6 个场景，15分钟分辨率）
data/smart_ds/load_profiles/  # 负荷 per-unit profile（4 个商业场景）
data/smart_ds/parquet/        # 建筑级负荷数据（2 个场景）
docs/MUPC/                    # 项目文档
.claude/agents/               # AI 代理协作框架配置
```

## AI 代理协作框架

本项目配置了一套 AI Agent 协作工作流（合同与路径驱动），定义在 `.claude/agents/` 下：

- **角色定义**: `.claude/agents/AI_WORKFLOW/01_AGENTS.md` — 10 个角色（项目经理 → 需求分析师 → 架构师 → 开发工程师 → 代码评审员 → QA 等）
- **工作流**: `.claude/agents/AI_WORKFLOW/02_WORKFLOW.md` — 三条路径（标准功能/简单修改/纯端），强制代码评审门禁
- **行为准则**: `.claude/agents/AI_WORKFLOW/03_AI_RULES.md` — 文档驱动、拒绝硬编码、KISS 原则、中文沟通
- **术语表**: `.claude/agents/AI_WORKFLOW/05_GLOSSARY.md`
- **工作流采用「合同+门禁」模式**：每个阶段产出需获得通过标记（`[REVIEWED: PASS]`、`[DESIGN_APPROVED]`、`[CODE_REVIEWED: PASS]`、`[TEST_PASSED]`）才能推进

当用户提出功能需求时，项目经理（Manager）Agent 将按工作流调度各角色协作。

## 常用命令

```bash
# 下载/补全 SMART-DS 数据集
python data/download_smart_ds.py

# --- 以下命令对应尚未实现的模块，仅作规划参考 ---

# 训练（默认多模式 PPO，一个模型覆盖全部 5 种场景）
python train.py

# 单模式训练
python train.py --mode combined

# 切换算法
python train.py --algo sac
python train.py --algo ppo   # 默认

# 自定义奖励权重
python train.py --reward-weights w_safety=0.6 w_economic=0.3

# 快速测试
python train.py --total_timesteps 50000

# 导出模型（ONNX + RKNN）
python export_onnx.py
python export_onnx.py --to-rknn

# 环境自测
python mupc_env.py

# 数据加载器自测
python data_loader.py
```

## 架构总览（规划）

```
data_loader.py          →  加载 SMART-DS 数据，TOU 电价生成，归一化，8:2 切分
         ↓
mupc_env.py             →  Gymnasium Env：6维观测，2维连续动作，多任务奖励
         ↓
train.py                →  PPO/SAC 训练，默认多模式单模型，EvalCallback
         ↓
export_onnx.py          →  策略网络 → ONNX → RKNN（INT8），含 onnxruntime 验证
```

**关键模块关系（规划）**：

- `mupc_env.py` 不依赖任何 RL 框架，可独立运行
- `_ppo_core.py` 是纯 NumPy 的 PPO 实现（MLP + GAE + Clipped Objective），SB3 不可用时 `train.py` 自动切换
- `_gym_stub.py` 提供 `gymnasium.Env` / `spaces.Box` 的最小替代，无 gymnasium 时自动降级
- `export_onnx.py` 从 NumPy PPO checkpoint（`ppo_weights.npz`）读取权重并导出 ONNX

## 观测与动作空间

**观测 (6-dim)**：`[load_rate, SOC, pv_norm, price_norm, hour_encoded, mode_id]`


| 维度 | 名称         | 范围      | 说明                                        |
| ---- | ------------ | --------- | ------------------------------------------- |
| 0    | load_rate    | [0, ~1.5] | 变压器视在功率(kVA) / 额定容量，含 P+Q      |
| 1    | SOC          | [0, 1]    | 电池荷电状态，reset 时随机初始化 [0.2, 0.8] |
| 2    | pv_norm      | [0, 1]    | 光伏功率归一化                              |
| 3    | price_norm   | [0, 1]    | 电价归一化                                  |
| 4    | hour_encoded | [-1, 1]   | sin(hour × 2π / 24)                       |
| 5    | mode_id      | [0, 1]    | 运行模式编码：0.0/0.25/0.5/0.75/1.0         |

**动作 (2-dim)**：`[p_ratio, q_ratio]` 各 ∈ [-1, 1]

- p_ratio > 0 → 放电，p_ratio < 0 → 充电
- q_ratio → 无功补偿，影响变压器 kVA 视在功率

## 环境参数


| 参数           | 值        | 说明                       |
| -------------- | --------- | -------------------------- |
| 变压器容量     | 500 kVA   | `TRANSFORMER_KVA`          |
| 电池容量       | 200 kWh   | `BATTERY_CAPACITY_KWH`     |
| 最大充放电功率 | 100 kW    | `BATTERY_P_MAX_KW`         |
| SOC 硬限制     | 10% ~ 90% | 代码中 hard clip，不可弱化 |
| 光伏容量       | 200 kW    | `PV_ARRAY_KW`              |
| 负荷峰值       | 400 kW    | `LOAD_PEAK_KW`             |
| 过载阈值       | 85%       | `OVERLOAD_THRESHOLD`       |
| 时间步长       | 15 分钟   | 全年 35040 步              |

## 5 种奖励模式

所有模式的奖励分量均归一化到 [-1, 1]，通过权重组合。可通过 `--mode` 选择：


| 模式              | 核心目标               | 主要权重分布                                |
| ----------------- | ---------------------- | ------------------------------------------- |
| `demand_control`  | 变压器不过载           | safety 0.70 + smooth 0.15                   |
| `economic`        | 峰谷套利               | economic 0.55 + safety 0.25                 |
| `combined`        | 安全>经济>寿命（推荐） | safety 0.50 + economic 0.30 + lifetime 0.10 |
| `pv_self_consume` | 光伏本地消纳           | pv_util 0.60 + safety 0.30                  |
| `backup_reserve`  | 需量+最小备用容量      | safety 0.55 + soc_margin 0.20               |

奖励分量（`mupc_env.py` 中 `_reward_*` 方法）：

- `_reward_safety`: 二次过载惩罚（从 75% 开始有梯度，85% 达 -0.05，100% 达 -0.31，100%+ 迅猛上升）
- `_reward_economic`: 归一化价差套利
- `_reward_lifetime`: C-rate² 电池衰减
- `_reward_smooth`: 有功+无功功率变化惩罚（防抖）
- `_reward_soc_margin`: SOC 边际管理（<20% 或 >80% 惩罚）
- `_reward_pv_utilization`: 弃光惩罚（仅 pv_self_consume 模式启用）

安全惩罚关键设计：从 75% 负载率开始提供梯度（而非旧版 85% 零点起步），agent 在安全区内就能感知到负载上升的危险方向，学会"预防"而非"事后补救"。

每个 step 的 info dict 包含所有分量的原始值，TensorBoard / CSV 中可单独监控。

## 代码规范

1. **安全第一**: 硬件控制相关代码必须注释 "SAFETY"，SOC guard 是硬约束不可弱化
2. **类型提示**: 所有函数包含完整 Type Hints 和 Docstrings
3. **模块化**: 状态空间、动作空间、奖励函数解耦为独立方法，方便修改
4. **不过度设计**: 不引入不必要的第三方库，确保嵌入式 Linux 可运行
5. **所有回复请用中文**
6. **文档驱动**: 任何代码修改必须先有对应的 PRD 或设计文档更新
7. **拒绝硬编码**: 密钥、Token、绝对路径使用配置文件或环境变量
