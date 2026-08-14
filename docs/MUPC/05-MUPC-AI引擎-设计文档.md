---
# MUPC AI 引擎 - 模块设计文档

## 目录

1. [模块架构](#1-模块架构)
2. [LSTM 预测管线设计](#2-lstm-预测管线设计)（原 Ch2 + Ch14）
   - 2.1 功能概述 & 分层增强架构
   - 2.2 预测规格 (7维输入, 24步, p10p50p90)
   - 2.3 核心技术选型 (VMD/Attention/7维/BiLSTM/误差修正)
   - 2.4 核心数据模型 (LstmInput/HistorySample/LstmOutput/EnhancedForecastResult)
   - 2.5 LSTM 核心模型 (LstmModel)
   - 2.6 预测增强管线 (PredictionPipeline/增强等级/降级机制)
   - 2.7 数据流 (基线/VMD/误差修正/full_decision_cycle集成)
   - 2.8 模块划分
   - 2.9 性能预算
   - 2.10 错误处理与降级
   - 2.11 跨项目接口 (MUPC-AI2)
3. [多源数据融合设计](#3-多源数据融合设计)
4. [强化学习模型设计](#4-强化学习模型设计)
   - 4.1 功能概述
   - 4.2 分层控制架构
   - 4.3 算法选择
   - 4.4 完整状态空间表（10 大类，78 维）
   - 4.5 完整动作空间表（2 维）
   - 4.6 ActionOutput 结构体 & 解析
   - 4.7 RLModel 结构体
   - 4.8 ActionValidator 约束规则校验
   - 4.9 场景切换平滑过渡（R3）
5. [奖励函数计算模块](#5-奖励函数计算模块)
   - 5.1 功能概述
   - 5.2 RewardCalculator 结构体
   - 5.3 SCENE-01：台区季节性负荷模式
   - 5.4 SCENE-B1：自主套利
   - 5.5 SCENE-B2：需量控制
   - 5.6 SCENE-B3：虚拟电厂
   - 5.7 SCENE-B5：极致绿色
   - 5.8 SceneWeights 映射表
   - 5.9 折扣累积奖励机制（R2）
   - 5.10 冲击负荷概率预测
   - 5.11 变压器过载分段惩罚（R-04）
   - 5.12 电压斜率惩罚动态权重（R-05）
   - 5.13 冲击负荷响应奖励（R-06）
   - 5.14 P-Q 协同度阈值可配置化（R-07）
   - 5.15 奖励函数精细化改进
6. [RKNN Runtime 设计](#6-rknn-runtime-设计)
7. [ModelManager 统一调度设计](#7-modelmanager-统一调度设计)
   - 7.1 功能概述
   - 7.2 结构体
   - 7.3 full_decision_cycle() 完整流程
   - 7.4 影子模型验证+渐进式切换（R1）
   - 7.5 自适应权重优化器
8. [安全 RL 包装器](#8-安全-rl-包装器safety-rl-wrapper)（原 §5.16）
   - 物理模型前置过滤器，基于戴维南等效电路预测电压变化
   - 调用位置：full_decision_cycle() 中 RL 决策后、ActionValidator 前
9. [与策略引擎集成设计](#9-与策略引擎集成设计)
10. [文件结构](#10-文件结构)
11. [配置结构](#11-配置结构)
12. [错误类型](#12-错误类型)
13. [消息总线集成](#13-消息总线集成)
14. [技术决策记录](#14-技术决策记录)
15. [MSSA 超参优化工具设计](#15-mssa-超参优化工具设计)（已迁移至训练管线设计文档）
附录. [版本演进](#附录版本演进)

## 1. 模块架构

### 1.1 整体架构

AI 优化引擎是 MUPC 通信管理模块的核心智能决策组件，对应 workspace crate `mupc-ai-engine`。架构遵循\"AI 优先，本地兜底\"策略：正常时 AI 引擎主导决策，AI 失效时自动降级至本地策略引擎接管控制。

```
+-----------------------------------------------------------------------------+
|                         AI 优化引擎 (ai-engine)                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|  数据源层                       融合层                 决策层               |
|  +------------+              +----------------+    +-------------------+    |
|  | intercore  |---TCP------->|                |    | RewardCalculator  |    |
|  | (实时数据)  |              | DataFusion     |    | (奖励计算)         |   |
|  +------------+              | Engine         |--->+--------+----------+    |
|  |    LSTM    |---预测------>| (1Hz融合)      |             |               |
|  | (预测值)   |              |                |    +--------v----------+    |
|  +------------+              | 输出:          |    | RLModel            |    |
|  |  气象 API  |---拉取------>| FusedSystem    |--->| (决策模型)          |   |
|  +------------+              | State          |    +--------+----------+    |
|  | 物联平台   |---订阅------>| (78维向量)     |             |               |
|  | (电价)     |              |                |    +--------v----------+    |
|  +------------+              |                |    | ActionValidator    |    |
|  | gateway    |---事件------>|                |    | (5条约束校验)       |   |
|  | (调度指令)  |              |                |    +--------+----------+    |
|  +------------+              |                |             |               |
|  | ModeSelector|---模式----->|                |    +--------v----------+    |
|  | (预设5场景) |              +-------+--------+    | ModelManager       |    |
|  +------------+                      |              | (统一调度)          |   |
|                                      |              +--------+----------+    |
|                    +--------v--------+                       |               |
|                    |  RKNN Runtime   |---FFI--- librknnrt.so |               |
|                    |  (NPU 推理)     |                       |               |
|                    +--------+--------+                       |               |
|                             |                                |               |
|              +--------------+------------+                   |               |
|              v              v            v                    |               |
|        +---------+   +---------+   +---------+              |               |
|        | RK3588  |   |  x86    |   | CPU降级 |              |               |
|        | NPU     |   | Server  |   | 兜底    |              |               |
|        +---------+   +---------+   +---------+              |               |
+-----------------------------------------------------------------------------+
                                       |
                                +------v-------+
                                | strategy-     |
                                | engine        |
                                | (AiIntegrator)|
                                | (AiValidator) |
                                +--------------+
```

### 1.2 核心模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| LSTM 预测模型 | `lstm_model.rs` | 光伏出力与负荷功率时序预测，输出 15~30 分钟预测向量 |
| 多源数据融合 | `data_fusion.rs` | 周期性（1Hz）从 5 个数据源采集数据，融合为 FusedSystemState |
| 模式选择器 | `mode_selector.rs` | 5 种预设运行场景互斥选择，支持远程（IEC 104/61850）和本地（Web UI）切换 |
| 强化学习模型 | `rl_model.rs` | MADDPG/PPO 多目标决策，2 维动作空间输出（v2.15） |
| 奖励计算器 | `reward_calculator.rs` | 5 种场景奖励函数计算，驱动在线微调 |
| 鲁棒性管理器 | `robustness_manager.rs` | 电压异常应急策略，检测并返回应急动作（v2.9 新增） |
| 动作约束校验 | `action_validator.rs` | 4 条约束规则校验（ACT-DUAL-01~04），防止异常值危害设备（v2.15：load_shedding/pv_limit 下沉至策略引擎） |
| RKNN Runtime | `rknn_runtime.rs` | RK3588 NPU FFI 推理封装，异步安全 |
| FFI 绑定 | `rknn_runtime_sys.rs` | librknnrt.so C API 声明 |
| RKNN 类型 | `rknn_types.rs` | FFI 边界数据结构 |
| 模型管理器 | `model_manager.rs` | 统一调度 LSTM 预测 + RL 决策，full_decision_cycle() |
| 在线微调 | `online_updater.rs` | 基于新数据持续更新模型权重 |
| 配置 | `config.rs` | AiEngineConfig 及所有子配置 |
| 错误 | `error.rs` | AiEngineError 枚举 |

### 1.3 模块依赖关系

```
mupc-ai-engine (独立 crate)
  |-- tokio (异步运行时)
  |-- serde / serde_json (序列化)
  |-- tracing (日志)
  |-- chrono (时间戳)
  |-- thiserror (错误派生)
  +-- librknnrt.so (FFI 动态链接)
```

被依赖方：
```
mupc-strategy-engine --> mupc-ai-engine (AiIntegrator, AiCommandValidator)
mupc-web-api          --> mupc-ai-engine (通过 AiIntegrator 门面)
```

### 1.4 数据流

```
历史数据 --> LSTM.predict() --> 光伏/负荷预测值 --> 供 RL 模型使用
                                                          |
远程指令/本地选择 --> ModeSelector --> 运行模式 --> 权重映射 + 奖励函数选择
                                                          |
5数据源 --> DataFusionEngine.fuse() --> FusedSystemState --> to_input_vector()
                                                          |
融合向量 + 预测值 + 运行模式 --> RLModel.decide() --> ActionOutput
                                                          |
ActionOutput (p_ref, k_droop) --> ActionValidator.validate() --> 通过--> 下发 strategy-engine
                                                     --> 不通过--> clamp + WARN 日志
                                                      |
决策-执行对 --> RewardCalculator.calculate() --> 奖励值 --> OnlineUpdater
```

> **数据流说明：** Q_batt由实时电压调节器闭环控制，不经过RL模型

### 1.5 完整决策周期

`ModelManager.full_decision_cycle()` 是每个决策周期（默认 1 秒）的顶层入口，串联全部子模块调用：

```
full_decision_cycle():
  1. running_mode = mode_selector.current()
  2. lstm_output = lstm_model.predict(lstm_input)
  3. fused_state = data_fusion.fuse()
  4. input_vector = fused_state.to_input_vector()
  5. scene_weights = SceneWeights::lookup(running_mode)
  6. action = rl_model.decide(input_vector, lstm_output, scene_weights)
  7. validated = action_validator.validate(action)
  8. reward = reward_calculator.calculate(running_mode, action, fused_state)
  9. online_updater.add_sample(DataPoint { input_vector, validated, reward })
  10. return validated_action
```

### 1.6 非功能性指标

| 指标 | 要求 |
|------|------|
| NPU 推理延迟 | < 100ms (P99) |
| AI 完整决策总延迟 | < 120ms（状态输入 + 推理 + 校验） |
| 场景切换延迟 | < 2s（远程）、< 1s（本地） |
| 数据融合周期 | 1Hz（默认），可配置 1s ~ 60s |
| 动作约束校验延迟 | < 0.5ms |
| 奖励函数计算延迟 | < 1ms（单场景） |
| LSTM 预测延迟 | < 1s |
| 推理运行时内存 | <= 200MB |
| 单模型 INT8 大小 | <= 5MB |
| 训练数据本地存储 | <= 1GB（最近 30 天） |
| AI 引擎 MTBF | >= 1000 小时 |
| AI 失效自动降级 | < 2s |
| 权重优化推理延迟 | < 100ms（v2.11）|
| 权重优化更新周期 | >= 1 小时（v2.11）|
| 分位数预测延迟 | <= 1s（v2.11）|
| 冲击负荷概率计算延迟 | <= 10ms（v2.11）|
| P90 分位数误差 | < 15%（v2.11）|

## 2. LSTM 预测管线设计

> 整合原第 2 章"LSTM 模型设计"与第 14 章"LSTM 预测增强管线设计"。
> 覆盖：核心模型设计 + 接口定义 + VMD/Attention/BiLSTM/误差修正增强 + 性能预算 + 降级策略。

### 2.1 功能概述

LSTM 时序预测管线负责预测未来 225 分钟（15 步 × 15 分钟）的光伏出力和负荷功率，为 RL 决策模型提供 D2 前瞻性输入和 D10 概率负荷预测。

**分层增强架构**（对应论文吸收方案三轮路径）：

```
基线: LSTM (24步×7维) → 联合预测 (47维或(2,15,3))
 R1: + VMD 信号分解 (CPU, 纯 Rust) + Attention (ONNX 内嵌)
 R2: + BiLSTM 可选 + 误差修正 BiLSTM (独立模型)
 R3: + MSSA 超参自动优化 (离线 Python)
```

| 维度 | v2.16 (基线) | v3.0 (R1 增强) | v3.0 R2 (最大) |
|------|:--:|:--:|:--:|
| 模型架构 | 单向 LSTM | LSTM + AdditiveAttention | BiLSTM + Attention |
| 信号预处理 | 无 | VMD (K=5 PV / K=6 Load) | VMD |
| 误差修正 | 无 | 无 | 独立 BiLSTM 残差模型 |
| 输入维度 | 1 维单变量 | 7 维多特征 | 7 维多特征 |
| 输入步长 | 24 步 | 24 步 | 24 步 |
| NPU 推理次数 | 2 (PV+Load 分别) | 1 (联合) 或 K×2 (VMD IMF) | 1-3 |
| 模型文件数 | 1 | 1 | 1-3 (lstm_attn + bilstm_attn + error_correction) |

### 2.2 预测规格

| 项目 | 规格 |
|------|------|
| 预测目标 | 光伏出力 (PV)、负荷功率 (Load) 联合预测 |
| 负荷分类 | 基荷、可调负荷、冲击负荷（D10 概率预测） |
| 预测范围 | 225 分钟（15 步 × 15 分钟） |
| 输入窗口 | 6 小时（24 步 × 15 分钟 = 21600s），MSSA 可搜索 {12, 24, 36} |
| 输入特征 | **7 维**：[pv, load, ghi, temp, sin_hour, cos_hour, yesterday_pv]，展平 row-major 168 f32 |
| 输出格式 | p10p50p90: `(2, 15, 3)` = 90 f32；legacy: `(47,)` |
| 模型格式 | ONNX（PyTorch 训练）→ INT8 量化 → .rknn（RK3588 NPU 部署） |
| 精度要求 | 光伏 MAPE ≤ 8%（R1）/ ≤ 7.5%（R2）；负荷 MAPE ≤ 13%（R1）/ ≤ 12%（R2） |

### 2.3 核心技术选型

#### 2.3.1 VMD 信号分解：纯 Rust 实现

纯 Rust ADMM 迭代，依赖 `rustfft` (FFT) + `nalgebra` (矩阵)。无 FFI 依赖，直接 `cross build` 到 aarch64。性能约束：单次 VMD (N=24, K≤8) ≤ 50ms。

**约束**：VMD 仅适用于单变量时序。当 `input_features > 1` 时，VMD 路径自动降级为 AttentionOnly，由 7 维 LSTM 内部的跨特征交互替代 VMD 的角色。

#### 2.3.2 Attention 机制：ONNX 图内嵌

MUPC-AI2 训练管线在 ONNX 导出时将 AdditiveAttention (Bahdanau) 嵌入计算图（MatMul + Tanh + Softmax + ReduceSum，全为 ONNX 标准算子）。Rust 侧零代码改动，推理在 NPU 上执行。

#### 2.3.3 7 维多特征输入

训练管线硬编码 7 维特征，Rust 侧 `HistorySample` 结构体采集全部特征：

| 索引 | 特征 | Rust 数据来源 |
|:--:|------|-------------|
| 0 | pv_power | `FusedSystemState.pv_power` |
| 1 | load_power | `FusedSystemState.load_power` |
| 2 | solar_irradiance (GHI) | `FusedSystemState.solar_irradiance` (D5) |
| 3 | temperature | `FusedSystemState.temperature` (D5) |
| 4 | sin_hour | CPU 计算 `sin(2π × hour / 24)` |
| 5 | cos_hour | CPU 计算 `cos(2π × hour / 24)` |
| 6 | yesterday_pv | 96 步前 pv_power (冷启动 fallback = 当前 PV) |

`LstmConfig.input_features` 默认 7，设为 1 回退 v2.16 单变量模式。

#### 2.3.4 BiLSTM：双模型文件 + Go/No-Go 准入

训练管线导出两个独立模型：`lstm_attn.rknn`（必须）和 `bilstm_attn.rknn`（可选）。

双重门控：
- **配置门** `bilstm.enabled`: 运维主动开启
- **硬件验证门** `bilstm.gate_passed`: RK3588 P99 延迟摸底通过后设为 true

No-Go 时自动回退单向 LSTM。

#### 2.3.5 误差修正 BiLSTM：独立模型 + 独立 Runtime

`error_correction.rknn` 为独立模型文件，拥有独立的 `RknnRuntime` 实例。输入为历史残差序列（T 步 actual−predicted），输出为 15 步修正量。与主预测严格串行，总延迟 ≤ 200ms。

冷启动保护：缓冲未满时 `zero_init=true` 使用零向量填充（修正量 = 0，等效直接输出）。

### 2.4 核心数据模型

#### 2.4.1 模型输入 — LstmInput

```rust
/// LSTM 模型输入（v3.0: 展平 2D 数组）
///
/// 布局: row-major — [t0_f0, t0_f1, ..., t1_f0, ...]
/// 长度 = input_window_steps × input_features (默认 24 × 7 = 168)
pub struct LstmInput {
    pub history: Vec<f32>,   // 展平的多特征历史序列
    pub timestamp: i64,
}
```

#### 2.4.2 历史样本 — HistorySample

```rust
/// 7 维历史样本（与训练管线 prepare_data() 特征顺序一致）
pub struct HistorySample {
    pub pv_power: f64,
    pub load_power: f64,
    pub solar_irradiance: f64,
    pub temperature: f64,
    pub sin_hour: f64,
    pub cos_hour: f64,
    pub yesterday_pv: f64,
}
```

#### 2.4.3 模型输出

```rust
/// 点预测输出
pub struct LstmOutput {
    pub predictions: Vec<f32>,  // v3.0: 自动检测 90 或 47 维输出格式
}

/// 概率负荷预测输出（D10 数据流）
pub struct ProbabilisticLoadOutput {
    pub timestamp: i64,
    pub quantiles: Vec<QuantilePrediction>,  // P10/P50/P90 × 15 步
    pub base_load: f32,
    pub shock_probability: f64,
    pub confidence: f64,   // 基于分位数间距，非预测序列方差
}
```

> **confidence 字段说明**：`LstmOutput.confidence`（基于预测序列方差）已在 v2.16 删除。`ProbabilisticLoadOutput.confidence`（基于 P50/P90 分位数间距）保留，是有效的统计量。

#### 2.4.4 增强预测结果

```rust
pub struct EnhancedForecastResult {
    pub pv_forecast: Vec<f64>,
    pub load_forecast: Vec<f64>,
    pub load_quantiles: Option<ProbabilisticLoadOutput>,
    pub enhancement_level: EnhancementLevel,
    pub vmd_degraded: bool,
    pub error_correction_applied: bool,
}
```

### 2.5 LSTM 核心模型 — LstmModel

```rust
pub struct LstmModel {
    config: LstmConfig,
    runtime: RknnRuntime,
}

impl LstmModel {
    pub fn new(config: LstmConfig) -> Result<Self, AiEngineError>;
    pub async fn load(&mut self) -> Result<(), AiEngineError>;
    /// v3.0: 输入展平的 (T, K) 序列，单次 ONNX 联合推理
    pub async fn predict(&self, input: &LstmInput) -> Result<LstmOutput, AiEngineError>;
    /// 分位数后处理（CPU，基于预测输出计算 P10/P50/P90）
    pub async fn predict_quantiles(&self, input: &LstmInput, covariates: &LoadCovariates) -> Result<ProbabilisticLoadOutput, AiEngineError>;
    pub fn model_type(&self) -> ModelType;
}
```

**ONNX 输出格式自动检测**：`predict()` 根据输出长度自动识别：
- 90 维 → p10p50p90: `(2, 15, 3)`
- 47 维 → legacy D2+D10
- 30 维 → legacy D2 only

**输出维度校验**：输出不足时返回 `OutputShapeMismatch` 错误（不静默补零）。

```rust
let output_steps = self.config.output_horizon_secs / self.config.step_seconds; // = 15
if output.len() < output_steps {
    return Err(AiEngineError::OutputShapeMismatch);
}
```

### 2.6 预测增强管线 — PredictionPipeline

#### 2.6.1 增强等级（降级追踪）

```
Level 0: FullVmdAttentionCorrection — VMD + (Bi)LSTM/Attention + 误差修正 (Go 路径)
Level 1A: BiLstmVmdAttention        — BiLSTM + VMD + Attention (无误差修正)
Level 2: VmdAttention               — VMD + LSTM/Attention
Level 3: AttentionOnly              — LSTM/Attention (无 VMD)
Level 4: Baseline                   — 基线 LSTM (v2.16 等效)
Level 5: (ModelManager 层) — 全零预测
```

自动升级/降级机制：
- 连续 **3** 次某模块失败 → 降级到下一层
- 连续 **5** 次某模块成功 → 尝试升回上一层

#### 2.6.2 管线结构体

```rust
pub struct PredictionPipeline {
    vmd_pv: Option<VmdDecomposer>,
    vmd_load: Option<VmdDecomposer>,
    lstm_model: Arc<RwLock<Option<LstmModel>>>,
    lstm_history: Arc<RwLock<VecDeque<HistorySample>>>,
    input_size: usize,
    input_features: usize,    // v3.0
    config: PredictionEnhancementConfig,
    health: RwLock<PipelineHealth>,
    error_correction_runtime: Option<RknnRuntime>,   // R2
    residual_buffer_pv: Option<RwLock<ResidualBuffer>>,  // R2
    residual_buffer_load: Option<RwLock<ResidualBuffer>>, // R2
}
```

#### 2.6.3 VMD 分解器

```rust
pub struct VmdDecomposer { config: VmdConfig; }

pub struct VmdConfig {
    pub k: usize,          // 模态数 (PV=5, Load=6)
    pub alpha: f64,        // 惩罚因子 (默认 2000)
    pub tau: f64,          // 噪声容忍度
    pub tol: f64,          // 收敛容差 (默认 1e-6)
    pub max_iter: usize,   // 最大迭代 (默认 500)
}

pub struct VmdResult {
    pub imfs: Vec<Vec<f32>>,
    pub reconstructed: Vec<f32>,
    pub reconstruction_error: f64,
    pub iterations: usize,
    pub converged: bool,
}
```

#### 2.6.4 残差缓冲 (R2)

```rust
pub struct ResidualBuffer {
    capacity: usize,          // = residual_window_steps (默认 24)
    buffer: VecDeque<f32>,    // FIFO 循环缓冲
    zero_init: bool,          // 冷启动零填充标志
    total_pushed: usize,
}
```

#### 2.6.5 管线健康状态

```rust
pub struct PipelineHealth {
    pub vmd_consecutive_failures: u32,
    pub vmd_consecutive_successes: u32,
    pub ec_consecutive_failures: u32,     // R2
    pub ec_consecutive_successes: u32,    // R2
    pub bilstm_consecutive_failures: u32, // R2
    pub bilstm_consecutive_successes: u32,// R2
    pub current_level: EnhancementLevel,
}
```

### 2.7 数据流

#### 2.7.1 基线数据流 (Level 4)

```
HistorySample Buffer → build_flat_input() → (168,) ONNX 输入
  → LstmModel::predict() → 单次 NPU 推理 → parse_baseline_output()
  → {pv_forecast, load_forecast, load_quantiles}
```

#### 2.7.2 VMD 增强数据流 (Level 0-2, 仅 input_features=1)

```
pv_history → VMD(K) → {IMF_1..IMF_K} → 逐 IMF NPU 推理 → 重构 (Σ)
load_history → VMD(K) → {IMF_1..IMF_K} → 逐 IMF NPU 推理 → 重构 (Σ)
  → 重构值注入 FusedSystemState
```

#### 2.7.3 误差修正数据流 (Level 0, R2)

```
主预测 y_pred → ResidualBuffer.build_input() → 误差修正推理 (RknnRuntime #2)
  → e_pred → y_corrected = y_pred + e_pred → 注入 FusedSystemState
```

#### 2.7.4 与 full_decision_cycle() 集成

```rust
// model_manager.rs — 增强路径优先，失败时降级到基线
let (pv_forecast, load_forecast, load_quantiles) = self
    .run_enhanced_predict()  // PredictionPipeline 优先
    .await
    .unwrap_or_else(|_| self.run_lstm_predict_with_quantiles().await);
```

### 2.8 模块划分

| 文件 | 职责 | 轮次 |
|------|------|:--:|
| `lstm_model.rs` | LSTM 核心模型 (LstmModel, LstmInput, 分位数结构体) | 基线 |
| `prediction_pipeline.rs` | 增强管线编排器 (VMD→推理→重构→误差修正→降级) | R1+R2 |
| `vmd.rs` | 纯 Rust VMD 分解 (ADMM) | R1 |
| `pipeline_config.rs` | 增强配置 (VMD/Attention/BiLSTM/EC 配置 + EnhancementLevel) | R1+R2 |
| `residual_buffer.rs` | 残差滑动窗口缓冲 | R2 |
| `model_validator.rs` | ONNX metadata 校验 | R2 |
| `model_manager.rs` | 集成 PredictionPipeline + 历史缓冲管理 + 7D 输入构建 | R1 |
| `config.rs` | LstmConfig 扩展 (input_features, yesterday_offset_steps, step_seconds) | 基线 |

**不修改的模块**：`rknn_runtime.rs`（ONNX 内嵌 Attention 对 Runtime 透明）、`data_fusion.rs`（MIC 离线筛选在训练阶段完成）。

### 2.9 性能预算

| 阶段 | 基线 (v2.16) | R1 (VMD+Attn) | R2 (+BiLSTM+EC) |
|------|:--:|:--:|:--:|
| VMD 分解 (CPU) | — | ≤ 50ms | ≤ 50ms |
| 主预测 NPU 推理 | ≤ 200ms | ≤ 250ms | ≤ 500ms |
| 误差修正推理 | — | — | ≤ 200ms |
| **总延迟** | **≤ 200ms** | **≤ 300ms** | **≤ 750ms** |
| 内存增量 | 0 | +5MB (VMD 缓冲) | +8MB (含残差缓冲+第二 Runtime) |
| 模型文件大小 | ≤ 5MB | ≤ 8MB | ≤ 15MB (三模型) |

### 2.10 错误处理与降级

```
Level 0 全功能 → 误差修正失败 → Level 1A (BiLSTM+VMD)
              → BiLSTM 失败   → Level 2 (VMD+Attention)
              → VMD 失败      → Level 3 (AttentionOnly)
Level 1A      → BiLSTM 失败   → Level 2
              → VMD 失败      → Level 3
Level 2       → VMD 失败      → Level 3
Level 3       → Attention 失败 → Level 4 (Baseline)
Level 4       → LSTM 推理失败  → Level 5 (全零预测, ModelManager 层)
```

关键降级约束：
- VMD 与 `input_features > 1` 互斥：多特征模式下 VMD 自动禁用（VMD 仅适用于单变量）
- 误差修正连续 3 次失败后持久化禁用（需 OTA 恢复）
- 错误码：`VmdFailed` / `VmdNotConverged` / `AttentionDegraded` / `ErrorCorrectionFailed` / `OutputShapeMismatch` / `ModelValidationFailed`

### 2.11 跨项目接口（与 MUPC-AI2 训练管线）

| 接口项 | Rust 侧 (MUPC) | Python 侧 (MUPC-AI2) |
|------|------|------|
| ONNX 输入 shape | `(batch, 24, 7)` 展平 168 f32 | `dummy_input = (1, 24, 7)` |
| ONNX 输出 (p10p50p90) | `(batch, 2, 15, 3)` = 90 f32 | 6 头 Linear，stack 为 (2,15,3) |
| 特征顺序 | [pv, load, ghi, temp, sin_hour, cos_hour, yesterday_pv] | prepare_data() 同序 |
| 步长 | step_seconds = 900s | dt_hours = 0.25 |
| metadata_props | 交叉校验 mupc_model_type / mupc_with_attention / mupc_input_window | 导出时注入 |

### 2.12 测试策略

| 测试类型 | 覆盖范围 | 工具 |
|------|------|------|
| VMD 分解单元测试 | K=2~8，信号长度 24~96，收敛/未收敛/NaN 输入 | `cargo test -p mupc-ai-engine vmd` |
| 增强管线降级测试 | 5 级降级路径逐级触发，连续 3 次失败→降级，连续 5 次成功→升级 | `cargo test -p mupc-ai-engine pipeline` |
| 误差修正零填充测试 | ResidualBuffer 冷启动 < capacity → 零向量输入 → 修正量=0 | `cargo test -p mupc-ai-engine residual` |
| ONNX 输出兼容测试 | 90 维(p10p50p90)、47 维(legacy+D10)、30 维(legacy) 自动检测 | `cargo test -p mupc-ai-engine lstm` |
| 7 维输入构建测试 | `build_flat_input()` 输出 168 f32，`HistorySample::to_features()` 返回 [f32; 7] | `cargo test -p mupc-ai-engine model_manager` |
| 模型文件校验测试 | SHA256 匹配/不匹配/文件缺失/RKNN 文件大小=0/类型不匹配 | `cargo test -p mupc-ai-engine model_validator` |
| 全管线集成测试 | `full_decision_cycle()` 完整流程 (融合→预测→RL决策→安全校验) | `cargo test -p mupc-ai-engine core_pipeline_integration` |

### 2.13 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:--:|:--:|------|
| Attention ONNX 算子 RKNN 不支持 | 低 | 高 | Softmax/MatMul/Tanh 均为 ONNX 标准算子；提前验证 RKNN Toolkit 2 兼容性 |
| VMD ADMM 收敛慢 (max_iter=500) | 低 | 中 | 50ms 硬超时 + 自动降级回退原始序列 |
| BiLSTM 参数量翻倍导致 NPU 内存/延迟超标 | 中 | 中 | Go/No-Go 双重门控；RK3588 延迟摸底通过后才启用 |
| 残差缓冲冷启动期间无修正效果 | 高 | 低 | `zero_init=true` 零填充，修正量=0 等效直接输出 |
| 7 维气象特征 (GHI/温度) 采集不到 | 中 | 中 | 缺失时用上一周期值回填；训练侧同时产出 1 维 fallback 模型 |
| 误差修正 BiLSTM 与主预测模型版本不兼容 | 低 | 中 | OTA 下发的两模型必须包含匹配的 metadata.mupc_version |
| 历史缓冲冷启动 (前 96 步无 yesterday_pv) | 高 | 低 | 用当前 PV 作为 fallback，与训练侧 `prepare_data()` 一致 |
| ONNX output shape 与 Rust 预期不一致 | 低 | 高 | `predict()` 中自动检测 90/47/30 维输出格式并分支处理 |

## 3. 多源数据融合设计

### 3.1 功能概述

DataFusionEngine 周期性（默认 1Hz）从 5 个数据源采集数据，使用 DataSourceAdapter trait 统一接入，融合为 FusedSystemState 供 RL 决策器使用。所有数据源采集并行执行，超时不影响其他数据源。

### 3.2 DataFusionEngine 结构体

```rust
/// 多源数据融合引擎
pub struct DataFusionEngine {
    /// 融合周期（秒）
    fusion_period: Duration,
    /// 上一次融合输出（用于缺失数据回填）
    last_fused_state: Arc<RwLock<FusedSystemState>>,
    /// 5 个数据源适配器（Box<dyn DataSourceAdapter> 多态）
    sources: Vec<Box<dyn DataSourceAdapter>>,
    /// 每个数据源的最后成功采集时间戳
    source_health: Vec<SourceHealth>,
    /// 健康监控使能
    health_monitoring: bool,
}

/// 数据源健康状态
#[derive(Debug, Clone)]
pub struct SourceHealth {
    pub source_name: String,
    pub last_success_ts: i64,
    pub consecutive_failures: u32,
    pub status: HealthStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,    // 正常（绿色）
    Degraded,   // 延迟 3+ 周期（黄色）
    Failed,     // 断连 10+ 周期（红色）
}
```

### 3.3 DataSourceAdapter Trait

```rust
/// 数据源适配器 trait
#[async_trait]
pub trait DataSourceAdapter: Send + Sync {
    /// 数据源名称
    fn name(&self) -> &str;

    /// 获取最新数据
    async fn fetch(&self) -> Result<SourceData, AiEngineError>;

    /// 数据源类型（用于缺失处理策略选择）
    fn source_type(&self) -> SourceType;

    /// 超时时间（毫秒）
    fn timeout_ms(&self) -> u64;
}

/// 数据源类型（决定缺失处理策略）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    Realtime,       // 实时数据 (intercore) -- 最高优先级
    Prediction,     // LSTM 预测数据
    Price,          // 电价数据
    Weather,        // 气象数据
    Dispatch,       // 调度指令
}

/// 数据源采集结果（含时间戳）
#[derive(Debug, Clone)]
pub struct SourceData {
    /// 数据来源类型
    pub source_type: SourceType,
    /// 采集时间戳 (UTC ms)
    pub fetch_ts: i64,
    /// 实时电气数据（仅 Realtime 源填充）
    pub realtime_data: Option<RealtimeData>,
    /// 预测数据（仅 Prediction 源填充）
    pub prediction_data: Option<PredictionData>,
    /// 电价数据（仅 Price 源填充）
    pub price_data: Option<PriceData>,
    /// 气象数据（仅 Weather 源填充）
    pub weather_data: Option<WeatherData>,
    /// 调度指令（仅 Dispatch 源填充）
    pub dispatch_data: Option<DispatchData>,
}
```

### 3.4 5 个数据源适配器实现

**Adapter 1: IntercoreAdapter (Realtime)**

```rust
pub struct IntercoreAdapter {
    /// 核间 TCP 连接 (连接到 intercore 模块)
    connection: Arc<IntercoreConnection>,
}
```

数据字段：battery_soc, pv_power, load_power, grid_power, transformer_load, battery_power, voltage_phase_a/b/c, current_demand。更新频率 1Hz，通过核间 TCP 获取。

**Adapter 2: LstmAdapter (Prediction)**

```rust
pub struct LstmAdapter {
    /// LSTM 模型引用
    model: Arc<RwLock<Option<LstmModel>>>,
    /// 历史数据缓冲区
    history_buffer: Arc<RwLock<VecDeque<f32>>>,
}
```

数据字段：pv_forecast_15min, load_forecast_15min。调用 LSTM 模型预测接口获取。缺失时使用全零向量，RL 决策仅依赖实时数据。

**Adapter 3: PriceAdapter (Price)**

```rust
pub struct PriceAdapter {
    /// MQTT 订阅客户端
    mqtt_client: Arc<MqttClient>,
    /// 最近一次有效电价数据
    cached_price: Arc<RwLock<PriceData>>,
}
```

数据字段：current_electricity_price, next_period_price, price_tariff_id, peak_price, valley_price。通过 MQTT 北向订阅 `price/real_time` 主题获取。缺失时使用上一有效值填充，连续缺失 3 个周期后使用默认分时电价表。

**Adapter 4: WeatherAdapter (Weather)**

```rust
pub struct WeatherAdapter {
    /// 气象 API HTTP 客户端
    http_client: reqwest::Client,
    /// API 端点
    api_url: String,
    /// 最近一次有效气象数据
    cached_weather: Arc<RwLock<WeatherData>>,
}
```

数据字段：solar_irradiance, temperature。通过定时拉取气象 API (15 分钟间隔) 获取。缺失时使用上一有效值，连续缺失 10 个周期后绿色场景奖励函数 R_green 中碳减排项强制置 0。

**Adapter 5: DispatchAdapter (Dispatch)**

```rust
pub struct DispatchAdapter {
    /// gateway IEC 104 / IEC 61850 事件接收器
    event_receiver: broadcast::Receiver<DispatchEvent>,
    /// 最新调度指令（事件驱动，可能为 None）
    current_dispatch: Arc<RwLock<Option<DispatchData>>>,
}
```

数据字段：dispatch_p_set (Option<f64>), dispatch_q_set (Option<f64>)。通过 gateway 事件驱动接收。缺失时两个字段均为 None，RL 决策跳过调度相关约束 (ACT-DUAL-04)。

### 3.5 FusedSystemState 结构体（34 字段，78 维输入向量）

```rust
/// 融合系统状态（v2.14：10 大类，78 维输入向量）
#[derive(Debug, Clone)]
pub struct FusedSystemState {
    // ------- D1: 实时数据 (9 个 RL 字段，q_realtime_margin 已移至 D7) -------
    /// UTC 时间戳（毫秒），辅助字段
    pub timestamp: i64,
    /// 电池荷电状态 [0.0, 1.0]
    pub battery_soc: f64,
    /// 光伏功率 (kW)，正值=发电，[-1000.0, 1000.0]
    pub pv_power: f64,
    /// 负荷功率 (kW)，正值=用电，[-1000.0, 1000.0]
    pub load_power: f64,
    /// 电网交换功率 (kW)，正值=购电，[-1000.0, 1000.0]
    pub grid_power: f64,
    /// 变压器负载率 [0.0, 2.0]
    pub transformer_load: f64,
    /// 电池当前充放电功率 (kW)，[-500.0, 500.0]
    pub battery_power: f64,
    /// A 相电压标幺值 [0.8, 1.2] -- 用于过/低电压检测指导 P/Q 控制
    pub voltage_phase_a: f64,
    /// B 相电压标幺值 [0.8, 1.2]
    pub voltage_phase_b: f64,
    /// C 相电压标幺值 [0.8, 1.2]
    pub voltage_phase_c: f64,
    /// 实时模块剩余无功容量比例 [0.0, 1.0]，0=打满，1=空闲（v2.5 新增）
    pub q_realtime_margin: f64,

    // ------- D2: 预测数据 (2 个向量字段) -------
    /// 未来 15-30 分钟光伏预测 (kW)，默认 15 维
    pub pv_forecast_15min: Vec<f64>,
    /// 未来 15-30 分钟负荷预测 (kW)，默认 15 维
    pub load_forecast_15min: Vec<f64>,

    // ------- D3: 电价 (3 RL + 2 aux = 5 字段) -------
    /// 当前实时电价 (元/kWh)
    pub current_electricity_price: f64,
    /// 下一时段电价 (元/kWh)
    pub next_period_price: f64,
    /// 分时电价时段标识: 0=谷, 1=平, 2=峰, 3=尖峰
    pub price_tariff_id: u8,
    /// 峰时段电价 (元/kWh)，辅助字段
    pub peak_price: f64,
    /// 谷时段电价 (元/kWh)，辅助字段
    pub valley_price: f64,

    // ------- D4: 需量状态 (3 字段) -------
    /// 当前实际需量 (kW)
    pub current_demand: f64,
    /// 需量合同值 (kW)
    pub contract_demand: f64,
    /// 本月最大需量 (kW)
    pub peak_demand_this_month: f64,

    // ------- D5: 气象 (2 字段) -------
    /// 当前光照强度 (W/m^2)
    pub solar_irradiance: f64,
    /// 环境温度 (deg C)
    pub temperature: f64,

    // ------- D6: 调度指令 (2 字段) -------
    /// 调度主站下发的有功设定值 (kW)，None 表示无调度指令
    pub dispatch_p_set: Option<f64>,
    /// 调度主站下发的无功设定值 (kVar)，None 表示无调度指令
    pub dispatch_q_set: Option<f64>,

    // ------- D7: 季节时段 (8 字段，v2.5 新增) -------
    /// 季节 one-hot 编码（6 维）：[灌溉季, 炒茶季, 空调季, 常规季, 保留, 保留]
    pub season_encoding: [f64; 6],
    /// 时段 one-hot 编码（2 维）：[白天, 夜间]
    pub time_period_encoding: [f64; 2],
    // ------- D9: 安全覆盖状态 (v2.10 新增，v2.14 扩展至 5 字段) -------
    /// 安全覆盖激活标志
    /// true = 实时控制模块正在覆盖 AI 有功指令
    pub safety_override_active: bool,
    /// 安全覆盖触发原因（仅在 active=true 时有效）
    pub safety_override_reason: Option<String>,
    /// 安全覆盖强制放电功率 (kW)（仅在 active=true 时有效）
    pub safety_override_p_ref: Option<f64>,
    /// 安全覆盖连续触发次数（v2.14 新增）
    pub safety_override_consecutive: u32,
    /// 安全覆盖滑动窗口内覆盖比例（v2.14 新增，范围 [0.0, 1.0]）
    pub safety_override_ratio: f64,

    // ------- D10: 概率负荷预测 (v2.11 新增，3 字段) -------
    /// 分位数负荷预测（P10/P50/P90...），15 维向量
    pub load_forecast_quantiles: Vec<f64>,
    /// 冲击负荷发生概率 [0.0, 1.0]
    pub shock_load_probability: f64,
    /// 基础负荷（50% 分位数），单位 kW
    pub base_load: f64,
}
```

### 3.6 to_input_vector() -- 78 维序列化

将 FusedSystemState 转换为 RL 模型输入时，各维度按定义顺序拼接为 **78 维向量**（v2.14 从 76 维扩展）。Option 字段为 None 时填充 0.0。预测向量长度超过配置时裁剪，不足时补零。

```rust
impl FusedSystemState {
    /// 序列化为 78 维输入向量（v2.14，v2.15 修正 D1 重复）
    ///
    /// 布局（与 MUPC-AI2 训练管线 `observation.py:to_input_vector` 严格对齐）:
    ///   [0..8]   D1 实时数据 (9 标量，q_realtime_margin 已移至 D7)
    ///   [9..23]  D2 pv_forecast (15 维)
    ///   [24..38] D2 load_forecast (15 维)
    ///   [39..41] D3 电价 (3 个 RL 字段)
    ///   [42..44] D4 需量 (3 字段)
    ///   [45..46] D5 气象 (2 字段)
    ///   [47]     D6 dispatch_p_set (1 维，None 时填 0.0)
    ///   [48]     D7 q_realtime_margin (1 维，v2.14 从 D1 移入)
    ///   [49..54] D8 season_encoding (6 维)
    ///   [55..56] D8 time_period_encoding (2 维)
    ///   [57..60] D9 safety_override (4 维，v2.14 扩展 consecutive+ratio)
    ///   [61..75] D10 load_forecast_quantiles (15 维，v2.11 新增)
    ///   [76]     D10 shock_load_probability (1 维，v2.11 新增)
    ///   [77]     D10 base_load (1 维，v2.11 新增)
    pub fn to_input_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(78);

        // [0..8] D1: 9 标量（v2.15 修正：q_realtime_margin 已移至 D7，不在此处）
        v.push(self.battery_soc as f32);
        v.push(self.pv_power as f32);
        v.push(self.load_power as f32);
        v.push(self.grid_power as f32);
        v.push(self.transformer_load as f32);
        v.push(self.battery_power as f32);
        v.push(self.voltage_phase_a as f32);
        v.push(self.voltage_phase_b as f32);
        v.push(self.voltage_phase_c as f32);

        // [9..23] D2 pv_forecast: 15 维
        let pv = pad_or_truncate(&self.pv_forecast_15min, 15);
        v.extend(pv.iter().map(|&x| x as f32));

        // [24..38] D2 load_forecast: 15 维
        let load = pad_or_truncate(&self.load_forecast_15min, 15);
        v.extend(load.iter().map(|&x| x as f32));

        // [39..41] D3: 3 个 RL 字段
        v.push(self.current_electricity_price as f32);
        v.push(self.next_period_price as f32);
        v.push(self.price_tariff_id as f32);

        // [42..44] D4: 3 字段
        v.push(self.current_demand as f32);
        v.push(self.contract_demand as f32);
        v.push(self.peak_demand_this_month as f32);

        // [45..46] D5: 2 字段
        v.push(self.solar_irradiance as f32);
        v.push(self.temperature as f32);

        // [47] D6: dispatch_p_set (None 时 0.0)
        v.push(self.dispatch_p_set.unwrap_or(0.0) as f32);

        // [48] D7: q_realtime_margin（v2.14 从 D1 移入，v2.15 修正：仅出现一次）
        v.push(self.q_realtime_margin as f32);

        // [49..56] D8: season_encoding (6 维) + time_period_encoding (2 维)
        for &s in &self.season_encoding { v.push(s as f32); }
        for &t in &self.time_period_encoding { v.push(t as f32); }

        // [57..60] D9: safety_override (4 维，v2.14 扩展)
        v.push(if self.safety_override_active { 1.0 } else { 0.0 });
        v.push(self.safety_override_p_ref.unwrap_or(0.0) as f32);
        v.push(self.safety_override_consecutive as f32);
        v.push(self.safety_override_ratio as f32);

        // [61..75] D10 load_forecast_quantiles: 15 维 (v2.11 新增)
        let quantiles = pad_or_truncate(&self.load_forecast_quantiles, 15);
        v.extend(quantiles.iter().map(|&x| x as f32));

        // [76] D10 shock_load_probability (v2.11 新增)
        v.push(self.shock_load_probability as f32);

        // [77] D10 base_load (v2.11 新增)
        v.push(self.base_load as f32);

        debug_assert_eq!(v.len(), 78, "输入向量必须为 78 维");
        v
    }
}

/// 辅助函数：填充或裁剪向量到目标长度
fn pad_or_truncate(vec: &[f64], target_len: usize) -> Vec<f64> {
    let mut result: Vec<f64> = vec.iter().take(target_len).copied().collect();
    while result.len() < target_len {
        result.push(0.0);
    }
    result
}

/// 验证输入向量无 NaN/Inf（PRD 9.5 安全要求）
///
/// 在将输入向量传入 RKNN Runtime 之前调用，防止异常值导致 NPU 推理异常。
/// 检测到 NaN 或 Inf 时返回错误并记录 ERROR 日志。
fn validate_input_vector(v: &[f32]) -> Result<(), AiEngineError> {
    for (i, &val) in v.iter().enumerate() {
        if val.is_nan() || val.is_infinite() {
            tracing::error!("输入张量第 {} 维包含 NaN/Inf: {}", i, val);
            return Err(AiEngineError::InferenceFailed(
                format!("输入张量第 {} 维包含 NaN/Inf", i)
            ));
        }
    }
    Ok(())
}
```

### 3.7 缺失数据处理策略

| 缺失数据 | 处理方式 | 告警级别 |
|----------|----------|----------|
| 实时数据 (intercore) | 使用上一有效值填充，连续缺失 3 个周期后触发 AI 降级 | ERROR --> 降级 |
| 预测数据 (LSTM) | 使用全零向量，RL 决策仅依赖实时数据 | WARN |
| 电价数据 | 使用上一有效值，连续缺失 3 个周期后使用默认分时电价表 | WARN |
| 气象数据 | 使用上一有效值，连续缺失 10 个周期后绿色场景 R_carbon 强制置 0 | WARN |
| 调度指令 | 对应字段置 None，RL 决策跳过 ACT-DUAL-04 约束 | INFO |

### 3.8 融合执行流程

```rust
impl DataFusionEngine {
    pub async fn fuse(&self) -> Result<FusedSystemState, AiEngineError> {
        let mut fused = FusedSystemState::default();

        // 并行采集 5 个数据源，各自超时不影响其他源
        let handles: Vec<_> = self.sources.iter().map(|src| {
            let timeout = Duration::from_millis(src.timeout_ms());
            async {
                tokio::time::timeout(timeout, src.fetch()).await
            }
        }).collect();

        let results = futures::future::join_all(handles).await;

        // 逐源填充 + 健康状态更新
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(Ok(data)) => {
                    self.apply_source_data(&mut fused, data);
                    self.source_health[i].mark_success();
                }
                _ => {
                    self.source_health[i].mark_failure();
                    self.apply_fallback(&mut fused, self.sources[i].source_type());
                }
            }
        }

        // 更新上次融合状态缓存
        *self.last_fused_state.write().await = fused.clone();

        Ok(fused)
    }
}
```

## 4. 强化学习模型设计

### 4.1 功能概述

RLModel 使用 MADDPG（多智能体深度确定性策略梯度）或 PPO（近端策略优化）算法，基于融合状态向量、LSTM 预测值和运行场景权重，输出 2 维动作空间（p_ref + k_droop）的最优控制指令。load_shedding 和 pv_limit 已下沉至策略引擎（需量控制/防逆流策略独立执行），confidence 保留在 ModelOutput 内部用于校验，不再作为动作维度。

### 4.2 分层控制架构

为适应台区季节性负荷"高频随机脉冲叠加"工况，采用分层控制架构，实现时间尺度解耦：

**底层（实时控制模块）**
- 无功补偿（Q_batt）：根据电压实时闭环调节，响应时间 ms 级
- 三相不平衡：不涉及电池充放电，由实时控制核心模块独立处理
- 调节方式：查表法或 PID，不经过 AI
- 执行器按下垂公式 `P_output = P_ref - k_droop × ΔV` 执行毫秒级暂态调节

**上层（RL决策）**— v2.15 现行，2 维动作空间
- `p_ref`（有功基准点，[-50.0, 50.0] kW）：AI 负责稳态全局优化，通过核间 TCP 下发
- `k_droop`（电压-有功下垂系数，[0.0, 30.0] kW/V）：AI 设置暂态调节灵敏度，通过核间 TCP 下发
- `load_shedding`（可中断负荷切除量）：下沉至 strategy-engine（需量控制策略独立执行）
- `pv_limit`（光伏限功率比例）：下沉至 strategy-engine（防逆流策略独立执行）
- `confidence`（决策置信度）：保留在 ModelOutput 中（action_validator 内部校验使用）

**分层优点：**
- P 是 s/min 级慢变量，Q 是 ms 级快变量，单一网络同时学习两个时间尺度任务收敛困难且易振荡
- RL 专注于能量管理（光伏消纳、SOC平衡、过载预防），电压质量由底层保障
- 部署时 Q 失控风险与 RL 解耦
- v2.7 双参数模式（p_ref + k_droop）实现时间尺度解耦：AI 负责稳态，执行器负责暂态

**动作空间对比：**

| 维度 | v2.3（4维） | v2.4~v2.6（3维） | v2.7~v2.12（4维） | v2.13~v2.14（5维） | v2.15（2维） | 说明 |
|------|------------|-----------------|------------------|-------------------|-------------|------|
| A1 | p_batt_set [-50,50]kW | p_batt_set [-50,50]kW | p_ref [-50,50]kW | p_ref [-50,50]kW | p_ref [-50,50]kW | 有功基准点（RL控制） |
| A2 | q_batt_set | ~~Q替代~~ | k_droop [0,30]kW/V | k_droop [0,30]kW/V | k_droop [0,30]kW/V | 下垂系数（v2.7新增，实时模块闭环） |
| A3 | load_shedding [0,60]kW | load_shedding [0,60]kW | load_shedding [0,60]kW | load_shedding [0,60]kW | —（下沉至策略引擎） | v2.15 起由需量控制策略独立执行 |
| A4 | pv_limit [0,1] | pv_limit [0,1] | pv_limit [0,1] | pv_limit [0,1] | —（下沉至策略引擎） | v2.15 起由防逆流策略独立执行 |
| A5 | - | - | - | confidence [0,1] | —（保留在 ModelOutput） | v2.15 起仅用于内部校验，非动作维度 |

> **注：** v2.4 起 Q 控制完全交给实时控制模块闭环调节。v2.15 起 AI 动作空间精简为 2 维（p_ref + k_droop），load_shedding 和 pv_limit 下沉至 strategy-engine 本地策略独立执行，confidence 保留在 ModelOutput 中供 action_validator 内部校验。表中 v2.3 的 q_batt_set 和 v2.4~v2.6 的 p_batt_set 为历史版本字段，现行代码中已不再使用。

### 4.3 算法选择

| 算法 | 适用场景 | 特点 |
|------|----------|------|
| MADDPG | 多目标优化（默认） | 支持连续动作空间，经验回放，目标网络 |
| PPO | 需快速收敛时 | 信任区域限制，稳定性好，on-policy |

算法类型由 `RlConfig.algorithm` 指定，训练阶段在 x86 服务器完成，部署阶段仅执行推理。

### 4.4 完整状态空间表（10 大类，78 维）

| 维度 | 字段名 | 类型 | 取值范围 | 单位 | 说明 |
|------|--------|------|----------|------|------|
| **D1-实时** | battery_soc | f64 | [0.0, 1.0] | - | 电池荷电状态 |
| | pv_power | f64 | [-1000.0, 1000.0] | kW | 光伏出力 |
| | load_power | f64 | [-1000.0, 1000.0] | kW | 负荷功率 |
| | grid_power | f64 | [-1000.0, 1000.0] | kW | 电网交换功率 |
| | transformer_load | f64 | [0.0, 2.0] | - | 变压器负载率 |
| | battery_power | f64 | [-500.0, 500.0] | kW | 电池充放电功率 |
| | voltage_phase_a | f64 | [0.8, 1.2] | p.u. | A 相电压标幺值 |
| | voltage_phase_b | f64 | [0.8, 1.2] | p.u. | B 相电压标幺值 |
| | voltage_phase_c | f64 | [0.8, 1.2] | p.u. | C 相电压标幺值 |
| **D2-预测** | pv_forecast_15min | Vec\<f64\> | 15 维 | kW | 光伏 15 分钟预测 |
| | load_forecast_15min | Vec\<f64\> | 15 维 | kW | 负荷 15 分钟预测 |
| **D3-电价** | current_electricity_price | f64 | [0.0, 2.0] | 元/kWh | 当前电价 |
| | next_period_price | f64 | [0.0, 2.0] | 元/kWh | 下时段电价 |
| | price_tariff_id | u8 | {0~3} | 枚举 | 谷/平/峰/尖峰 |
| | peak_price | f64 | [0.0, 2.0] | 元/kWh | 峰值电价（辅助） |
| | valley_price | f64 | [0.0, 2.0] | 元/kWh | 谷值电价（辅助） |
| **D4-需量** | current_demand | f64 | [0.0, 10000.0] | kW | 实时需量 |
| | contract_demand | f64 | [0.0, 10000.0] | kW | 合同需量 |
| | peak_demand_this_month | f64 | [0.0, 10000.0] | kW | 月最大需量 |
| **D5-气象** | solar_irradiance | f64 | [0.0, 1500.0] | W/m^2 | 光照强度 |
| | temperature | f64 | [-20.0, 60.0] | deg C | 环境温度 |
| **D6-调度** | dispatch_p_set | Option\<f64\> | [-1000.0, 1000.0] | kW | 调度有功设定 |
| | dispatch_q_set | Option\<f64\> | [-1000.0, 1000.0] | kVar | 调度无功设定 |
| **D7-实时模块** | q_realtime_margin | f64 | [0.0, 1.0] | - | 实时模块剩余无功容量比例（0=打满，1=空闲） |
| **D8-季节时段** | season_encoding | [f64; 6] | one-hot | - | 季节编码：[灌溉季, 炒茶季, 空调季, 常规季, 保留, 保留] |
| | time_period_encoding | [f64; 2] | one-hot | - | 时段编码：[白天, 夜间] |
| **D9-安全覆盖（v2.10新增，v2.14扩展）** | safety_override_active | bool | {0, 1} | - | 安全覆盖激活标志 |
| | safety_override_reason | Option\<String\> | - | - | 触发原因（voltage_violation/q_exhausted/emergency） |
| | safety_override_p_ref | Option\<f64\> | [-50.0, 50.0] | kW | 强制放电功率 |
| | safety_override_consecutive | u32 | [0, ∞) | - | 连续触发次数（v2.14 新增） |
| | safety_override_ratio | f64 | [0.0, 1.0] | - | 滑动窗口内覆盖比例（v2.14 新增） |
| **D10-概率负荷预测（v2.11新增）** | load_forecast_quantiles | Vec\<f64\> | 15 维 | kW | 分位数负荷预测（P10/P50/P90...） |
| | shock_load_probability | f64 | [0.0, 1.0] | - | 冲击负荷发生概率 |
| | base_load | f64 | [0.0, 1000.0] | kW | 基础负荷（50% 分位数） |

**输入向量维度：** 78 维 = 9(D1) + 30(D2) + 3(D3) + 3(D4) + 2(D5) + 1(D6) + 1(D7) + 8(D8) + 4(D9) + 17(D10)。（D1 10→9，q_realtime_margin 仅计入 D7，不再重复）

> **注：** D9 新增 `safety_override_consecutive` 和 `safety_override_ratio` 字段（2 维），用于精细化 SafetyOverride 惩罚计算。D9 从 2 维扩展至 4 维，输入向量从 76 维扩展至 78 维。

> **历史说明：** PRD v2.10/v2.11 中 59 维的描述不准确，实际应为 61 维（v2.10）和 76 维（v2.11）。

**电压感知 P/Q 协同控制策略（双参数模式）：**

| 场景 | 电压特征 | P 控制 (p_ref) | Q 控制（实时模块闭环） |
|------|----------|---------------------|---------------------|
| 光伏超发 | 电压 > 1.05 p.u. | 充电 (p_ref<0) → 吸收有功消纳光伏 | 实时控制模块根据电压自动调节 |
| 台区季节性负荷 | 电压 < 0.95 p.u. | 放电 (p_ref>0) → 释放有功补充缺口 | 实时控制模块根据电压自动调节 |
| 末端低电压 | 电压 < 0.95 p.u. | 放电 (p_ref>0) — 仅当 Q 裕度不足时 | 实时控制模块优先调节 Q |

> **注：** 双参数模式将 Q 控制完全交给实时控制模块，RL 仅输出 P 控制指令（P_ref + k_droop），实现时间尺度解耦。

### 4.5 完整动作空间表（2 维）

> **符号约定（统一声明）：p_ref > 0 = 放电（向电网注入功率），p_ref < 0 = 充电（从电网吸收功率）。**
> 此约定与实时控制模块、MUPC-AI2 训练管线三方一致。

| 维度 | 字段名 | 类型 | 取值范围 | 单位 | 说明 | 分发路径 |
|------|--------|------|----------|------|------|----------|
| A1 | p_ref | f64 | [-50.0, 50.0] | kW | 有功基准点（负=充电，正=放电） | 核间→实时控制模块 |
| A2 | k_droop | f64 | [0.0, 30.0] | kW/V | 电压-有功下垂系数 | 核间→实时控制模块 |

> **下沉说明：** load_shedding 下沉至 strategy-engine（需量控制策略独立执行），pv_limit 下沉至 strategy-engine（防逆流策略独立执行），confidence 保留在 ModelOutput 中（action_validator 内部校验使用）。AI 引擎仅通过核间通信下发 p_ref + k_droop 至实时控制模块。

### 4.6 ActionOutput 结构体

```rust
/// 强化学习决策输出（2 维动作，v2.15）
///
/// v2.7 双参数模式：p_ref（有功基准）+ k_droop（电压-有功下垂系数）
/// v2.15 精简：load_shedding/pv_limit 下沉至策略引擎，confidence 保留在 ModelOutput 中
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutput {
    /// A1: 有功功率基准点 (kW), [-50.0, 50.0], 负=充电, 正=放电
    pub p_ref: f64,
    /// A2: 电压-有功下垂系数 (kW/V), [0.0, 30.0], 范围由实时控制模块提供
    pub k_droop: f64,
}
```

### 4.6.1 旧版 ActionOutput（legacy）

```rust
/// 动作输出结构体（v1.x 单参数模式，legacy）
/// 仅用于兼容旧模式，正常情况下不使用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutputLegacy {
    pub p_batt_set: f64,      // 废弃，使用 p_ref 替代
    pub load_shedding: f64,   // 保留
    pub pv_limit: f64,        // 保留
    pub confidence: f64,
}
```

### 4.6.2 parse_action_output 双参数解析

从 RKNN Runtime 推理输出的 f32 向量解析为 ActionOutput 结构体，并在解析阶段执行 clamp 限幅。动作空间从 5 维精简为 2 维（p_ref + k_droop）。

```rust
/// 解析 RL 模型原始输出为 ActionOutput（双参数模式，v2.15）
///
/// 输出格式: [p_ref, k_droop]（2 维）
/// v2.15: load_shedding/pv_limit 下沉至策略引擎，不再作为 RL 动作维度
pub fn parse_action_output(raw: &[f32], config: &ActionSpaceConfig) -> Option<ActionOutput> {
    if raw.len() < 2 {
        return None;
    }

    Some(ActionOutput {
        p_ref: (raw[0] as f64).clamp(
            -config.max_batt_discharge_power,
            config.max_batt_charge_power,
        ),
        k_droop: (raw[1] as f64).clamp(0.0, 30.0),
    })
}
```

### 4.7 RLModel 结构体

```rust
/// RL 决策模型
pub struct RLModel {
    config: RlConfig,
    runtime: RknnRuntime,
}

impl RLModel {
    /// 创建 RL 模型
    pub fn new(config: RlConfig) -> Result<Self, AiEngineError>;

    /// 加载 .rknn 模型到 NPU
    pub async fn load(&mut self) -> Result<(), AiEngineError>;

    /// 执行决策
    ///
    /// 输入：78 维融合状态向量（v2.14）
    /// 输出：2 维动作 (p_ref, k_droop)（v2.15）
    pub async fn decide(&self, input_vector: &[f32]) -> Result<ActionOutput, AiEngineError>;

    /// 获取模型类型
    pub fn model_type(&self) -> ModelType;

    /// 获取算法类型
    pub fn algorithm(&self) -> RlAlgorithm;
}
```

### 4.8 ActionValidator — 约束规则校验（4 条规则 ACT-DUAL-01~04，load_shedding/pv_limit/confidence 下沉）

```rust
/// 动作约束校验器
pub struct ActionValidator {
    config: ActionConstraintConfig,
    /// 上一周期的动作输出（用于变化率检测）
    previous_action: Arc<RwLock<Option<ActionOutput>>>,
    /// v2.7 双参数模式：k_droop 范围（由实时控制模块提供）
    droop_range: RwLock<(f64, f64)>,
    /// v2.7 双参数模式：启用 ACT-DUAL-01~05 校验
    dual_mode: bool,
}
```

**4 条双参数校验规则（ACT-DUAL-01 ~ ACT-DUAL-04）：**

v2.15 起 load_shedding 和 pv_limit 不再由 AI 输出，其约束下沉至 strategy-engine（需量控制/防逆流策略内置边界检查）。confidence 保留在 ModelOutput 中用于内部校验。AI 引擎仅校验 p_ref 和 k_droop。

| 规则 ID | 约束条件 | 校验逻辑 |
|---------|----------|----------|
| ACT-DUAL-01 | p_ref 值域约束 | `clamp(p_ref, p_ref_min, p_ref_max)` |
| ACT-DUAL-02 | k_droop 值域约束 | `clamp(k_droop, k_droop_min, k_droop_max)` |
| ACT-DUAL-03 | p_ref 变化率 <= 50kW/步 | `abs(p_ref_new - p_ref_prev) <= config.p_batt_ramp_limit_kw` |
| ACT-DUAL-04 | 调度约束 | `abs(p_ref) <= abs(dispatch_p_set)` (仅 dispatch_p_set 不为 None 时) |

> **LEGACY（v2.4~v2.6）：** 旧版 `validate()` 方法使用已废弃字段 `p_batt_set`/`q_batt_set`。v2.7 起由 `validate_dual()` 完全替代。
> **LEGACY（v2.7~v2.14）：** ACT-05(load_shedding)/ACT-06(pv_limit)/ACT-07(dispatch_p_set) 已随 v2.15 动作空间精简下沉至策略引擎。

```rust
impl ActionValidator {
    /// v2.7 双参数模式校验（ACT-DUAL-01~05，现行版本）
    pub async fn validate_dual(
        &self,
        action: &ActionOutput,
        dispatch_p_set: Option<f64>,
        is_anti_reverse: bool,
        action_space_config: &ActionSpaceConfig,
    ) -> (ActionOutput, Vec<ViolationRecord>) {
        let mut validated = action.clone();
        let mut violations = Vec::new();
        let last = self.previous_action.read().await;

        // ACT-DUAL-01: p_ref 值域约束
        let p_ref_min = -action_space_config.max_batt_discharge_power;
        let p_ref_max = action_space_config.max_batt_charge_power;
        if validated.p_ref < p_ref_min {
            violations.push(ViolationRecord {
                rule: "ACT-DUAL-01",
                field: "p_ref",
                original: action.p_ref,
                clamped: p_ref_min,
            });
            validated.p_ref = p_ref_min;
        } else if validated.p_ref > p_ref_max {
            violations.push(ViolationRecord {
                rule: "ACT-DUAL-01",
                field: "p_ref",
                original: action.p_ref,
                clamped: p_ref_max,
            });
            validated.p_ref = p_ref_max;
        }

        // ACT-DUAL-02: k_droop 值域约束
        let (k_min, k_max) = *self.droop_range.read().unwrap();
        if validated.k_droop < k_min {
            violations.push(ViolationRecord {
                rule: "ACT-DUAL-02",
                field: "k_droop",
                original: action.k_droop,
                clamped: k_min,
            });
            validated.k_droop = k_min;
        } else if validated.k_droop > k_max {
            violations.push(ViolationRecord {
                rule: "ACT-DUAL-02",
                field: "k_droop",
                original: action.k_droop,
                clamped: k_max,
            });
            validated.k_droop = k_max;
        }

        // ACT-DUAL-03: p_ref 变化率约束
        if let Some(ref prev) = *last {
            let delta = (action.p_ref - prev.p_ref).abs();
            if delta > self.config.p_batt_ramp_limit_kw {
                let sign = if action.p_ref > prev.p_ref { 1.0 } else { -1.0 };
                validated.p_ref = prev.p_ref + sign * self.config.p_batt_ramp_limit_kw;
                violations.push(ViolationRecord {
                    rule: "ACT-DUAL-03",
                    field: "p_ref",
                    original: action.p_ref,
                    clamped: validated.p_ref,
                });
            }
        }

        // ACT-DUAL-04: 调度指令权限约束
        if let Some(dp) = dispatch_p_set {
            if validated.p_ref.abs() > dp.abs() {
                let sign = validated.p_ref.signum();
                validated.p_ref = sign * dp.abs();
                violations.push(ViolationRecord {
                    rule: "ACT-DUAL-04",
                    field: "p_ref",
                    original: action.p_ref,
                    clamped: validated.p_ref,
                });
            }
        }

        // v2.15: load_shedding/pv_limit/confidence 已从 ActionOutput 移除
        // 其约束下沉至 strategy-engine（需量控制/防逆流策略内置边界检查）

        *self.previous_action.write().await = Some(validated.clone());
        (validated, violations)
    }

    /// 更新 k_droop 范围（由 intercore 从实时控制模块获取后调用）
    pub fn update_droop_range(&self, k_min: f64, k_max: f64) {
        let mut range = self.droop_range.write().unwrap();
        *range = (k_min, k_max);
        tracing::debug!("Updated k_droop range: [{}, {}]", k_min, k_max);
    }

    /// 获取当前 k_droop 范围
    pub fn get_droop_range(&self) -> (f64, f64) {
        *self.droop_range.read().unwrap()
    }
}

/// 约束违规记录
#[derive(Debug, Clone)]
pub struct ViolationRecord {
    pub rule: &'static str,
    pub field: &'static str,
    pub original: f64,
    pub clamped: f64,
}
```

### 4.9 场景切换平滑过渡（R3）

#### 4.9.1 平滑过渡配置

```rust
// 平滑过渡配置
#[derive(Debug, Clone)]
pub struct TransitionConfig {
    pub transition_steps: usize, // 默认 10
}

// 平滑过渡状态
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionState {
    Idle,
    InProgress,
    Completed,
}
```

#### 4.9.2 SmoothSceneTransition 结构体

```rust
/// 平滑过渡器（嵌入 ModeSelector）
pub struct SmoothSceneTransition {
    config: TransitionConfig,
    current_weights: Option<Vec<f32>>,
    target_weights: Option<Vec<f32>>,
    step_counter: usize,
    state: TransitionState,
    /// 事件发布器（通知 ModelManager 权重更新）
    weight_tx: broadcast::Sender<WeightUpdateEvent>,
}

/// 权重更新事件
#[derive(Debug, Clone)]
pub struct WeightUpdateEvent {
    pub blended_weights: Vec<f32>,
    pub step: usize,
    pub total_steps: usize,
}
```

#### 4.9.3 线性插值数学定义

```
alpha = step_counter / transition_steps
weight_i = (1 - alpha) * current_weight_i + alpha * target_weight_i
```

- `step = 0` 时：`weight = current_weights`
- `step = steps` 时：`weight = target_weights`
- 线性插值确保过渡平滑，避免权重突变

#### 4.9.4 与 ModeSelector 集成

```rust
pub async fn switch(&self, new_mode: RunningMode, source: SwitchSource) -> Result<RunningMode, AiEngineError> {
    let mut current = self.current_mode.lock().await;
    let previous = *current;

    if previous == new_mode {
        return Ok(previous);
    }

    // 触发平滑过渡
    if let Some(ref mut transition) = self.smooth_transition {
        transition.on_scene_switch(previous, new_mode);
    }

    // ... 后续现有逻辑（模型热切换、持久化、事件发布）不变 ...
}
```

#### 4.9.5 错误处理

| 错误类型 | 触发条件 | 处理策略 |
|----------|----------|----------|
| `WeightsNotAvailable` | 场景切换前调用 current_weights | 返回默认权重或错误 |
| `TransitionNotStarted` | 切换已完成后继续调用 | 直接返回目标权重 |

#### 4.9.6 测试策略

| 测试名称 | 验收条件 | 测试方法 |
|----------|----------|----------|
| `test_transition_steps_configurable` | 步数可配置 | 构造 5 步，验证 step 5 后 state = Completed |
| `test_linear_interpolation_first_last` | 首尾步正确 | step 0 返回 current_weights，step 10 返回 target_weights |
| `test_interpolation_middle` | 中间步线性 | step 5 时每权重 = (current + target) / 2 |
| `test_transition_auto_stop` | 完成后自动停止插值 | step > steps 时返回目标权重 |
| `test_no_control_jump` | 梯度 < 5% | 验证相邻步权重变化 < 5% |

## 5. 奖励函数计算模块

### 5.1 功能概述

RewardCalculator 根据当前运行场景，选择对应的奖励函数公式计算即时奖励值。奖励值用于在线微调阶段模型权重更新，以及 Web UI 展示决策质量。

### 5.2 RewardCalculator 结构体

```rust
/// 奖励函数计算器
pub struct RewardCalculator {
    /// 场景权重配置
    weights: SceneWeights,
    /// 碳排因子 (kg CO2/kWh)
    carbon_emission_factor: f64,
    /// 需量费率 (元/kW)
    demand_penalty_rate: f64,
    /// 电池退化系数
    battery_degradation_alpha: f64,
}

impl RewardCalculator {
    /// 创建奖励计算器
    pub fn new(config: SceneWeights) -> Self;

    /// 根据运行场景计算奖励值
    pub fn calculate(
        &self,
        mode: RunningMode,
        action: &ActionOutput,
        state: &FusedSystemState,
    ) -> f64;
}
```

### 5.3 SCENE-01：台区季节性负荷模式 (MODE-01)

**优化目标：** 最大化光伏消纳 + 防止变压器过载 + 电池寿命保护 + P-Q 协同优化 + 安全覆盖感知

> **核心变更（安全覆盖惩罚）：**
> - 新增 R_safety_override 惩罚项，当 safety_override_active=true 时触发
> - AI 引擎感知被实时控制模块覆盖事件，学习避免触发覆盖的策略

> **核心变更（RobustnessManager 集成）：**
> - dispatch_ai_decision 前进行异常检测
> - 存在异常时使用应急策略（不经过 RL 模型）

> **核心变更（P-Q 协同度奖励）：**
> - 移除"电压硬惩罚"P_voltage_deviation，改为"行为奖励"R_PQ_coordination
> - AI 仅控制 P（p_ref），但需感知 Q 裕度做最优决策
> - 核心原则：Q 有裕度时"偷懒"省电池；Q 饱和时正确出手（低压放电/高压充电）
> - 弃光场景差异化：高电压时检查 AI 动作方向而非简单置零
> - 新增下垂系数平滑惩罚 R_smooth，防止 k_droop 极大化导致系统震荡

> **分层架构原则（继续适用）：**
> - AI 仅在实时模块无功耗尽时才对电压偏差负责（q_realtime_margin <= 10% + 越限连续 2 步）
> - 实时模块有裕度时，电压问题由实时模块自行处理，AI 不因"旁观"被惩罚
> - 自适应损耗系数 α(s) ∈ {1.0, 0.2, 3.0} 区分"常规调度"与"应急处置"的电池损耗价值差异

**奖励公式：**

```
R_agri = w1 * R_pv_consumption          // 弃光奖励（含差异化电压处理）
         - α(s) * w2 * P_battery_degradation   // 自适应损耗系数
         - w3 * P_transformer_overload
         + w4 * R_PQ_coordination              // P-Q 协同度奖励（v2.8 新增）
         - w5 * R_ramp
         - w6 * R_voltage_slope
         - w7 * R_smooth                        // 下垂系数平滑惩罚（v2.8 新增）
         - w8 * R_safety_override               // 安全覆盖惩罚（v2.10 新增）
```

**P-Q 协同度奖励 R_PQ_coordination：**

当 |V_deviation| > 5% 时，根据 Q 裕度和 AI 动作方向计算奖励：

```
// Q 有裕度（q_margin > 10%）：AI 最优解是"偷懒"省电池
if q_margin > Q_THRESHOLD:
    if |p_ref| < P_THRESHOLD (省电策略):
        R_PQ_coordination = +50.0  // 大额奖励
    else:
        R_PQ_coordination = -5.0   // 轻微惩罚（强行出力浪费电池）

// Q 已饱和（q_margin <= 10%）：AI 必须正确出手
else:
    if v_low && p_ref < 0:                    // 低电压 + 放电（正确）
        R_PQ_coordination = +50.0
    elif v_high && p_ref > 0:                  // 高电压 + 充电（正确）
        R_PQ_coordination = +50.0
    elif v_low && p_ref >= 0:                  // 低电压 + 不放电（失职）
        R_PQ_coordination = -30.0
    elif v_high && p_ref <= 0:                 // 高电压 + 不充电（失职）
        R_PQ_coordination = -30.0
    else:
        R_PQ_coordination = 0.0
```

其中：`v_low = (v_avg < 0.95)`，`v_high = (v_avg > 1.05)`，`P_THRESHOLD = 5.0 kW`，`Q_THRESHOLD = 0.10`

**弃光奖励差异化：**

```
// 高电压时（v_avg >= 1.05），检查 AI 动作方向而非简单置零
if v_avg >= 1.05:
    if p_ref < 0:                           // 充电消纳光伏
        R_pv_consumption = min(P_pv_self_consume / P_pv_total, 1.0) * 100.0
    else:                                    // 反而在放电
        R_pv_consumption = -20.0            // 严厉惩罚
else:
    R_pv_consumption = min(P_pv_self_consume / P_pv_total, 1.0) * 100.0
```

**下垂系数平滑惩罚 R_smooth：**

```
R_smooth = -|Δk_droop| - λ * max(0, k_droop - K_MAX)
```

- `Δk_droop = k_droop_t - k_droop_{t-1}`：防止 AI 频繁调整 k_droop
- `K_MAX`：k_droop 惩罚触发阈值（默认 30.0 kW/V）
- `λ`：超限惩罚系数（默认 10.0）

**子项定义：**

```
R_pv_consumption       = 按差异化逻辑计算（见上）
α(s)                   = 3.0   // SOC 极低保护：battery_soc < SOC_CRITICAL (10%)
                        = 0.2   // 电压支撑模式：q_realtime_margin <= 10% 且越限 >= 2 步
                        = 1.0   // 常规调度
P_battery_degradation  = α(s) * (|P_batt| / BATTERY_CAPACITY_KWH)²   # C-rate² × α(s)
P_transformer_overload = Quadratic(L_trafo, start=75%)                 # 见 4.5 节
R_PQ_coordination      = 按 Q 裕度和动作方向计算（见上）
R_ramp                 = λ * |P_batt_t - P_batt_{t-1}| / BATTERY_CAPACITY_KWH
R_voltage_slope        = |V_avg_t - V_avg_{t-1}|
R_smooth               = -|Δk_droop| - λ * max(0, k_droop - K_MAX)
R_safety_override      = 安全覆盖惩罚（v2.10 新增，见下）

其中 v_avg = (voltage_phase_a + voltage_phase_b + voltage_phase_c) / 3.0
```

**安全覆盖惩罚 R_safety_override：**

v2.14 采用分层计算策略，结合滑动窗口统计信息。v2.15 删除 `match reason` 分支（因 D9 无 reason_code 字段）：

```rust
if state.safety_override_active {
    if state.safety_override_consecutive < 10 {
        // 样本不足：使用固定中等惩罚（v2.15：删除 reason 差异化）
        -3.33
    } else {
        // 样本充足：比例 + 连续次数惩罚，归一化至 [-1, 0]
        let ratio_penalty = -5.0 * state.safety_override_ratio;
        let consecutive_penalty = -10.0 * (state.safety_override_consecutive as f64 / 10.0).clamp(0.0, 1.0);
        (ratio_penalty + consecutive_penalty) / 15.0
    }
} else {
    0.0
}
```

**修正说明：** D9 字段表已无 `safety_override_reason_code`（仅 4 维：active/p_ref/consecutive/ratio），样本不足时无 reason 数据可用，故删除 v2.13 引入的 reason 差异化惩罚分支。固定惩罚 -3.33 取原 voltage_violation 档位（最常见原因）。

**系数说明：**

| 系数 | 值 | 说明 |
|------|------|------|
| k_override | 5.0 | 覆盖比例惩罚系数 |
| k_consecutive | 10.0 | 连续触发次数惩罚系数 |
| min_sample_threshold | 10 | 最小样本阈值 |
| norm_divisor | 15.0 | 归一化除数 |
| cold_start_penalty | 3.33 | 样本不足时固定惩罚（v2.15 新增，替代原 reason 差异化） |

**互斥惩罚逻辑：**

当 `safety_override_active = true` 时，跳过该步的 **P-Q 协同度惩罚**，避免同一次事件双重惩罚：

```rust
let r_pq = if state.safety_override_active {
    0.0  // 互斥：SafetyOverride 事件不重复惩罚
} else {
    self.calc_pq_coordination(state, action.p_ref)
};
```

**权重表：**

| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 光伏消纳奖励（含差异化电压处理） | [0.0, 3.0] |
| w2 | 0.5 | 电池损耗惩罚（C-rate² × α(s)） | [0.0, 2.0] |
| w3 | 2.0 | 变压器过载惩罚 | [0.0, 5.0] |
| w4 | 1.0 | P-Q 协同度奖励权重（v2.8 新增） | [0.0, 5.0] |
| w5 | 0.5 | 功率变化率惩罚 | [0.0, 2.0] |
| w6 | 0.5 | 电压变化斜率惩罚 | [0.0, 2.0] |
| w7 | 0.5 | 下垂系数平滑惩罚权重（v2.8 新增） | [0.0, 2.0] |
| w8 | 1.0 | 安全覆盖惩罚权重（v2.10 新增） | [0.0, 5.0] |

**Rust 代码实现：**

```rust
/// SCENE-01: 台区季节性负荷模式 v2.8
fn calc_agri_v2_8(&self, state: &FusedSystemState, action: &ActionOutput, prev_k_droop: f64) -> f64 {
    let w = &self.weights.seasonal_load_management;

    // 1. 弃光奖励（含差异化电压处理）
    let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
    let r_pv = if v_avg >= self.voltage_high_limit {
        if action.p_ref < 0.0 {
            // 充电消纳光伏
            (state.pv_power.max(0.0) / (state.pv_power.max(0.0) + state.grid_power.max(0.0) + 1e-6))
                .min(1.0) * 100.0
        } else {
            -20.0 // 高电压时反而放电，严厉惩罚
        }
    } else {
        (state.pv_power.max(0.0) / (state.pv_power.max(0.0) + state.grid_power.max(0.0) + 1e-6))
            .min(1.0) * 100.0
    };

    // 2. 自适应损耗系数 α(s)
    let alpha = self.compute_alpha(state);

    // 3. 电池损耗
    let c_rate = state.battery_power.abs() / self.battery_capacity_kwh;
    let p_batt_deg = alpha * c_rate * c_rate;

    // 4. 变压器过载
    let p_trafo = self.overload_penalty(state.transformer_load);

    // 5. P-Q 协同度奖励（v2.8 新增，替代电压惩罚）
    let r_pq = self.calc_pq_coordination(state, action.p_ref);

    // 6. 变化率惩罚
    let r_ramp = w[4] * (action.p_ref - self.last_p_ref()).abs() / self.battery_capacity_kwh;

    // 7. 电压变化斜率惩罚
    let prev_v = *self.last_voltage.read().unwrap();
    let r_voltage_slope = (v_avg - prev_v).abs();

    // 8. 下垂系数平滑惩罚（v2.8 新增）
    let r_smooth = self.calc_smooth_penalty(action.k_droop, prev_k_droop);

    w[0] * r_pv - w[1] * p_batt_deg - w[2] * p_trafo + w[3] * r_pq - w[4] * r_ramp - w[5] * r_voltage_slope - w[6] * r_smooth
}

/// P-Q 协同度奖励（v2.8 新增）
fn calc_pq_coordination(&self, state: &FusedSystemState, p_ref: f64) -> f64 {
    let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
    let dev = (v_avg - 1.0).abs();

    // 死区内，无协同度问题
    if dev <= 0.05 {
        return 0.0;
    }

    let q_margin = state.q_realtime_margin;
    let v_low = v_avg < 0.95;
    let v_high = v_avg > 1.05;
    const P_THRESHOLD: f64 = 5.0; // kW
    const Q_THRESHOLD: f64 = 0.10;

    // Q 有裕度：最优解是"偷懒"省电池
    if q_margin > Q_THRESHOLD {
        if p_ref.abs() < P_THRESHOLD {
            return 50.0; // 省电策略，大额奖励
        } else {
            return -5.0; // 强行出力，轻微惩罚
        }
    }

    // Q 已饱和：AI 必须正确出手
    if v_low && p_ref < 0.0 {
        return 50.0;  // 低电压 + 放电（正确）
    } else if v_high && p_ref > 0.0 {
        return 50.0;  // 高电压 + 充电（正确）
    } else if v_low && p_ref >= 0.0 {
        return -30.0; // 低电压 + 不放电（失职）
    } else if v_high && p_ref <= 0.0 {
        return -30.0; // 高电压 + 不充电（失职）
    }
    0.0
}

/// 下垂系数平滑惩罚（v2.8 新增）
fn calc_smooth_penalty(&self, k_droop: f64, prev_k_droop: f64) -> f64 {
    const K_MAX: f64 = 30.0; // kW/V
    const LAMBDA: f64 = 10.0;
    let delta = (k_droop - prev_k_droop).abs();
    let excess = (k_droop - K_MAX).max(0.0);
    delta + LAMBDA * excess
}

/// 计算自适应损耗系数 α(s)
fn compute_alpha(&self, state: &FusedSystemState) -> f64 {
    if state.battery_soc < self.soc_critical {
        return 3.0;
    }
    let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
    let v_dev = (v_avg - 1.0).abs();
    let in_violation = v_dev > 0.05 && self.voltage_violation_count() >= 2;
    if state.q_realtime_margin <= self.q_margin_threshold && in_violation {
        return 0.2;
    }
    1.0
}
```

**RewardCalculator 结构体变更：**

```rust
pub struct RewardCalculator {
    weights: SceneWeights,
    carbon_emission_factor: f64,
    demand_penalty_rate: f64,
    battery_degradation_alpha: f64,
    battery_capacity_kwh: f64,
    last_p_ref: RwLock<f64>,           // 上一周期 p_ref
    last_voltage: RwLock<f64>,          // 上一周期平均电压
    last_k_droop: RwLock<f64>,          // 上一周期 k_droop（v2.8 新增）
    voltage_violation_count: AtomicU32,
    q_margin_threshold: f64,
    voltage_high_limit: f64,
    soc_critical: f64,
    voltage_penalty_high: f64,
    voltage_penalty_low: f64,
}
```

### 5.4 SCENE-B1：工商业模式-自主套利 (MODE-02)

**优化目标：** 最大化峰谷电价差收益，最小化电池损耗。

**奖励公式：**

```
R_arbitrage = w1 * R_price_spread - w2 * P_battery_degradation

R_price_spread         = (price_current - price_avg) * p_ref * conversion_factor
P_battery_degradation  = β · (|p_ref| / E_battery_total)²   # C-rate² 应力模型（v2.15）
```

**权重表：**

| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 电价差收益权重 | [0.0, 3.0] |
| w2 | 1.0 | 电池损耗惩罚权重 | [0.0, 3.0] |

```rust
fn reward_commercial_arbitrage(
    &self, state: &FusedSystemState, action: &ActionOutput
) -> f64 {
    let w = &self.weights.commercial_arbitrage;
    let avg_price = (state.peak_price + state.valley_price) / 2.0;
    let spread = (state.current_electricity_price - avg_price) * action.p_ref * 0.001;
    let r_spread = spread * 100.0;
    // C-rate² 应力模型（v2.15，对齐 MODE-01）
    let c_rate = action.p_ref.abs() / BATTERY_CAPACITY_KWH;
    let p_deg = c_rate * c_rate * 100.0;
    w[0] * r_spread - w[1] * p_deg
}
```

### 5.5 SCENE-B2：工商业模式-需量控制 (MODE-03)

**优化目标：** 减免需量罚金。

**奖励公式：**

```
R_demand = w1 * R_demand_penalty_avoidance - w2 * P_comfort_loss

R_demand_penalty_avoidance = max(0, D_peak_baseline - D_peak_actual) * penalty_rate
P_comfort_loss             = gamma * P_load_shed * delta_t * price_loss
```

**权重表：**

| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 需量罚金减免权重 | [0.0, 3.0] |
| w2 | 0.5 | 舒适度损失惩罚权重 | [0.0, 3.0] |

```rust
fn reward_demand_control(
    &self, state: &FusedSystemState, action: &ActionOutput
) -> f64 {
    let w = &self.weights.demand_control;
    let demand_saved = (state.contract_demand - state.current_demand).max(0.0);
    let r_penalty_avoid = demand_saved * self.demand_penalty_rate;
    let p_comfort = action.load_shedding * 0.5; // 每切负荷惩罚
    w[0] * r_penalty_avoid - w[1] * p_comfort
}
```

### 5.6 SCENE-B3：工商业模式-虚拟电厂 (MODE-04)

**优化目标：** 最大化辅助服务收益，最大化响应精度。

**奖励公式：**

```
R_vpp = w1 * R_ancillary_service + w2 * R_response_accuracy - w3 * P_deadline_deviation

R_ancillary_service  = P_regulation_capacity * capacity_price + P_regulation_mileage * mileage_price
R_response_accuracy  = 100 * max(0, 1 - abs(P_actual - P_target) / P_target_range)
P_deadline_deviation = delta_t_response / T_allowed * 100
```

**权重表：**

| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 辅助服务收益权重 | [0.0, 3.0] |
| w2 | 2.0 | 响应精度权重（VPP 考核重点） | [0.0, 5.0] |
| w3 | 1.0 | 响应延迟惩罚权重 | [0.0, 3.0] |

```rust
fn reward_virtual_power_plant(
    &self, state: &FusedSystemState, action: &ActionOutput
) -> f64 {
    let w = &self.weights.virtual_power_plant;
    match state.dispatch_p_set {
        Some(p_target) => {
            let p_actual = action.p_ref;
            let r_accuracy = 100.0 * (1.0 - (p_actual - p_target).abs() / 100.0).max(0.0);
            let p_deadline = 0.0; // 延迟由外部计时器注入
            w[0] * p_target.abs() * 0.01 + w[1] * r_accuracy - w[2] * p_deadline
        }
        None => 0.0, // 无 VPP 调度指令时奖励为 0
    }
}
```

### 5.7 SCENE-B5：工商业模式-极致绿色 (MODE-05)

**优化目标：** 最大化绿电消纳比例，最小化碳排放。

**奖励公式：**

```
R_green = w1 * R_green_consumption + w2 * R_carbon_reduction

R_green_consumption = 100 * E_green_self_consume / E_total_consume
R_carbon_reduction   = 100 * (C_baseline - C_actual) / C_baseline
```

**权重表：**

| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 绿电消纳比例权重 | [0.0, 3.0] |
| w2 | 1.0 | 碳减排量权重 | [0.0, 3.0] |

```rust
fn reward_ultra_green(
    &self, state: &FusedSystemState
) -> f64 {
    let w = &self.weights.ultra_green;
    let total_consume = state.load_power.max(1e-6);
    let green_consume = state.pv_power.max(0.0);
    let r_green = 100.0 * (green_consume / total_consume).min(1.0);
    let c_baseline = 0.581; // 中国电网平均排放因子 kg CO2/kWh
    let c_actual = state.grid_power.max(0.0) * c_baseline / 1000.0;
    let r_carbon = if c_baseline > 0.0 {
        100.0 * (c_baseline - c_actual).max(0.0) / c_baseline
    } else {
        0.0
    };
    w[0] * r_green + w[1] * r_carbon
}
```

### 5.8 SceneWeights 映射表

| 场景 | w1(op1) | w2(op2) | w3(op3) | w4(op4) | w5(op5) |
|------|---------|---------|---------|---------|---------|
| 台区季节性负荷 MODE-01 | 1.0 (光伏消纳) | 0.5 (电池损耗) | 2.0 (变压器) | 1.0 (电压质量) | 0.5 (功率变化率) |
| 自主套利 MODE-02 | 1.0 (电价收益) | 1.0 (电池损耗) | - | - | - |
| 需量控制 MODE-03 | 1.0 (需量减免) | 0.5 (舒适损失) | - | - | - |
| VPP MODE-04 | 1.0 (辅助收益) | 2.0 (响应精度) | 1.0 (延迟惩罚) | - | - |
| 极致绿色 MODE-05 | 1.0 (绿电消纳) | 1.0 (碳减排) | - | - | - |

权重映射查找逻辑：

```rust
impl SceneWeights {
    /// 根据运行场景返回对应的权重数组
    pub fn lookup(&self, mode: RunningMode) -> &[f64] {
        match mode {
            RunningMode::SeasonalLoadManagement => &self.agricultural_irrigation[..3],
            RunningMode::CommercialArbitrage => &self.commercial_arbitrage[..2],
            RunningMode::DemandControl => &self.demand_control[..2],
            RunningMode::VirtualPowerPlant => &self.virtual_power_plant[..3],
            RunningMode::UltraGreen => &self.ultra_green[..2],
        }
    }
}
```

### 5.9 折扣累积奖励机制（R2）

#### 5.9.1 折扣累积奖励配置

```rust
// 折扣累积奖励配置
#[derive(Debug, Clone)]
pub struct DiscountedConfig {
    pub gamma: f32,        // 折扣因子，默认 0.99，范围 [0.9, 0.999]
    pub buffer_size: usize, // 缓冲区大小，默认 1000
}
```

#### 5.9.2 DiscountedAccumulator 结构体

```rust
/// 折扣累积奖励计算器（嵌入 RewardCalculator）
pub struct DiscountedAccumulator {
    gamma: f32,
    buffer: Vec<f32>,
    buffer_size: usize,
    cumulative: RwLock<f32>,
}

impl DiscountedAccumulator {
    /// 创建折扣累积器
    /// - gamma: 折扣因子，范围 [0.9, 0.999]
    /// - 返回 ConfigError 若 gamma 超出范围或 buffer_size 为 0
    pub fn new(gamma: f32, buffer_size: usize) -> Result<Self, ConfigError>;
    pub fn push(&mut self, reward: f32);
    pub fn discounted_sum(&self) -> f32;
    pub fn reset(&mut self);
}
```

#### 5.9.3 折扣因子数学定义

折扣累积奖励（Discounted Cumulative Reward）：

```
D_t = Σ_{i=0}^{T} γ^i * r_{T-i}
    = r_t + γ * r_{t-1} + γ² * r_{t-2} + ... + γ^T * r_0
```

- `γ = 0.99`：每步向后追溯，100 步前奖励权重约为 `0.99^100 ≈ 0.366`
- `γ = 0.9`：短期记忆，100 步前奖励权重约为 `0.9^100 ≈ 0.000027`

#### 5.9.4 与现有奖励函数正交性

即时奖励计算不受影响：

```rust
pub fn calculate(&self, mode: RunningMode, action: &ActionOutput, state: &FusedSystemState) -> f64 {
    let immediate = match mode { /* ... */ }; // 现有逻辑不变
    immediate // 返回即时奖励
}

pub fn calculate_discounted(&self, current_reward: f32) -> f32 {
    self.discounted_accumulator.discounted_sum()
}
```

#### 5.9.5 错误处理

| 错误类型 | 触发条件 | 处理策略 |
|----------|----------|----------|
| `GammaOutOfRange` | gamma 不在 [0.9, 0.999] | 构造函数返回错误 |
| `BufferOverflow` | buffer_size = 0 | 构造函数返回错误 |

#### 5.9.6 测试策略

| 测试名称 | 验收条件 | 测试方法 |
|----------|----------|----------|
| `test_gamma_range_valid` | gamma ∈ [0.9, 0.999] 正常工作 | 边界测试 gamma = 0.9, 0.999 |
| `test_gamma_out_of_range_reject` | gamma ∉ [0.9, 0.999] 拒绝 | gamma = 0.8, 1.0 应报错 |
| `test_discounted_buffer_size` | 缓冲区大小固定 1000 | 填充 1500 个奖励，验证移除最旧 |
| `test_discounted_100_steps` | gamma=0.99 时 100 步前权重约 0.366 | 数学验证：0.99^100 ≈ 0.366 |
| `test_immediate_reward_unchanged` | 即时奖励计算结果不变 | 对比 new_with_discount 前后的即时奖励 |

### 5.10 冲击负荷概率预测

#### 5.10.1 需求描述

当前负荷预测为确定性格点预测（单一均值），无法捕捉概率分布特征。农网台区灌溉水泵等冲击负荷具有随机性，可能导致需量控制失效。

#### 5.10.2 新增文件

| 文件路径 | 职责 |
|---------|------|
| `ai-engine/src/load_covariates.rs` | 负荷协变量结构体（温度、日期类型、灌溉季、小时） |
| `ai-engine/src/weather_service.rs` | 气象数据服务 trait（PLF-05 数据源定义） |

#### 5.10.3 架构设计

```
LstmModel
  ├─ predict()           （原有确定性预测）
  ├─ predict_quantiles()  （v2.11 分位数预测，v2.16 重构为 15 步）
  └─ erfc()               （正态分布 CDF 近似）
        ↓
ProbabilisticLoadOutput (v2.16 重构)
  ├─ quantile_steps: Vec<StepQuantiles>  (15 步 × [P10, P50, P90])
  ├─ base_load (第 1 步 P50)
  ├─ shock_probability
  └─ confidence (基于 P50/P90 间距)
        ↓
  ┌────┴────┐
  ↓         ↓
RewardCalculator  FusedSystemState (D10)
calc_demand_with_   load_forecast_quantiles
uncertainty()       (15 维，v2.16 接通数据流)
```

#### 5.10.4 详细设计

**LoadCovariates：**

```rust
pub struct LoadCovariates {
    pub temperature: f32,        // 温度（摄氏度）
    pub date_type: u8,           // 0=工作日, 1=周末, 2=节假日
    pub is_irrigation_season: bool, // 是否灌溉季
    pub hour: u8,               // 小时（0-23）
}
```

**ProbabilisticLoadOutput：**

```rust
/// 单分位数预测（v2.11，向后兼容保留）
pub struct QuantilePrediction {
    pub quantile: f32,  // 分位数（0.0 ~ 1.0）
    pub value: f32,      // 预测值 (kW)
}

/// 单步分位数预测（v2.16 新增）
pub struct StepQuantiles {
    pub step_index: usize,   // 步索引 0..14
    pub p10: f32,            // P10 分位数预测值
    pub p50: f32,            // P50 分位数预测值
    pub p90: f32,            // P90 分位数预测值
}

/// 概率负荷预测输出（v2.11 新增，v2.16 重构为 15 步结构）
pub struct ProbabilisticLoadOutput {
    pub timestamp: i64,
    pub quantile_steps: Vec<StepQuantiles>,  // 15 个未来时间步（v2.16）
    pub base_load: f32,                       // 第 1 步 P50（向后兼容）
    pub shock_probability: f64,               // 冲击负荷概率
    pub confidence: f64,                      // 基于 P50/P90 间距
}
```

**LstmModel 扩展方法：**

```rust
pub async fn predict_quantiles(
    &self,
    input: &LstmInput,
    covariates: &LoadCovariates,
) -> Result<ProbabilisticLoadOutput, AiEngineError> {
    // 1. 获取多分位数预测值
    let quantile_values = self.predict_multi_quantile(input, covariates).await?;
    // 2. 提取基础负荷（P50）
    let base_load = quantile_values.iter().find(|q| (q.quantile - 0.5).abs() < 0.01)...
    // 3. 计算冲击负荷概率
    let shock_probability = self.calculate_shock_probability(base_load, high_quantile);
    // 4. 计算置信度
    let confidence = self.calculate_confidence(&quantile_values);
    Ok(ProbabilisticLoadOutput { ... })
}

fn calculate_shock_probability(&self, base_load: f32, high_quantile: f32) -> f64 {
    // P(shock) = 1 - Φ((shock_threshold - μ) / σ)
    // 其中 std ≈ (P90 - P50) / 1.28
    let spread = (high_quantile - base_load).max(1e-6);
    let std_approx = spread / 1.28;
    let shock_threshold = base_load + 2.0 * std_approx;
    let z_score = (shock_threshold - base_load) / std_approx.max(1e-6);
    0.5 * Self::erfc(z_score / 1.41421356) as f64
}
```

**WeatherService Trait（PLF-05）：**

```rust
pub trait WeatherService: Send + Sync {
    fn get_current_temperature(&self) -> Result<f32, AiEngineError>;
    fn get_temperature_forecast(&self, hours_ahead: u32) -> Result<Vec<f32>, AiEngineError> { Ok(Vec::new()) }
}
```

**FusedSystemState 扩展：**

```rust
// v2.11 新增字段
pub load_forecast_quantiles: Vec<f64>,  // 分位数负荷预测
pub shock_load_probability: f64,         // 冲击负荷概率
pub base_load: f64,                     // 基础负荷（P50）
```

**RewardCalculator 扩展：**

```rust
fn calc_demand_with_uncertainty(
    &self,
    action: &ActionOutput,
    state: &FusedSystemState,
    load_forecast: &ProbabilisticLoadOutput,
) -> f64 {
    // 风险感知调整：考虑冲击负荷概率，预留额外安全裕度
    let high_quantile = load_forecast.quantiles.iter()
        .find(|q| (q.quantile - 0.9).abs() < 0.01)
        .map(|q| q.value as f64)
        .unwrap_or(load_forecast.base_load as f64);

    let risk_adjusted_demand = state.current_demand + 2.0 * (high_quantile - load_forecast.base_load as f64);
    let risk_margin = if risk_adjusted_demand > state.contract_demand * 0.95 {
        -20.0 * ((risk_adjusted_demand - state.contract_demand * 0.95) / state.contract_demand)
    } else { 0.0 };

    let r_avoid = (state.contract_demand - state.current_demand).max(0.0) * self.demand_penalty_rate;
    let p_comfort = action.load_shedding * 0.5;
    w[0] * (r_avoid + risk_margin) - w[1] * p_comfort
}
```

#### 5.10.5 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| LSTM 模型不支持多输出 | 回退到确定性预测（50% 分位数）|
| 分位数预测值异常 | 校验范围 [0, max_capacity]，超范围裁剪 |
| 协变量缺失 | 使用默认值（温度=25°C，日期类型=工作日）|

#### 5.10.6 测试策略

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| PLF-01 | 分位数输出 P10 < P50 < P90 | 单元测试 |
| PLF-02 | 冲击概率计算 | 已知分布验证概率值 |
| PLF-04 | 风险奖励计算 | 风险惩罚正确应用 |
| PLF-07 | FusedSystemState 存储 | 字段正确填充 |

### 5.11 变压器过载分段惩罚（R-04）

#### 5.11.1 需求描述

当前变压器过载惩罚使用二次函数，不够精细。当负载率在 75%~90% 时为安全区，90%~100% 时风险上升，>100% 时需要硬惩罚。

#### 5.11.2 分段惩罚函数定义

```rust
/// 变压器过载分段惩罚（v2.12 R-04）
///
/// 分段逻辑：
///   L < 0.75:          0.0                    // 安全区
///   0.75 <= L < 0.90:  linear 0~10           // 线性增长
///   0.90 <= L < 1.00:  exponential 10~50     // 指数增长
///   L >= 1.00:          100.0                 // 硬惩罚
fn overload_penalty_piecewise(&self, load: f64) -> f64 {
    if load < 0.75 {
        0.0
    } else if load < 0.90 {
        (load - 0.75) / 0.15 * 10.0  // 线性：0~10
    } else if load < 1.00 {
        let excess = (load - 0.90) / 0.10;
        10.0 + excess * excess * 40.0  // 指数：10~50
    } else {
        100.0  // 硬惩罚
    }
}
```

#### 5.11.3 验收标准

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| TO-01 | L=0.70 时惩罚为 0 | 单元测试 |
| TO-02 | L=0.90 时惩罚为 10 | 单元测试 |
| TO-03 | L=1.00 时惩罚为 50 | 单元测试 |
| TO-04 | L=1.05 时惩罚为 100 | 单元测试 |

#### 5.11.4 影响文件

- `mupc/crates/ai-engine/src/reward_calculator.rs`

### 5.12 电压斜率惩罚动态权重（R-05）

#### 5.12.1 需求描述

当前 `w6` 电压变化斜率惩罚权重固定，电压偏差越大应权重越高。

#### 5.12.2 动态权重公式

```rust
/// 电压斜率惩罚动态权重（v2.12 R-05）
///
/// w6(v) = base_w6 × (1.0 + k × |ΔV|)
///
/// 参数：
///   base_w6: 基础权重（默认 0.5）
///   k: 放大系数（默认 2.0，可配置）
///   |ΔV|: 电压变化量
fn dynamic_voltage_slope_weight(&self, delta_v: f64) -> f64 {
    let base = self.base_w6_voltage_slope;  // 默认 0.5
    let k = self.voltage_slope_k;           // 默认 2.0
    base * (1.0 + k * delta_v.abs())
}
```

#### 5.12.3 验收标准

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| DV-01 | ΔV=0 时 w6 = base_w6 | 单元测试 |
| DV-02 | ΔV=0.05 时 w6 = base × (1.0 + k × 0.05) | 数学验证 |
| DV-03 | k 值可配置，范围 [0.0, 5.0] | 配置测试 |

#### 5.12.4 影响文件

- `mupc/crates/ai-engine/src/reward_calculator.rs`

### 5.13 冲击负荷响应奖励（R-06）

#### 5.13.1 需求描述

当前缺失冲击负荷响应奖励，需量控制鲁棒性不足。结合 v2.11 分位数预测，引入基于风险感知的冲击负荷响应奖励。

#### 5.13.2 冲击负荷响应奖励公式

```rust
/// 冲击负荷响应奖励（v2.12 R-06）
///
/// R_shock = w_shock × load_shedding / max_load_shedding
///           - λ × response_time / max_response_time
///
/// 条件：当 P90 - P50 > threshold（冲击负荷检测）
fn shock_response_reward(
    &self,
    load_shedding: f64,
    response_time: f64,
    p90: f64,
    p50: f64,
) -> f64 {
    let threshold = self.shock_threshold_kw;  // 默认 10.0 kW
    let spread = p90 - p50;

    if spread <= threshold {
        return 0.0;  // 无冲击负荷
    }

    let w_shock = self.shock_response_weight;  // 默认 20.0
    let lambda = self.response_time_penalty;  // 默认 5.0
    let max_load = 60.0;  // 最大切负荷
    let max_response = 60.0;  // 最大响应时间（秒）

    let load_reward = w_shock * (load_shedding / max_load);
    let time_penalty = lambda * (response_time / max_response);
    load_reward - time_penalty
}
```

#### 5.13.3 验收标准

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| SH-01 | 无冲击负荷时 R_shock = 0 | 单元测试 |
| SH-02 | 冲击负荷发生时，load_shedding 越大奖励越高 | 单元测试 |
| SH-03 | 响应时间越长惩罚越大 | 单元测试 |

#### 5.13.4 影响文件

- `mupc/crates/ai-engine/src/reward_calculator.rs`

### 5.14 P-Q 协同度阈值可配置化（R-07）

#### 5.14.1 需求描述

当前 `Q_THRESHOLD=0.10` 和 `P_THRESHOLD=5.0kW` 硬编码，不同台区工况差异大，需要支持灵活配置。

#### 5.14.2 配置结构

```rust
/// P-Q 协同度阈值配置（v2.12 R-07）
#[derive(Debug, Clone)]
pub struct PqCoordinationThresholds {
    /// Q 裕度阈值，低于此值视为"无功耗尽"
    pub q_margin_threshold: f64,     // 默认 0.10
    /// P 阈值（kW），省电策略判定阈值
    pub p_threshold_kw: f64,        // 默认 5.0
}

impl Default for PqCoordinationThresholds {
    fn default() -> Self {
        Self {
            q_margin_threshold: 0.10,
            p_threshold_kw: 5.0,
        }
    }
}
```

#### 5.14.3 配置更新

```toml
# mupc/config/mupc_env_config.yaml
[reward_thresholds]
q_margin_threshold = 0.10    # Q 裕度阈值
p_threshold_kw = 5.0         # P 阈值（kW）
```

#### 5.14.4 验收标准

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| TH-01 | Q_THRESHOLD 可通过配置修改 | 配置测试 |
| TH-02 | P_THRESHOLD 可通过配置修改 | 配置测试 |
| TH-03 | 配置缺失时使用默认值 | 单元测试 |
| TH-04 | 阈值修改后即时生效 | 配置热重载测试 |

#### 5.14.5 影响文件

- `mupc/crates/ai-engine/src/reward_calculator.rs`
- `mupc/config/mupc_env_config.yaml`

### 5.15 奖励函数精细化改进

#### 5.15.1 功能概述

v2.13 在 v2.12 基础上进一步精细化奖励函数设计，解决专家建议中的"精细化打磨"与"跨场景泛化"问题。

#### 5.15.2 P-Q协同Sigmoid平滑化

**实现位置**：`reward_calculator.rs` - `calc_pq_coordination()` / `calc_pq_coordination_static()`

```rust
// v2.13: Sigmoid平滑过渡
let k = 50.0;
let w_save = 1.0 / (1.0 + (-k * (q_margin - q_threshold)).exp());
let w_support = 1.0 - w_save;

// 省电模式奖励（Q有裕度时AI"偷懒"省电池）
let r_lazy = if p_ref.abs() < p_threshold { 50.0 } else { -5.0 };

// 支撑模式奖励（Q饱和时AI正确出手）
let r_correct = if (v_low && p_ref < 0.0) || (v_high && p_ref > 0.0) {
    50.0
} else if (v_low && p_ref >= 0.0) || (v_high && p_ref <= 0.0) {
    -30.0
} else {
    0.0
};

// Sigmoid加权组合 + 死区平滑因子
let r_pq = w_save * r_lazy + w_support * r_correct;
let dead_zone_factor = ((v_dev - 0.05) / 0.05).clamp(0.0, 1.0);
r_pq * dead_zone_factor
```

#### 5.15.3 动态自适应归一化

**新增文件**：`reward_normalizer.rs`

```rust
/// 滑动统计量（Welford 在线算法）
#[derive(Debug, Clone)]
pub struct RunningStats {
    mean: f64,
    m2: f64,
    count: usize,
}

impl RunningStats {
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn std(&self) -> f64 {
        if self.count < 2 { return 1.0; }
        (self.m2 / (self.count - 1) as f64).sqrt()
    }
}

/// 归一化公式：z = (r - μ) / (σ + ε)，clamp到[-1,1]
pub fn normalize(r: f64, stats: &RunningStats) -> f64 {
    let z = (r - stats.mean) / (stats.std() + 1e-6);
    z.clamp(-1.0, 1.0)
}
```

#### 5.15.4 状态改善率奖励

**实现位置**：`reward_calculator.rs` - `calc_state_improvement_reward()`

```rust
/// 公式：R_improve = w * (V_dev_prev - V_dev_curr) * sign(P_action)
fn calc_state_improvement_reward(&self, state: &FusedSystemState, p_action: f64, prev_v_avg: f64) -> f64 {
    let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
    let v_dev_curr = (v_avg - 1.0).abs();
    let v_dev_prev = *self.last_v_dev.read().unwrap();

    *self.last_v_dev.write().unwrap() = v_dev_curr;

    if v_dev_prev < 1e-6 {
        return 0.0;  // 首次调用，无改善率奖励
    }

    let delta_v_dev = v_dev_prev - v_dev_curr;
    let sign_p = if p_action > 0.0 { 1.0 } else { -1.0 };
    let w_improve = 10.0;

    w_improve * delta_v_dev * sign_p
}
```

#### 5.15.5 冲击负荷预备度奖励重构

**实现位置**：`reward_calculator.rs` - `shock_readiness_reward()`

```rust
/// 重构为"预备度奖励"
/// R_readiness = w1 * (soc_reserve - current_soc) + w2 * (p_ref_reserve - |p_ref|)
fn shock_readiness_reward(&self, state: &FusedSystemState, p_ref: f64, p90: f64, p50: f64) -> f64 {
    let spread = p90 - p50;
    if spread <= self.shock_threshold_kw {
        return 0.0;
    }

    let soc_gap = self.soc_reserve_target - state.battery_soc;
    let r_soc = self.shock_readiness_weight_soc * soc_gap;

    let p_ref_gap = self.p_ref_reserve_target - p_ref.abs();
    let r_p = self.shock_readiness_weight_p * p_ref_gap;

    r_soc + r_p
}
```

#### 5.15.6 在线微调PER+KL正则化强化

**实现位置**：`online_updater.rs`

```rust
/// PER 缓冲区（优先经验回放）
pub struct PerBuffer {
    samples: Vec<PerSample>,
    capacity: usize,
    alpha: f32,  // 优先级权重
    beta: f32,   // 重要性采样权重
}

/// KL 散度计算器
pub struct KLDivergenceCalculator {
    config: KLDivergenceConfig,
}

/// L_online = L_task + β * D_KL(π_new || π_offline)
pub fn compute_online_loss(&self, task_loss: f32, new_logits: &[f32], offline_logits: &[f32]) -> f32 {
    let kl = self.kl_divergence_calculator.compute(new_logits, offline_logits);
    let beta_adaptive = self.kl_divergence_calculator.get_adaptive_beta();
    task_loss + beta_adaptive * kl
}
```

#### 5.15.7 策略混合替代权重混合

**实现位置**：`mode_selector.rs` - `blend_actions()`

```rust
/// 公式：a_blended = (1 - α) * a_old + α * a_new
/// v2.15: 动作空间精简为 2 维（p_ref + k_droop）
pub fn blend_actions(&self, a_old: &ActionOutput, a_new: &ActionOutput, alpha: f64) -> ActionOutput {
    let one_minus_alpha = 1.0 - alpha;
    ActionOutput {
        p_ref: one_minus_alpha * a_old.p_ref + alpha * a_new.p_ref,
        k_droop: one_minus_alpha * a_old.k_droop + alpha * a_new.k_droop,
    }
}
```

#### 5.15.8 影响文件

| 文件 | 变更类型 |
|------|----------|
| `ai-engine/src/reward_calculator.rs` | 修改 |
| `ai-engine/src/reward_normalizer.rs` | **新增** |
| `ai-engine/src/online_updater.rs` | 修改 |
| `ai-engine/src/mode_selector.rs` | 修改 |

## 6. RKNN Runtime 设计

### 6.1 功能概述

RKNN Runtime 是 Rockchip 提供的 NPU 推理引擎，通过 FFI 调用 `librknnrt.so` C 库，在 RK3588 NPU 上执行 INT8 量化模型推理。所有 FFI 调用使用 `tokio::task::spawn_blocking` 在后台线程执行，不阻塞 Tokio async runtime。

### 6.2 FFI 绑定（rknn_runtime_sys.rs）

```rust
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct rknn_input {
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
    pub pass_timestamp: c_int,
}

#[repr(C)]
pub struct rknn_output {
    pub buf: *mut c_void,
    pub size: u32,
    pub is_preallocated: c_int,
}

#[link(name = \"rknnrt\")]
extern \"C\" {
    pub fn rknn_init(
        ctx: *mut u64,
        model_path: *const c_char,
        model_type: c_int,
        flag: c_int,
    ) -> c_int;

    pub fn rknn_inputs_set(ctx: u64, n: u32, inputs: *mut rknn_input) -> c_int;

    pub fn rknn_run(ctx: u64, reserved: *mut u64) -> c_int;

    pub fn rknn_outputs_get(ctx: u64, n: u32, outputs: *mut rknn_output) -> c_int;

    pub fn rknn_destroy(ctx: u64) -> c_int;

    pub fn rknn_query(ctx: u64, cmd: c_int, info: *mut c_void, size: u32) -> c_int;
}
```

### 6.3 C API 到 Rust 方法映射表

| C API | 功能 | Rust 封装 | 错误映射 |
|-------|------|-----------|----------|
| `rknn_init` | 模型加载与初始化 | `RknnRuntime::load()` | 失败 -> `AiEngineError::ModelLoadFailed` |
| `rknn_inputs_set` | 输入 tensor 设置 | `RknnRuntime::run()` 内部 | 失败 -> `AiEngineError::InferenceFailed` |
| `rknn_run` | 推理执行 | `RknnRuntime::run()` 内部 | 失败 -> `AiEngineError::InferenceFailed` |
| `rknn_outputs_get` | 输出 tensor 获取 | `RknnRuntime::run()` 内部 | 失败 -> `AiEngineError::InferenceFailed` |
| `rknn_destroy` | 资源释放 | `RknnContext::drop()` | Drop 中调用，错误仅记录日志 |
| `rknn_query` | 查询模型信息 | `RknnRuntime::load()` 内部 | 查询失败使用默认值 |

### 6.4 核心数据结构

```rust
/// RKNN 上下文（RAII 资源管理）
struct RknnContext {
    ctx: u64,
    input_count: u32,
    output_count: u32,
}

impl Drop for RknnContext {
    fn drop(&mut self) {
        unsafe { rknn_destroy(self.ctx); }
    }
}

/// RKNN Runtime 推理器
pub struct RknnRuntime {
    model_path: PathBuf,
    ctx: Arc<RwLock<Option<RknnContext>>>,
}
```

### 6.5 异步封装接口

```rust
impl RknnRuntime {
    /// 创建推理器
    pub fn new(model_path: &Path, expected_sha256: Option<&str>) -> Result<Self, AiEngineError>;

    /// 加载模型（spawn_blocking 异步封装）
    ///
    /// 加载前进行 SHA256 完整性校验（PRD 9.5 安全要求），校验失败拒绝加载并记录 ERROR。
    /// SHA256 校验通过后调用 rknn_init 加载模型到 NPU。
    pub async fn load(&self) -> Result<(), AiEngineError>;

    /// 执行推理（spawn_blocking 异步封装）
    ///
    /// 推理前调用 validate_input_vector() 检查输入张量无 NaN/Inf。
    /// 检测到异常值时拒绝推理并记录 ERROR 日志。
    pub async fn run(&self, input: &[f32]) -> Result<Vec<f32>, AiEngineError>;

    /// 释放资源（将 ctx 置 None，触发 Drop）
    pub async fn destroy(&self) -> Result<(), AiEngineError>;

    /// 检查模型是否已加载
    pub fn is_loaded(&self) -> bool;
}

// 线程安全声明
unsafe impl Send for RknnRuntime {}
unsafe impl Sync for RknnRuntime {}
```

**模型完整性校验（SHA256）：**

`RknnRuntime::new()` 接受可选的 `expected_sha256: Option<&str>` 参数，存入 `RknnRuntime.expected_sha256` 字段。`load()` 执行流程：

```
1. 读取模型文件全部字节到内存
2. 若 expected_sha256 为 Some(hash)：
   a. 计算文件内容的 SHA256 哈希
   b. 与 expected_sha256 比对
   c. 不匹配 → 记录 ERROR 日志，返回 AiEngineError::ChecksumMismatch
3. 校验通过 → spawn_blocking 调用 rknn_init
4. rknn_init 成功 → 设置 ModelStatus::Ready
```

SHA256 校验失败恢复路径：
- 尝试从 OTA 备份目录加载同名模型文件
- 备份文件通过 SHA256 校验 → 加载备份版本并记录 WARN
- 备份文件也不可用 → 触发 AI 降级，切换到本地策略引擎

LstmConfig 和 RlConfig 均包含 `expected_sha256: Option<String>` 字段，默认值为 `None`（开发环境跳过校验）。

### 6.6 错误码映射

```rust
fn map_rknn_error(code: c_int) -> Result<(), AiEngineError> {
    match code {
        0 => Ok(()),
        -1 => Err(AiEngineError::ModelLoadFailed(\"初始化失败\".into())),
        -2 => Err(AiEngineError::ModelLoadFailed(\"模型格式错误\".into())),
        -3 => Err(AiEngineError::ModelLoadFailed(\"模型不符合框架要求\".into())),
        -4 => Err(AiEngineError::ModelLoadFailed(\"SDK 版本不匹配\".into())),
        -5 => Err(AiEngineError::InferenceFailed(\"输入数量不匹配\".into())),
        -6 => Err(AiEngineError::InferenceFailed(\"输出数量不匹配\".into())),
        -7 => Err(AiEngineError::InferenceFailed(\"输入格式错误\".into())),
        -8 => Err(AiEngineError::InferenceFailed(\"输出格式错误\".into())),
        -9 => Err(AiEngineError::InferenceFailed(\"推理超时\".into())),
        -10 => Err(AiEngineError::InferenceFailed(\"上下文无效\".into())),
        _ => Err(AiEngineError::InferenceFailed(format!(\"未知错误: {}\", code))),
    }
}
```

### 6.7 推理延迟预算分配

| 阶段 | 最大延迟 | 说明 |
|------|----------|------|
| 状态输入准备 | 5ms | 融合数据读取 + to_input_vector() 序列化 |
| NPU 推理 | 100ms | rknn_inputs_set + rknn_run + rknn_outputs_get |
| 动作输出校验 | 0.5ms | 4 条约束规则 clamp |
| **总端到端延迟** | **120ms** | 从状态输入就绪到校验后动作输出可用 |

### 6.8 NPU 降级机制

当 NPU 不可用时（温度过高、推理连续失败），自动降级至 CPU 推理模式。

```rust
/// NPU 降级管理器
pub struct NpuFallbackManager {
    config: NpuConfig,
    /// 连续推理失败次数
    consecutive_failures: AtomicU32,
    /// 当前推理模式
    mode: AtomicU8, // 0=NPU, 1=CPU
    /// NPU 温度传感器读取函数
    temp_reader: Box<dyn Fn() -> f32 + Send + Sync>,
}

impl NpuFallbackManager {
    /// 降级条件：温度 > 85 deg C 或连续失败 > 3 次
    pub fn should_fallback(&self) -> bool;

    /// 切换到 CPU 模式
    pub async fn switch_to_cpu(&self);

    /// 恢复到 NPU 模式
    pub async fn switch_to_npu(&self);
}
```

温度监控逻辑：

- NPU 温度超过 `npu.temperature_limit_c` (85 deg C) 时触发降频保护，推理频率降低不超过初始频率的 `npu.throttle_factor` (0.5 = 50%)
- 温度连续 5 个周期恢复正常后自动恢复全速

### 6.9 RKNN 类型定义

```rust
/// RKNN 输入结构（高层封装）
#[derive(Debug, Clone)]
pub struct RknnInput {
    pub index: u32,
    pub buf: Vec<u8>,
    pub pass_timestamp: c_int,
}

/// RKNN 输出结构（高层封装）
#[derive(Debug)]
pub struct RknnOutput {
    pub buf: Vec<u8>,
}

impl RknnOutput {
    /// 安全转换 Vec<u8> 到 Vec<f32>，处理非对齐情况
    pub fn as_f32(&self) -> Vec<f32>;
}
```

## 7. ModelManager 统一调度设计

### 7.1 功能概述

ModelManager 是 ai-engine crate 的顶层编排器，统一管理 LSTM 预测模型、RL 决策模型、数据融合引擎、运行场景选择器和在线微调器。提供线程安全的异步访问接口。

### 7.2 结构体

```rust
/// 模型管理器 -- AI 引擎统一调度入口
pub struct ModelManager {
    config: AiEngineConfig,
    lstm_model: Arc<RwLock<Option<LstmModel>>>,
    rl_model: Arc<RwLock<Option<RLModel>>>,
    data_fusion: Option<DataFusionEngine>,
    reward_calculator: RewardCalculator,
    action_validator: ActionValidator,
    online_updater: Arc<RwLock<OnlineUpdater>>,
    status: Arc<RwLock<ModelStatus>>,
    mode_selector: Arc<ModeSelector>,
}

/// 模型状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Unloaded,
    Loading,
    Ready,
    Error,
}
```

### 7.3 full_decision_cycle() 完整流程

```rust
impl ModelManager {
    /// 完整 AI 决策周期（每个周期执行一次，默认 1Hz）
    ///
    /// 串联：模式获取 -> LSTM 预测 -> 数据融合 -> RL 决策 -> 约束校验 -> 奖励计算 -> 在线微调
    pub async fn full_decision_cycle(&self) -> Result<ActionOutput, AiEngineError> {
        // Step 0: 前置检查 -- 模型是否就绪
        if !self.is_ready().await {
            return Err(AiEngineError::ModelNotLoaded);
        }

        // Step 1: 获取当前运行场景
        let running_mode = self.mode_selector.current();

        // Step 2: LSTM 预测
        let lstm_input = self.build_lstm_input().await?;
        let lstm_output = self.predict_lstm(&lstm_input).await?;

        // Step 3: 多源数据融合
        let fused_state = self.data_fusion.as_ref()
            .ok_or(AiEngineError::FusionFailed(\"融合引擎未初始化\".into()))?
            .fuse()
            .await?;

        // Step 4: 状态向量序列化（78 维）
        let input_vector = fused_state.to_input_vector();

        // Step 5: 获取场景权重
        let scene_weights = SceneWeights::lookup(&self.config.reward_weights, running_mode);

        // Step 6: RL 决策推理
        let rl_action = self.decide_rl(&input_vector).await?;

        // Step 7: 动作约束校验
        let (validated, violations) = self.action_validator.validate_dual(
            &rl_action,
            fused_state.dispatch_p_set,
            false, // is_anti_reverse_scenario
            &self.config.action_constraint,
        ).await;

        // 记录违规
        for v in &violations {
            tracing::warn!(
                \"动作约束违规: rule={}, field={}, original={}, clamped={}\",
                v.rule, v.field, v.original, v.clamped
            );
        }

        // Step 8: 奖励计算
        let reward = self.reward_calculator.calculate(running_mode, &validated, &fused_state);

        // Step 9: 在线微调数据收集
        {
            let mut updater = self.online_updater.write().await;
            updater.add_sample(DataPoint {
                timestamp: chrono::Utc::now().timestamp_millis(),
                input: input_vector,
                output: vec![
                    validated.p_ref as f32,
                    validated.k_droop as f32,
                ],
            });
        }

        // Step 10: 发布消息总线事件
        // ai/action_output -> strategy-engine
        // ai/reward_value -> OnlineUpdater, Web UI
        // ai/model_status -> Web UI, 告警

        Ok(validated)
    }
}
```

### 7.4 影子模型验证+渐进式切换（R1）

#### 7.4.1 组件关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        ModelManager                            │
│  (统一调度入口，管理 LSTM/RL/奖励计算/动作校验)                   │
└──────────────────────┬────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ SafeOnlineUpdater │ DiscountedRewardCalculator │ SmoothSceneTransition │
│    (R1 新增)    │    (R2 新增)    │    (R3 新增)    │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ShadowModel   │ │  RewardCalculator │ │  ModeSelector   │
│ (克隆现有模型) │ │   (扩展)        │ │  (扩展)        │
└───────────────┘ └───────────────┘ └───────────────┘
```

#### 7.4.2 接口定义

```rust
// 核心错误类型
#[derive(Debug)]
pub enum UpdateError {
    SafetyViolation { score: f32, threshold: f32 },
    PerformanceDegradation { current: f32, shadow: f32 },
    SwitchInProgress,
    ModelNotReady,
}

// 渐进式切换配置
#[derive(Debug, Clone)]
pub struct GradualSwitchConfig {
    pub enabled: bool,
    pub steps: usize,           // 默认 10
    pub step_interval_secs: f64, // 默认 1.0
}

// 安全约束检查器接口（由 RobustnessManager 实现）
pub trait SafetyConstraintChecker: Send + Sync {
    fn check(&self, model: &ShadowModel) -> f32; // 返回安全评分 [0, 100]
}

// 性能监视器接口
pub trait PerformanceMonitor: Send + Sync {
    fn evaluate(&self, model: &ShadowModel) -> f32; // 返回性能评分 [0, 100]
}

// SafeOnlineUpdater 公开 API
impl SafeOnlineUpdater {
    pub async fn safe_update(&self, new_weights: &[f32]) -> Result<bool, UpdateError>;
    pub async fn gradual_switch(&self, target_weights: &[f32]) -> Result<(), UpdateError>;
    pub fn is_switching(&self) -> bool;
    pub fn current_blend_ratio(&self) -> f32; // 当前混合比例 [0.0, 1.0]
}
```

#### 7.4.3 核心结构

```rust
/// 影子模型（克隆自 ModelManager 的 RL 模型）
pub struct ShadowModel {
    weights: RwLock<Vec<f32>>,
    meta: ModelMeta,
}

/// 渐进式切换器
pub struct GradualSwitcher {
    config: GradualSwitchConfig,
    current_weights: Vec<f32>,
    target_weights: Vec<f32>,
    step_counter: usize,
    state: RwLock<SwitchState>,
}

/// SafeOnlineUpdater（R1 核心，替换现有占位实现）
pub struct SafeOnlineUpdater {
    config: OnlineUpdateConfig,
    shadow_model: ShadowModel,
    safety_checker: Arc<dyn SafetyConstraintChecker>,
    performance_monitor: Arc<dyn PerformanceMonitor>,
    gradual_switcher: GradualSwitcher,
    safety_threshold: f32,
    performance_threshold: f32,
}
```

#### 7.4.4 数据流

```
safe_update(new_weights)
  │
  ├─→ load_weights(new_weights) → shadow_model
  │
  ├─→ safety_checker.check(shadow_model) → score
  │     └→ score < threshold? → Err(SafetyViolation)
  │
  ├─→ performance_monitor.evaluate(current_model) → current_score
  ├─→ performance_monitor.evaluate(shadow_model) → shadow_score
  │     └→ shadow_score < current_score * 0.95? → Err(PerformanceDegradation)
  │
  └─→ gradual_switch(target_weights)
        │
        ├─ 启动异步任务（tokio::spawn）
        ├─ 每步间隔 step_interval_secs
        │     blend_ratio += 1.0 / steps
        │     发布混合权重事件
        └─ 完成后通知 ModelManager
```

#### 7.4.5 错误处理

| 错误类型 | 触发条件 | 处理策略 | 日志级别 |
|----------|----------|----------|----------|
| `SafetyViolation` | 影子模型安全评分 < 阈值 | 拒绝更新，保留当前权重 | WARN |
| `PerformanceDegradation` | 影子模型性能 < 当前 * 0.95 | 拒绝更新，保留当前权重 | WARN |
| `SwitchInProgress` | 切换进行中再次调用 safe_update | 拒绝，返回 Ok(false) | INFO |
| `ModelNotReady` | 影子模型未初始化 | 返回错误 | ERROR |

#### 7.4.6 测试策略

| 测试名称 | 验收条件 | 测试方法 |
|----------|----------|----------|
| `test_safety_violation_reject` | 安全评分 < 阈值时拒绝更新 | Mock SafetyConstraintChecker 返回 70，阈值 80 |
| `test_performance_degradation_reject` | 影子性能 < 当前 95% 时拒绝 | Mock PerformanceMonitor 返回 90 vs 100 |
| `test_gradual_switch_blend_ratio` | 10 步后 blend_ratio = 1.0 | 验证 step 10 时返回全目标权重 |
| `test_switch_in_progress_reject` | 切换中调用 safe_update 返回 Ok(false) | 触发切换后立即再调用 |
| `test_blend_weights_interpolation` | 每步线性插值正确 | 对比中间步的加权平均值 |

### 7.5 自适应权重优化器

#### 7.5.1 组件关系

```
AdaptiveWeightOptimizer
  ├─ MetaLearner (简化实现：基于规则调整)
  ├─ PerformanceCollector (trait)
  └─ WeightBoundsEnforcer
         ↓
  SceneWeights
         ↓
  RewardCalculator

ParetoWeightOptimizer (NSGA-II)
  ├─ fast_non_dominated_sort()
  ├─ calculate_crowding_distance()
  └─ evolve()
```

#### 7.5.2 新增文件

| 文件路径 | 职责 |
|---------|------|
| `ai-engine/src/adaptive_weight_optimizer.rs` | 自适应权重优化器核心实现（MetaRL） |
| `ai-engine/src/pareto_optimizer.rs` | NSGA-II 多目标优化器 |
| `ai-engine/src/performance_collector.rs` | 性能指标收集器 |

#### 7.5.3 配置结构

```rust
pub struct AdaptiveOptimizerConfig {
    pub enabled: bool,
    pub update_interval_hours: u32,   // 默认 168（周级更新）
    pub meta_learning_rate: f64,      // 默认 0.001
    pub weight_bounds: WeightBounds,  // min: 0.01, max: 10.0
    pub constraints: WeightConstraints, // sum_normalized: 8.3, max_adjustment_per_update: 0.2
}

pub struct ParetoOptimizerConfig {
    pub enabled: bool,
    pub population_size: usize,  // 默认 100
    pub generations: usize,     // 默认 50
    pub crossover_rate: f64,     // 默认 0.9
    pub mutation_rate: f64,      // 默认 0.1
}
```

#### 7.5.4 AdaptiveWeightOptimizer 详细设计

```rust
pub struct AdaptiveWeightOptimizer {
    config: AdaptiveOptimizerConfig,
    current_weights: RwLock<SceneWeights>,
    adjustment_history: RwLock<Vec<WeightAdjustment>>,
    performance_collector: Arc<dyn PerformanceCollector>,
}

impl AdaptiveWeightOptimizer {
    pub async fn optimize_weights(
        &self,
        historical_performance: &HistoricalPerformance,
    ) -> Result<SceneWeights, AiEngineError> {
        // 1. 提取性能特征
        let features = self.extract_features(historical_performance);
        // 2. 元学习器预测最优权重调整方向
        let adjustment = self.meta_learn_predict(&features)?;
        // 3. 应用调整（带约束剪裁）
        let new_weights = self.apply_adjustment(&adjustment)?;
        // 4. 记录调整历史
        self.record_adjustment(historical_performance, &adjustment).await;
        Ok(new_weights)
    }

    pub async fn validate_reward_drift(
        &self,
        original_reward: f64,
        optimized_reward: f64,
    ) -> bool {
        // AWO-06：偏移 < 5% 时验证通过
        if original_reward.abs() < 1e-6 {
            return (optimized_reward - original_reward).abs() < 0.05;
        }
        ((optimized_reward - original_reward) / original_reward).abs() < 0.05
    }

    /// v3.1: 权重健康度检查 — 累计漂移监控
    ///
    /// 当优化后的权重组合导致关键性能指标连续 N 个周期劣于基线时，
    /// 自动触发权重冻结，回退到基线权重。
    ///
    /// 监控指标：
    ///   - 变压器过载次数（overload_count）
    ///   - 电压越限次数（voltage_violation_count）
    ///   - 累积奖励（cumulative_reward）
    ///
    /// 退化判定：任意 2 项劣于基线 → 计为一次退化周期
    /// 冻结阈值：连续 health_freeze_threshold（默认 3）个周期退化 → 自动冻结
    pub async fn check_cumulative_health(&self) -> WeightHealthStatus {
        // ... 见 adaptive_weight_optimizer.rs 实现
    }
}

/// v3.1: 权重健康度状态
pub enum WeightHealthStatus {
    Healthy,                              // 不劣于基线
    Degraded { consecutive: u32 },        // 连续 N 周期退化
    Frozen,                               // 退化超阈值，自动冻结
    NoBaseline,                           // 首次运行，无基线
    CollectorError,                       // 采集失败
}

pub trait PerformanceCollector: Send + Sync {
    fn collect_historical(&self) -> Result<HistoricalPerformance, AiEngineError>;
    fn collect_current(&self) -> Result<PerformanceFeatures, AiEngineError>;
}
```

#### 7.5.5 ParetoWeightOptimizer (NSGA-II) 详细设计

```rust
pub struct ParetoWeightOptimizer {
    config: ParetoOptimizerConfig,
    objectives: Vec<OptimizationObjective>,
    pareto_front: RwLock<Vec<ParetoSolution>>,
}

impl ParetoWeightOptimizer {
    pub async fn find_pareto_front(
        &self,
        initial_population: &[WeightCandidate],
    ) -> Result<Vec<ParetoSolution>, AiEngineError> {
        let mut population = initial_population.to_vec();
        for _gen in 0..self.config.generations {
            let fronts = self.fast_non_dominated_sort(&population);
            for front in &mut fronts.iter_mut() {
                self.calculate_crowding_distance(front);
            }
            population = self.evolve(&fronts);
        }
        let fronts = self.fast_non_dominated_sort(&population);
        // 返回第一前沿（Pareto 最优解）
        if let Some(first_front) = fronts.first() {
            let mut solutions: Vec<ParetoSolution> = first_front.iter()...
            self.calculate_crowding_distance(&mut solutions);
            Ok(solutions)
        } else { Ok(Vec::new()) }
    }

    fn fast_non_dominated_sort(&self, population: &[WeightCandidate]) -> Vec<Vec<WeightCandidate>> {
        // NSGA-II 标准实现
    }

    fn calculate_crowding_distance(&self, front: &mut Vec<ParetoSolution>) {
        // 保持多样性
    }
}
```

#### 7.5.6 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| 元学习器推理失败 | 使用上一有效权重，回退告警 |
| 权重边界违规 | 强制裁剪到合法范围 |
| 性能数据缺失 | 跳过本轮优化，使用当前权重 |
| NSGA-II 收敛失败 | 返回空 Pareto 前沿，不更新权重 |

#### 7.5.7 测试策略

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| AWO-01 | 加载配置 | 配置正确解析，无 panic |
| AWO-04 | 权重约束 | 权重为正，归一化和正确 |
| AWO-05 | 调整幅度限制 | 单次变化不超过 20% |
| AWO-07 | 权重更新后 RL 推理 | 推理延迟 < 1s |

## 8. 安全 RL 包装器（Safety RL Wrapper）

> 原 §5.16，独立成章。安全包装器非奖励函数——它是物理模型前置过滤器，
> 基于戴维南等效电路预测电压变化，在 RL 动作生效前拦截危险动作。
> 调用位置: `full_decision_cycle()` 中 RL 决策后、ActionValidator 前。

> **来源**：`docs/TODO/安全RL包装器.md` + `docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md §3.7`

#### 8.1 需求描述

**现存问题**：
- ActionValidator 仅做静态数值校验（值域、变化率、调度约束），无法预测动作施加后电网的短时动态响应
- RobustnessManager（v2.9 已实现）属被动防御，仅在异常已发生（电压<0.9p.u.）时才介入，存在滞后窗口
- 合法的 `p_ref`（-30kW）在低电压工况下可致电压从 0.98 骤降至 0.92

**设计目标**：在 RL 决策后、ActionValidator 前插入**物理模型前置过滤器**，基于戴维南等效电路预测电压变化，提前拒绝高风险动作。

**设计原则**（PRD §3.7.1）：
1. **轻量化**：单次检查 < 5ms（远小于 120ms 总预算）
2. **保守优先**：预测失败回退到上一有效动作
3. **可证明安全**：基于简化电路方程，非黑盒
4. **与现有模块正交**：不修改 RL/ActionValidator/RewardCalculator

#### 8.2 数据结构

**核心结构体**：

```rust
// 文件：crates/ai-engine/src/safety_wrapper.rs

/// 线路阻抗参数（来自配置文件）
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LineImpedance {
    pub r_ohm: f64,        // 电阻 R（Ω）
    pub x_ohm: f64,        // 电抗 X（Ω）
    pub v_base: f64,       // 基准电压（V），默认 220.0
}

/// 安全边界
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SafetyBounds {
    pub v_min: f64,        // 0.93（p.u.）
    pub v_max: f64,        // 1.07（p.u.）
    pub dv_dt_max: f64,    // 0.03（p.u./s）
    pub soc_margin: f64,   // 0.02（比临界多 2%）
}

/// 物理模型预测结果
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub v_predicted: f64,       // 预测电压（p.u.）
    pub dv_dt: f64,             // 电压变化率（p.u./s）
    pub soc_after: f64,         // 动作后 SOC 估算
    pub is_safe: bool,          // 综合安全标志
    pub reason: Option<String>, // 不安全原因
}

/// 检查结果
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CheckResult {
    Passed,
    Rejected { reason: String },
    FallbackDueToPredictionError,
}

/// 安全包装器主结构
pub struct SafetyRLWrapper {
    line_impedance: Arc<RwLock<LineImpedance>>,
    last_safe_action: Arc<RwLock<ActionOutput>>,
    bounds: SafetyBounds,
    stats: Arc<RwLock<SafetyStats>>,
    // v2.17 新增：事件广播（用于 SSE 推送给 Web UI）
    event_sender: Option<SafetyEventSender>,
}

/// 累计指标
#[derive(Debug, Default, Clone)]
pub struct SafetyStats {
    pub total_checks: u64,
    pub total_rejected: u64,
    pub total_fallback: u64,
    pub rejection_rate_1h: f64,
    pub avg_latency_us: u64,
    pub max_latency_us: u64,
}

/// 违规记录（持久化）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub timestamp: i64,
    pub reason: String,
    pub proposed_p_ref: f64,
    pub proposed_k_droop: f64,
    pub fallback_p_ref: f64,
    pub fallback_k_droop: f64,
    pub v_predicted: f64,
    pub latency_us: u64,
}

/// 安全包装器事件（v2.17 新增，broadcast 推送用）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyWrapperEvent {
    pub timestamp: i64,
    pub event_type: SafetyEventType,
    pub check_result: CheckResult,
    pub proposed_p_ref: f64,
    pub proposed_k_droop: f64,
    pub fallback_p_ref: f64,
    pub fallback_k_droop: f64,
    pub v_predicted: f64,
    pub latency_us: u64,
}

/// 事件类型（区分违规 vs 通过 vs 回退）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyEventType {
    Passed,
    Violation,
    Fallback,
}

/// 全局事件总线 Sender（由 main.rs 注入）
pub type SafetyEventSender = tokio::sync::broadcast::Sender<SafetyWrapperEvent>;
```

**SafetyPredictor trait（可替换）**：

```rust
#[async_trait]
pub trait SafetyPredictor: Send + Sync {
    async fn predict(
        &self,
        state: &FusedSystemState,
        action: &ActionOutput,
    ) -> Result<PredictionResult, AiEngineError>;
}

/// 默认线性灵敏度预测器（戴维南等效）
pub struct LinearSensitivityPredictor {
    impedance: LineImpedance,
    bounds: SafetyBounds,
}

impl LinearSensitivityPredictor {
    /// 戴维南等效电路 + 灵敏度分析
    /// 
    /// 公式：ΔV ≈ (R·ΔP + X·ΔQ) / V₀
    /// 单位换算：P/Q 单位 kW/kVar → V 单位 V → p.u. (÷v_base)
    pub async fn predict_inner(
        &self,
        state: &FusedSystemState,
        action: &ActionOutput,
    ) -> Result<PredictionResult, AiEngineError> {
        let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
        
        // 当前 P_output（含下垂）
        let p_cur = state.p_ref_current.unwrap_or(0.0)
            + state.k_droop_current.unwrap_or(0.0) * (v_avg - 1.0);
        
        // 新动作下的 P_output
        let p_new = action.p_ref + action.k_droop * (v_avg - 1.0);
        
        // ΔP 转换为 W
        let delta_p_w = (p_new - p_cur) * 1000.0;
        
        // ΔQ 估算（基于 q_realtime_margin）
        let q_margin = state.q_realtime_margin;
        let delta_q_var = if q_margin > 0.20 {
            0.0  // Q 有裕度，认为实时模块维持当前 Q
        } else {
            // Q 饱和：假设实时模块全力调节
            let q_max_var = 300.0;  // 与 yaml 配置对齐
            (1.0 - q_margin) * q_max_var * (v_avg - 1.0).signum()
        };
        
        // 灵敏度公式：ΔV ≈ (R·ΔP + X·ΔQ) / V₀
        let delta_v_volt = (self.impedance.r_ohm * delta_p_w 
            + self.impedance.x_ohm * delta_q_var) / self.impedance.v_base;
        let delta_v_pu = delta_v_volt / self.impedance.v_base;
        
        let v_predicted = v_avg + delta_v_pu;
        
        // 边界检查
        let is_safe = v_predicted >= self.bounds.v_min
            && v_predicted <= self.bounds.v_max
            && delta_v_pu.abs() <= self.bounds.dv_dt_max;
        
        // SOC 检查
        let soc_ok = !(action.p_ref > 0.0 && state.battery_soc < 0.10 + self.bounds.soc_margin);
        
        let reason = if !is_safe {
            Some(format!(
                "v_predicted={:.3} 越界 [{}, {}]",
                v_predicted, self.bounds.v_min, self.bounds.v_max
            ))
        } else if !soc_ok {
            Some(format!(
                "SOC={:.3} 低于安全阈值 {:.3}",
                state.battery_soc, 0.10 + self.bounds.soc_margin
            ))
        } else {
            None
        };
        
        Ok(PredictionResult {
            v_predicted,
            dv_dt: delta_v_pu,
            soc_after: state.battery_soc,  // 简化：假设 1 秒内 SOC 不变
            is_safe: is_safe && soc_ok,
            reason,
        })
    }
}
```

#### 8.3 核心算法：check_and_fallback 流程

```rust
impl SafetyRLWrapper {
    pub async fn check_and_fallback(
        &self,
        state: &FusedSystemState,
        proposed_action: &ActionOutput,
    ) -> (ActionOutput, CheckResult) {
        let start = std::time::Instant::now();
        
        // 1. 物理模型预测
        let pred = match self.predictor.predict(state, proposed_action).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("SafetyRLWrapper 预测失败: {:?}", e);
                let latency = start.elapsed().as_micros() as u64;
                self.update_stats(latency, true).await;
                let fallback = self.last_safe_action.read().await.clone();
                return (fallback, CheckResult::FallbackDueToPredictionError);
            }
        };
        
        // 2. 边界检查
        let latency = start.elapsed().as_micros() as u64;
        if !pred.is_safe {
            tracing::warn!(
                "SafetyRLWrapper 拒绝动作: reason={:?}, proposed={:?}",
                pred.reason, proposed_action
            );
            self.update_stats(latency, true).await;
            
            // 发布违规事件
            self.publish_violation(state, proposed_action, &pred, latency).await;
            
            let fallback = self.last_safe_action.read().await.clone();
            return (fallback, CheckResult::Rejected { reason: pred.reason.unwrap_or_default() });
        }
        
        // 3. 通过：更新 last_safe_action 和统计
        *self.last_safe_action.write().await = proposed_action.clone();
        self.update_stats(latency, false).await;
        (proposed_action.clone(), CheckResult::Passed)
    }
    
    async fn update_stats(&self, latency_us: u64, was_rejected: bool) {
        let mut stats = self.stats.write().await;
        stats.total_checks += 1;
        if was_rejected { stats.total_rejected += 1; }
        // 滑动平均延迟
        stats.avg_latency_us = 
            (stats.avg_latency_us * (stats.total_checks - 1) + latency_us) / stats.total_checks;
        if latency_us > stats.max_latency_us { stats.max_latency_us = latency_us; }
    }
    
    async fn publish_violation(
        &self,
        state: &FusedSystemState,
        proposed: &ActionOutput,
        pred: &PredictionResult,
        latency_us: u64,
    ) {
        let fallback = self.last_safe_action.read().await.clone();
        let violation = SafetyViolation {
            timestamp: chrono::Utc::now().timestamp(),
            reason: pred.reason.clone().unwrap_or_default(),
            proposed_p_ref: proposed.p_ref,
            proposed_k_droop: proposed.k_droop,
            fallback_p_ref: fallback.p_ref,
            fallback_k_droop: fallback.k_droop,
            v_predicted: pred.v_predicted,
            latency_us,
        };
        
        // v2.17 修订（D-01/D-02 修复）：事件驱动架构
// 使用 tokio::sync::broadcast 推送到全局事件总线
// Web API SsePushService 订阅后通过 SSE 推送给 Web UI
        let event = SafetyWrapperEvent {
            timestamp: violation.timestamp,
            event_type: SafetyEventType::Violation,
            check_result: CheckResult::Rejected { reason: violation.reason.clone() },
            proposed_p_ref: violation.proposed_p_ref,
            proposed_k_droop: violation.proposed_k_droop,
            fallback_p_ref: violation.fallback_p_ref,
            fallback_k_droop: violation.fallback_k_droop,
            v_predicted: violation.v_predicted,
            latency_us: violation.latency_us,
        };
        
        // 1. tracing 记录（持久审计日志）
        tracing::warn!(
            timestamp = event.timestamp,
            reason = %violation.reason,
            v_predicted = event.v_predicted,
            latency_us = event.latency_us,
            "SafetyRLWrapper 违规",
        );
        
        // 2. broadcast 推送（实时事件，Web UI 通过 SSE 接收）
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event.clone());  // 失败不影响主流程
        }
        
        // 3. storage 持久化（历史查询用）
        crate::storage::record_safety_violation(&violation).await.ok();
    }
}
```

#### 8.4 ModelManager 集成

**集成位置**：`full_decision_cycle` 第 6 步后、ActionValidator 前

```rust
// crates/ai-engine/src/model_manager.rs

pub struct ModelManager {
    // ... 现有字段 ...
    safety_wrapper: Arc<SafetyRLWrapper>,  // v2.17 新增
}

// main.rs 中组装示例
fn setup_safety_wrapper() -> (Arc<SafetyRLWrapper>, SafetyEventSender) {
    let (tx, _) = tokio::sync::broadcast::channel(256);
    let wrapper = Arc::new(SafetyRLWrapper {
        line_impedance: Arc::new(RwLock::new(LineImpedance::default())),
        last_safe_action: Arc::new(RwLock::new(ActionOutput::default())),
        bounds: SafetyBounds::default(),
        stats: Arc::new(RwLock::new(SafetyStats::default())),
        event_sender: Some(tx.clone()),
    });
    (wrapper, tx)
}

impl ModelManager {
    pub async fn full_decision_cycle(&self) -> Result<ActionOutput, AiEngineError> {
        // ... 既有步骤 1-5（数据融合、LSTM预测、RL决策）...
        
        // Step 6: RL 决策（已有）
        let rl_action = registry.decide(&input_vector, &action_space_config).await?;
        
        // Step 6.5: v2.17 新增 SafetyRLWrapper 检查
        let (safe_action, check_result) = self.safety_wrapper.check_and_fallback(
            &fused_state,
            &rl_action,
        ).await;
        
        // 记录检查结果
        tracing::info!("SafetyRLWrapper: {:?}", check_result);
        
        // Step 7: ActionValidator（继续使用 safe_action）
        let (validated, violations) = self.action_validator.validate(
            &safe_action,
            fused_state.dispatch_p_set,
            false,
            &action_space_config,
        );
        
        // ... 既有步骤 8+ ...
        Ok(validated)
    }
}
```

**与 RobustnessManager 协同顺序**（Q-W3=A 决策）：

```
完整决策链：

RLModel.decide() → 原始动作
   ↓
[新] SafetyRLWrapper.check_and_fallback()    ← 事前预测（v2.17）
   ↓ (安全/回退后的动作)
RobustnessManager.detect_and_respond()       ← 事中应急（v2.9 已有）
   ↓ (应急动作或原动作)
ActionValidator.validate_dual()              ← 静态校验（v2.15 已有）
   ↓
strategy-engine
```

**边界明确**：
- SafetyRLWrapper：**预测**未来电压越界则**事前拒绝**
- RobustnessManager：**检测**当前异常（v_avg<0.9、>1.1、SOC 极值）则**应急**
- ActionValidator：**校验**值域/变化率

#### 8.5 配置结构

**新增配置结构**（`crates/ai-engine/src/config.rs`）：

```rust
/// v2.17 安全 RL 包装器配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SafetyWrapperConfig {
    pub line_impedance_r_ohm: f64,    // 默认 0.1
    pub line_impedance_x_ohm: f64,    // 默认 0.05
    pub v_base: f64,                  // 默认 220.0
    pub v_min: f64,                   // 默认 0.93
    pub v_max: f64,                   // 默认 1.07
    pub dv_dt_max: f64,               // 默认 0.03
    pub soc_margin: f64,              // 默认 0.02
    pub max_check_latency_ms: u64,    // 默认 5
    pub alert_rejection_rate: f64,    // 默认 0.20（告警阈值）
}

impl Default for SafetyWrapperConfig {
    fn default() -> Self {
        Self {
            line_impedance_r_ohm: 0.1,
            line_impedance_x_ohm: 0.05,
            v_base: 220.0,
            v_min: 0.93,
            v_max: 1.07,
            dv_dt_max: 0.03,
            soc_margin: 0.02,
            max_check_latency_ms: 5,
            alert_rejection_rate: 0.20,
        }
    }
}
```

**ai.toml 配置段**：

```toml
[safety_wrapper]
line_impedance_r_ohm = 0.1      # 线路电阻（Ω），从台区档案读取
line_impedance_x_ohm = 0.05     # 线路电抗（Ω），从台区档案读取
v_base = 220.0                  # 基准电压（V）
v_min = 0.93                    # 电压下限
v_max = 1.07                    # 电压上限
dv_dt_max = 0.03                # 电压变化率上限
soc_margin = 0.02               # SOC 安全裕度
max_check_latency_ms = 5
alert_rejection_rate = 0.20     # 拒绝率告警阈值
```

#### 8.6 Web API 设计（SSE 推送为主，HTTP API 仅用于状态查询）

**架构变更**：
- 实时违规通知通过 **SSE 推送**（基于 broadcast channel）
- HTTP API 仅用于：状态查询、统计查询（冷路径）

**SSE 事件类型扩展**（`crates/web-api/src/sse/mod.rs`）：

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum SseEventType {
    // ... 既有类型 ...
    SafetyWrapperUpdate {
        check_result: CheckResult,
        reason: String,
        v_predicted: f64,
        latency_us: u64,
    },
}
```

**新增路由**（`crates/web-api/src/routes/ai/safety_wrapper.rs`）：

```rust
use axum::{
    routing::get,
    Router,
    extract::State,
    Json,
};
use crate::AppState;

/// 路由注册
pub fn safety_wrapper_routes() -> Router<AppState> {
    Router::new()
        .route("/status", get(get_status))                  // 状态查询
        .route("/stats", get(get_stats))                    // 统计查询
        // 实时通知通过 SSE EventSource 自动推送，无需轮询端点
}

/// GET /api/v1/safety_wrapper/status
/// 当前状态（边界条件、line_impedance、累计指标）
async fn get_status(
    State(state): State<AppState>,
    _user: crate::RequireRole<crate::Role>,
) -> Json<SafetyStatus> {
    let safety_wrapper = state.safety_wrapper.read().await;
    Json(SafetyStatus {
        bounds: safety_wrapper.bounds().clone(),
        line_impedance: safety_wrapper.line_impedance().clone(),
        stats: safety_wrapper.stats().clone(),
    })
}

/// GET /api/v1/safety_wrapper/stats
/// 统计指标
async fn get_stats(
    State(state): State<AppState>,
    _user: crate::RequireRole<crate::Role>,
) -> Json<SafetyStats> {
    let safety_wrapper = state.safety_wrapper.read().await;
    Json(safety_wrapper.stats().clone())
}
```

**Web API 订阅 broadcast**（`AppState` 初始化时）：

```rust
// main.rs 或 AppState 初始化
let safety_event_tx = setup_safety_wrapper_tx();
let mut safety_event_rx = safety_event_tx.subscribe();

let sse_service_clone = sse_service.clone();
tokio::spawn(async move {
    while let Ok(event) = safety_event_rx.recv().await {
        let sse_event = SseEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type: SseEventType::SafetyWrapperUpdate {
                check_result: event.check_result,
                reason: event.reason.clone(),
                v_predicted: event.v_predicted,
                latency_us: event.latency_us,
            },
            timestamp: event.timestamp,
            payload: serde_json::to_value(&event).unwrap_or_default(),
        };
        sse_service_clone.push(sse_event).await;
    }
});
```

**响应结构**：

```rust
#[derive(Serialize)]
pub struct SafetyStatus {
    pub bounds: SafetyBounds,
    pub line_impedance: LineImpedance,
    pub stats: SafetyStats,
    pub current_mode: RunningMode,
    pub last_check_result: Option<CheckResult>,
}
```

**性能对比**：

| 方案 | AI 引擎开销 | Web API 开销 | 实时性 |
|------|------------|--------------|--------|
| HTTP 轮询（10s）| 0（被动）| 0.1 req/s/客户端 | 最差 10s 延迟 |
| **SSE 推送（推荐）** | **0（push 即可）** | **0（无主动查询）** | **<100ms** |

**Web UI 集成**（修改 §8.7）：
- 使用 `EventSource` API 订阅 SSE
- 收到 `SafetyWrapperUpdate` 事件即更新 UI
- 无需任何轮询代码
```

#### 8.7 Web UI 设计

**页面**：`crates/web-api/src/static/ai-monitor.html`

**布局（ASCII Mockup）**：

```
┌──────────────────────────────────────────────────────────────┐
│ MUPC AI 安全监控面板                              [刷新 ⇄]   │
├──────────────────────────────────────────────────────────────┤
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐      │
│ │ 当前安全状态 │ │ 拒绝率（24h） │ │ 平均延迟（P99）  │      │
│ │   ✅ 安全    │ │   12.3%      │ │     1.2 ms       │      │
│ └──────────────┘ └──────────────┘ └──────────────────┘      │
│                                                              │
│ 安全边界配置                                                 │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 电压下限：0.93 p.u.    电压上限：1.07 p.u.              │ │
│ │ dv/dt 上限：0.03 p.u./s  SOC 裕度：0.02 (12% 临界)       │ │
│ │ 线路阻抗：R=0.10Ω, X=0.05Ω                             │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ 拒绝率趋势（24h）[折线图]                                    │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │   ╱╲                                                     │ │
│ │  ╱  ╲    ╱╲                                              │ │
│ │ ╱    ╲  ╱  ╲___                                          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ 最近违规记录（最近 10 条）                                   │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 2026-06-18 10:30:15 | v_predicted=0.92 | rejected      │ │
│ │ 2026-06-18 10:30:14 | SOC=0.118 | rejected              │ │
│ │ 2026-06-18 10:30:13 | v_predicted=0.91 | rejected      │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ⚠️ 拒绝率超过 20%，建议检查台区状态                          │
└──────────────────────────────────────────────────────────────┘
```

**交互要点**：
1. **自动刷新**：状态卡片 5s，趋势图 30s，违规列表 10s
2. **拒绝率告警**：当 `rejection_rate > 0.20` 时顶部显示红色横幅
3. **点击违规记录**：弹出详情（建议动作、回退动作、v_predicted 等）
4. **响应式布局**：≥1200px 显示完整布局，<1200px 卡片堆叠

**API 调用**（SSE EventSource 订阅模式，**无轮询**）：

```javascript
// === 状态查询（5s 轮询，仅冷路径） ===
setInterval(async () => {
  const res = await fetch('/api/v1/safety_wrapper/status');
  const status = await res.json();
  updateStatusCards(status);
  checkAlertThreshold(status.stats.rejection_rate_1h);
}, 5000);

// === 实时事件订阅（v2.17 新增，通过 SSE） ===
// 监听 SafetyWrapperUpdate 事件，自动接收违规通知
const eventSource = new EventSource('/api/v1/sse/safety_wrapper');
eventSource.addEventListener('SafetyWrapperUpdate', (e) => {
  const event = JSON.parse(e.data);
  updateViolationList([event]);  // 立即更新违规列表
  showAlertBanner(event);       // 显示告警横幅（如需要）
});
// 注意：违规列表接收是 push 模式，不是 pull 模式
// Web UI 收到事件后更新本地状态，无需轮询 AI 引擎
```

#### 8.8 错误处理

| 场景 | 处理策略 |
|------|----------|
| 物理模型预测 panic | 返回 `FallbackDueToPredictionError`，回退到 `last_safe_action` |
| 物理模型预测超时（>5ms）| 返回 `FallbackDueToPredictionError`，记录 WARN |
| 配置文件缺失 `[safety_wrapper]` | 使用代码内默认值 + 记录 INFO |
| `LineImpedance` 字段为 0 | 使用默认值 + 记录 WARN |
| `last_safe_action` 未初始化（首次）| 初始化为 `ActionOutput { p_ref: 0.0, k_droop: 0.0 }` |
| 消息总线 publish 失败 | 记录 ERROR，不影响主流程 |
| Storage 持久化失败 | 记录 ERROR，不影响主流程 |
| Web API 鉴权失败 | 返回 403 |

#### 8.9 测试策略

| 测试类型 | 测试项 | 验证方法 |
|----------|--------|----------|
| 单元 | `LinearSensitivityPredictor::predict_inner` 数学正确性 | 给定 v_avg=0.98, p_cur=10, p_new=-20, 验证 v_predicted |
| 单元 | 边界检查（v_min/v_max/dv_dt/soc）| 5 个边界场景 |
| 单元 | `check_and_fallback` 通过/拒绝/回退 3 路径 | 3 个场景 |
| 单元 | `update_stats` 累计指标正确性 | 模拟 100 次检查 |
| 集成 | ModelManager 集成（与 RL/Validator/RobustnessManager 协同）| 模拟完整决策链 |
| 集成 | 消息总线 publish 正确性 | Mock message_bus |
| 集成 | Storage 持久化正确性 | 集成测试 |
| API | 3 个端点返回正确 | API 测试 |
| UI | 监控面板渲染 + 自动刷新 | Playwright 测试 |
| 性能 | 单次检查 < 5ms（P99）| 1000 次采样 |

**单元测试示例**：

```rust
#[test]
fn test_predict_inner_low_voltage_risk() {
    let predictor = LinearSensitivityPredictor::new(
        LineImpedance { r_ohm: 0.1, x_ohm: 0.05, v_base: 220.0 },
        SafetyBounds::default(),
    );
    
    let state = FusedSystemState {
        voltage_phase_a: 0.94,
        voltage_phase_b: 0.94,
        voltage_phase_c: 0.94,
        battery_soc: 0.5,
        q_realtime_margin: 0.5,
        p_ref_current: Some(0.0),
        k_droop_current: Some(0.0),
        ..Default::default()
    };
    
    let action = ActionOutput {
        p_ref: 30.0,        // 放电 30kW
        k_droop: 0.0,
    };
    
    let pred = predictor.predict_inner(&state, &action).unwrap();
    
    // 放电使电压进一步降低，应触发安全检查
    assert!(!pred.is_safe);
    assert!(pred.reason.is_some());
}
```

#### 8.10 影响文件

| 文件 | 变更类型 | 估算代码行数 |
|------|----------|-------------|
| `ai-engine/src/safety_wrapper.rs` | **新增** | ~380 行 |
| `ai-engine/src/lib.rs` | 修改（导出新模块）| +5 行 |
| `ai-engine/src/model_manager.rs` | 修改（集成点）| +30 行 |
| `ai-engine/src/config.rs` | 修改（SafetyWrapperConfig）| +50 行 |
| `mupc/config/ai.toml` | 修改（[safety_wrapper] 段）| +15 行 |
| `web-api/src/routes/ai/safety_wrapper.rs` | **新增** | ~60 行 |
| `web-api/src/routes/ai/mod.rs` | 修改（注册路由）| +3 行 |
| `web-api/src/sse/mod.rs` | 修改（新增 SafetyWrapperUpdate 事件）| +20 行 |
| `web-api/src/static/ai-monitor.html` | **新增** | ~250 行 |
| `storage/src/repository.rs` | 修改（新增 safety_violations 表）| +40 行 |
| `main.rs` (bin) | 修改（依赖注入 broadcast channel）| +30 行 |
| **合计** | — | **~880 行** |

> **设计修订（D-01/D-02/D-03 修复）**：
> 1. 事件流采用 `tokio::sync::broadcast`（AI 引擎 → Web API → SSE → Web UI），不依赖 HTTP 轮询
> 2. AI 引擎 `event_sender: Option<SafetyEventSender>` 字段，main.rs 注入 Sender
> 3. Web API 订阅 broadcast Receiver，转发到现有 `SsePushService`
> 4. Web UI 用 `EventSource` 订阅 SSE 端点，零轮询开销
> 5. storage 持久化作为审计通道（与 broadcast 并行，独立存在）

#### 8.11 设计决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 物理模型选择 | 线性灵敏度（戴维南等效）| 5ms 预算下最简单可靠的模型 |
| 线路阻抗来源 | ai.toml 配置 | 跨台区可配，避免硬编码 |
| 检查失败时回退目标 | `last_safe_action`（上一周期有效动作）| 与现有"通信中断保持最后参数"一致 |
| 与 RobustnessManager 顺序 | SafetyRLWrapper 在前 | 事前预测先于事中应急 |
| 违规日志存储 | 消息总线 + storage 双重 | 实时性 + 持久化查询 |
| Web UI 技术 | 原生 HTML + JS（无框架）| 单页面简单，无构建工具链 |
| 拒绝率告警阈值 | 20%（可配置）| 经验值，需现场调优 |

## 9. 与策略引擎集成设计

### 9.1 集成架构

```
+------------------+         +-----------------------+
|   ai-engine      |-------->|   strategy-engine     |
|                  | ActionOutput                   |
| ModelManager     |  & ModelStatus  +------------+
| full_decision_   |               |AiIntegrator|
| cycle()          |<--------------|(门面)       |
|                  |  模式切换查询    +------------+
+------------------+                      |
                                         v
                                  +----------------+
                                  |AiCommandValidator|
                                  |(安全校验)        |
                                  +----------------+
```

### 9.2 AiIntegrator 扩展

AiIntegrator 是 web-api 访问 ai-engine 的服务门面，增加以下接口：

```rust
/// AI 集成器（strategy-engine -> ai-engine 门面）
pub struct AiIntegrator {
    model_manager: Arc<RwLock<Option<ModelManager>>>,
    status: Arc<RwLock<ModelStatus>>,
}

impl AiIntegrator {
    // ---- 查询接口 ----

    /// 获取 AI 引擎状态快照
    pub async fn engine_status(&self) -> AiEngineStatusInfo;

    /// 获取当前运行模式
    pub async fn current_mode(&self) -> Option<RunningMode>;

    /// 获取 ModeSelector 引用（用于订阅切换事件）
    pub async fn mode_selector(&self) -> Option<Arc<ModeSelector>>;

    /// 检查 AI 引擎是否就绪
    pub async fn is_ready(&self) -> bool;

    // ---- 控制接口 ----

    /// 初始化 AI 引擎并加载模型
    pub async fn initialize(&self, config: AiEngineConfig) -> Result<(), AiEngineError>;

    /// 切换运行场景
    pub async fn switch_mode(
        &self,
        new_mode: RunningMode,
        source: SwitchSource,
    ) -> Result<RunningMode, AiEngineError>;

    /// 获取最新 AI 决策结果
    pub async fn latest_decision(&self) -> Option<ActionOutput>;
}
```

### 9.3 AiCommandValidator 扩展

策略引擎中的 `AiCommandValidatorImpl` 对 AI 引擎下发的 ActionOutput 进行二次安全校验：

```rust
impl AiCommandValidatorImpl {
    /// 校验 AI 引擎输出的 ControlCommand
    ///
    /// 校验逻辑：
    /// 1. 无 AI 引擎时 -> 降级通过
    /// 2. 遥测数据过期 (>5s) -> 降级通过
    /// 3. 功率调节命令 -> 对比 AI 推荐值与实际命令偏差
    ///    - 偏差 > 10kW 且 AI 置信度 < 0.7 -> 拒绝
    /// 4. 开关控制命令 -> 直接通过
    /// 5. 非功率调节命令 -> 直接通过
    pub async fn validate_ai_command(
        &self,
        cmd: &ControlCommand,
        ai_output: &ActionOutput,
    ) -> ValidationResult;
}
```

### 9.4 兜底策略联动

AI 引擎失效时自动降级至本地策略引擎：

```
AI 引擎异常检测条件：
  - ModelStatus == Error
  - 连续 3 次推理失败
  - 数据融合任一数据源连续 10 周期无更新

降级流程：
  1. strategy-engine 检测到 AI 引擎异常
  2. 自动切换至本地策略模式 (< 2s)
  3. 系统发出告警 \"AI模式降级\"
  4. 本地策略（削峰填谷/需量控制/防逆流）接管控制
  5. AI 引擎恢复后（连续 5 个周期正常），自动切回 AI 模式
```

## 10. 文件结构

```
mupc/crates/ai-engine/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # 模块导出，重新导出所有公共类型
│   ├── model_manager.rs          # 模型管理器（full_decision_cycle 统一调度 + 多RknnRuntime管理）
│   ├── mode_selector.rs          # 运行场景选择器（5 种互斥场景 + 远程/本地切换）
│   ├── lstm_model.rs             # LSTM 时序预测模型（LstmInput, LstmOutput, LstmModel + predict_with_vmd()）
│   ├── rl_model.rs               # RL 决策模型（FusedSystemState, ActionOutput, RLModel）
│   ├── reward_calculator.rs      # 奖励函数计算器（5 种场景奖励公式 + SceneWeights）
│   ├── robustness_manager.rs     # 电压异常应急策略管理器（v2.9 新增）
│   ├── adaptive_weight_optimizer.rs  # 自适应权重优化器（v2.11 新增）
│   ├── pareto_optimizer.rs       # NSGA-II Pareto 多目标优化器（v2.11 新增）
│   ├── performance_collector.rs  # 性能指标收集器（v2.11 新增）
│   ├── load_covariates.rs        # 负荷协变量结构体（v2.11 新增）
│   ├── weather_service.rs        # 气象数据服务 trait（v2.11 新增）
│   ├── data_fusion.rs            # 多源数据融合引擎（DataSourceAdapter trait + 5 个实现）
│   ├── action_validator.rs       # 动作约束校验器（4 条双参数校验规则 ACT-DUAL-01~04，v2.15）
│   ├── online_updater.rs         # 在线微调（DataPoint, OnlineUpdater, batch_size=32）
│   ├── rknn_runtime.rs           # RKNN Runtime 推理器（RAII, spawn_blocking, NPU降级）
│   ├── rknn_runtime_sys.rs       # RKNN Runtime C API FFI 绑定（unsafe extern \"C\"）
│   ├── rknn_types.rs             # RKNN 类型定义（RknnInput, RknnOutput, as_f32）
│   ├── safety_wrapper.rs         # 安全 RL 包装器（v2.17 新增，~380行）
│   ├── reward_normalizer.rs      # 动态自适应归一化器（v2.13 新增）
│   ├── error.rs                  # 错误类型枚举（AiEngineError, thiserror）
│   └── config.rs                 # 配置结构（AiEngineConfig 及子配置 + SafetyWrapperConfig）
│
│   # ---- v3.0 预测增强管线新增文件 ----
│   ├── vmd.rs                    # VMD 分解器（纯Rust，~300行）
│   ├── prediction_pipeline.rs    # 预测增强管线编排器（~800行）
│   ├── pipeline_config.rs        # 增强配置结构体（~400行）
│   ├── residual_buffer.rs        # 残差滑动窗口缓冲（~120行）
│   └── model_validator.rs        # 模型文件 metadata 校验（~100行）
│
└── tests/
    ├── ai_engine_tests.rs        # 配置默认值测试
    ├── lstm_model_tests.rs       # LSTM 模型集成测试
    ├── rl_model_tests.rs         # RL 模型集成测试
    ├── rknn_runtime_tests.rs     # RKNN Runtime 集成测试
    └── online_updater_tests.rs   # 在线微调集成测试

# ---- v3.0 MSSA 超参优化工具（纯 Python，不进入 RK3588 部署）----
tools/mssa_optimizer/
├── __init__.py                   # 包初始化 (~5 行)
├── mssa.py                      # MSSA 算法核心 (~300 行)
├── search_space.py               # 搜索空间定义与编码/解码 (~150 行)
├── objective.py                  # 目标函数（调用训练+评估）(~200 行)
├── config.py                     # 配置加载/校验 (~100 行)
├── output.py                     # JSON 输出（对齐 PRD 7.4.2）(~80 行)
├── mssa_cache.json               # 评估缓存（运行时自动生成，.gitignore 排除）
├── test_mssa.py                  # 单元测试 (~200 行)
└── config/
    └── mssa_search_config.yaml   # MSSA 搜索配置文件模板 (~60 行)
```

### 10.1 lib.rs 模块导出

```rust
pub mod config;
pub mod error;
pub mod lstm_model;
pub mod mode_selector;
pub mod model_manager;
pub mod online_updater;
pub mod rknn_runtime;
pub mod rknn_runtime_sys;
pub mod rknn_types;
pub mod rl_model;
pub mod data_fusion;
pub mod reward_calculator;
pub mod robustness_manager;     // v2.9 新增
pub mod adaptive_weight_optimizer; // v2.11 新增
pub mod pareto_optimizer;        // v2.11 新增
pub mod performance_collector;    // v2.11 新增
pub mod load_covariates;        // v2.11 新增
pub mod action_validator;
pub mod safety_wrapper;          // v2.17 新增
pub mod reward_normalizer;       // v2.13 新增

// v3.0 预测增强管线新增
pub mod vmd;                     // VMD 分解器
pub mod prediction_pipeline;     // 预测增强管线编排器
pub mod pipeline_config;         // 增强配置结构体
pub mod residual_buffer;         // 残差滑动窗口缓冲
pub mod model_validator;         // 模型文件 metadata 校验

// 重新导出公共类型
pub use config::{
    ActionConstraintConfig, AiEngineConfig, FusionConfig, LstmConfig, ModeConfig,
    ModelType, NpuConfig, OnlineUpdateConfig, QuantizationType, RlAlgorithm, RlConfig,
    SceneWeights, SafetyWrapperConfig, PredictionEnhancementConfig,
};
pub use error::AiEngineError;
pub use mode_selector::{
    parse_mode_name, ModeSelector, ModeSwitchEvent, RunningMode, SwitchSource,
};
pub use model_manager::{ModelManager, ModelStatus};
pub use lstm_model::{LstmInput, LstmModel, LstmOutput};
pub use rl_model::{ActionOutput, FusedSystemState, RLModel, parse_action_output};
pub use reward_calculator::RewardCalculator;
pub use robustness_manager::{RobustnessManager, AnomalyType};  // v2.9 新增
pub use adaptive_weight_optimizer::{AdaptiveWeightOptimizer, PerformanceCollector, PerformanceFeatures, WeightAdjustment, HistoricalPerformance};  // v2.11 新增
pub use pareto_optimizer::{ParetoWeightOptimizer, ParetoSolution, WeightCandidate, OptimizationObjective};  // v2.11 新增
pub use load_covariates::{LoadCovariates};  // v2.11 新增
pub use data_fusion::{DataFusionEngine, DataSourceAdapter, SourceType, FusedSystemState};
pub use action_validator::{ActionValidator, ViolationRecord};
pub use online_updater::{DataPoint, OnlineUpdater};
pub use rknn_runtime::RknnRuntime;
pub use safety_wrapper::{SafetyRLWrapper, SafetyPredictor, LinearSensitivityPredictor, SafetyBounds, SafetyStats, CheckResult};  // v2.17 新增
pub use safety_wrapper::{SafetyWrapperEvent, SafetyEventType, SafetyViolation};  // v2.17 新增

// v3.0 预测增强管线重新导出
pub use vmd::{VmdDecomposer, VmdConfig, VmdResult};
pub use prediction_pipeline::{PredictionPipeline, EnhancedForecastResult, EnhancementLevel, PipelineHealth};
pub use pipeline_config::{VmdEnhancementConfig, AttentionConfig, BiLstmConfig, ErrorCorrectionConfig, FeatureSelectionConfig};
pub use residual_buffer::ResidualBuffer;
```

## 11. 配置结构

### 11.1 AiEngineConfig

```rust
/// AI 引擎总配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AiEngineConfig {
    /// LSTM 预测模型配置
    pub lstm: LstmConfig,
    /// RL 决策模型配置
    pub rl: RlConfig,
    /// 在线微调配置
    pub online_update: OnlineUpdateConfig,
    /// 数据融合配置
    pub fusion: FusionConfig,
    /// 运行场景选择配置
    pub mode: ModeConfig,
    /// 动作约束配置
    pub action_constraint: ActionConstraintConfig,
    /// 场景权重映射
    pub reward_weights: SceneWeights,
    /// NPU 推理配置
    pub npu: NpuConfig,
    /// v2.5 奖励阈值配置
    pub reward_thresholds: RewardThresholdConfig,
    /// v2.17 安全 RL 包装器配置
    pub safety_wrapper: SafetyWrapperConfig,
    /// v3.0 预测增强配置（VMD + Attention + BiLSTM + 误差修正）
    #[serde(default)]
    pub prediction_enhancement: Option<PredictionEnhancementConfig>,
}
```

### 11.2 预测增强配置

`PredictionEnhancementConfig` 定义于 `pipeline_config.rs`，在 `mupc/config/mupc_env_config.yaml` 中通过 `prediction_enhancement` 段配置。缺失时所有增强功能禁用，系统运行于 v2.16 基线模式。

```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PredictionEnhancementConfig {
    #[serde(default)]
    pub vmd: VmdEnhancementConfig,
    #[serde(default)]
    pub attention: AttentionConfig,
    #[serde(default)]
    pub bilstm: BiLstmConfig,
    #[serde(default)]
    pub error_correction: ErrorCorrectionConfig,
    #[serde(default)]
    pub feature_selection: FeatureSelectionConfig,
}
```

**VMD 配置：**

```rust
pub struct VmdEnhancementConfig {
    pub enabled: bool,        // 默认 false
    pub k_pv: usize,          // 光伏模态数 [2, 10]，默认 5
    pub k_load: usize,        // 负荷模态数 [2, 10]，默认 6
    pub alpha: f64,           // 惩罚因子 [100, 5000]，默认 2000.0
    pub tol: f64,             // 收敛容差，默认 1.0e-6
    pub max_iter: usize,      // 最大迭代次数，默认 500
    pub tau: f64,             // 噪声容忍度，默认 0.0
}
```

**Attention 配置：**

```rust
pub struct AttentionConfig {
    pub enabled: bool,        // 默认 false（需 ONNX 模型含 Attention 层）
    pub score_type: String,   // "additive" | "dot" | "general"
    pub export_weights: bool, // 是否导出注意力权重到日志
}
```

**BiLSTM 配置 (R2)：**

```rust
pub struct BiLstmConfig {
    pub enabled: bool,              // 默认 false
    pub gate_passed: bool,          // Go/No-Go 准入标志，默认 false
    pub model_path: Option<String>, // 模型文件路径
    pub hidden_size_override: Option<usize>, // 隐状态维度覆盖（调试用）
    pub fallback_on_failure: bool,  // 默认 true
}
```

**误差修正配置 (R2)：**

```rust
pub struct ErrorCorrectionConfig {
    pub enabled: bool,                  // 默认 false
    pub model_path: Option<String>,     // 修正模型文件路径
    pub residual_window_steps: usize,   // 残差窗口步数，默认 24
    pub zero_init: bool,                // 冷启动零填充，默认 true
    pub auto_disable_after_failures: u32, // 连续失败自动禁用阈值，默认 3
    pub enable_bias_check: bool,        // 系统性偏差检测，默认 true
}
```

**特征筛选配置：**

```rust
pub struct FeatureSelectionConfig {
    pub mic_top_k: usize,  // MIC 筛选 Top-K 特征数，默认 7
}
```

### 11.3 YAML 配置 -- prediction_enhancement 段

```yaml
# ============================================================================
# 预测增强配置（v3.0，2026-06-21）
# 缺失时系统运行于 v2.16 基线模式（全部增强功能禁用）
# ============================================================================
prediction_enhancement:
  vmd:
    enabled: true
    k_pv: 5
    k_load: 6
    alpha: 2000.0
    tol: 1.0e-6
    max_iter: 500
    tau: 0.0

  attention:
    enabled: true
    score_type: "additive"
    export_weights: false

  bilstm:
    enabled: false
    gate_passed: false
    model_path: "/etc/mupc/models/bilstm_attn.rknn"
    hidden_size_override: null
    fallback_on_failure: true

  error_correction:
    enabled: false
    model_path: "/etc/mupc/models/error_correction.rknn"
    residual_window_steps: 24
    zero_init: true
    auto_disable_after_failures: 3
    enable_bias_check: true

  feature_selection:
    mic_top_k: 7
```

**配置热加载策略：**

| 配置项 | 热加载行为 |
|--------|-----------|
| `vmd.enabled` | 下一推理周期生效 |
| `attention.enabled` | 下一推理周期生效（仅当模型含 Attention 层） |
| `bilstm.enabled` | 需重新加载模型文件；`gate_passed` 需重启生效 |
| `error_correction.enabled` | 下一推理周期生效 |
| `residual_window_steps` | 变更需重建 ResidualBuffer |
| `error_correction.model_path` | 需重新加载模型文件（OTA 场景） |

配置文件示例 (`mupc/config/ai.toml`)：

```toml
[lstm]
model_path = \"/etc/mupc/models/lstm.rknn\"
input_window_secs = 21600     # 6 小时（v2.16: 3600 → 21600）
output_horizon_secs = 22500   # 225 分钟 = 15 步 × 15 分钟（v2.16: 900 → 22500）
step_seconds = 900            # 15 分钟步长（v2.16 新增）
quantization = \"INT8\"

[rl]
model_path = \"/etc/mupc/models/rl.rknn\"
algorithm = \"MADDPG\"
quantization = \"INT8\"

[online_update]
enabled = true
batch_size = 32
learning_rate = 0.001

[fusion]
fusion_period_secs = 1
data_source_timeout_secs = 10
enable_health_monitoring = true

[mode]
default_mode = \"SeasonalLoadManagement\"
persist_path = \"/var/lib/mupc/current_mode\"

[action_constraint]
p_batt_ramp_limit_kw = 50.0
q_batt_ramp_limit_kvar = 30.0
max_apparent_power_kva = 500.0
pv_limit_min = 0.1

[reward_weights]
agricultural_irrigation = [1.0, 0.5, 2.0]
commercial_arbitrage = [1.0, 1.0]
demand_control = [1.0, 0.5]
virtual_power_plant = [1.0, 2.0, 1.0]
ultra_green = [1.0, 1.0]

[npu]
temperature_limit_c = 85.0
throttle_factor = 0.5
enable_fallback_to_cpu = true
```

### 11.4 子配置结构定义

所有子配置结构均提供 `Default` 实现。

| 配置结构 | 关键字段 | 默认值 |
|----------|----------|--------|
| LstmConfig | model_path, input_window_secs, output_horizon_secs, step_seconds, quantization | /etc/mupc/models/lstm.rknn, 21600, 22500, 900, INT8 |
| RlConfig | model_path, algorithm, quantization | /etc/mupc/models/rl.rknn, MADDPG, INT8 |
| OnlineUpdateConfig | enabled, batch_size, learning_rate | false, 32, 0.001 |
| FusionConfig | fusion_period_secs, data_source_timeout_secs, enable_health_monitoring | 1, 10, true |
| ModeConfig | default_mode, persist_path | \"SeasonalLoadManagement\", \"/var/lib/mupc/current_mode\" |
| ActionConstraintConfig | p_batt_ramp_limit_kw, q_batt_ramp_limit_kvar, max_apparent_power_kva, pv_limit_min | 50.0, 30.0, 500.0, 0.1 |
| SceneWeights | agricultural_irrigation[3], commercial_arbitrage[2], demand_control[2], virtual_power_plant[3], ultra_green[2] | 见上表默认值 |
| NpuConfig | temperature_limit_c, throttle_factor, enable_fallback_to_cpu | 85.0, 0.5, true |
| **RewardThresholdConfig（v2.5新增）** | voltage_deadband, q_margin_threshold, voltage_high_limit, soc_critical, voltage_penalty_high, voltage_penalty_low | 0.05, 0.10, 1.05, 0.10, 2.0, 1.0 |

**RewardThresholdConfig 结构定义：**

```rust
/// v2.5 奖励阈值配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RewardThresholdConfig {
    /// 电压死区（±5%），与现有设计一致
    pub voltage_deadband: f64,
    /// Q 裕度阈值：实时模块剩余容量低于此值视为"无功耗尽"
    pub q_margin_threshold: f64,
    /// 弃光前置电压阈值：电压高于此值时弃光奖励不计入
    pub voltage_high_limit: f64,
    /// SOC 极低保护阈值
    pub soc_critical: f64,
    /// 高电压侧电压惩罚系数（光伏超发）
    pub voltage_penalty_high: f64,
    /// 低电压侧电压惩罚系数（灌溉/炒茶/空调负荷）
    pub voltage_penalty_low: f64,
}
```

### 11.5 枚举类型定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum QuantizationType { FP32, FP16, INT8 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType { LSTM, MADDPG, PPO }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlAlgorithm { MADDPG, PPO }
```

## 12. 错误类型

### 12.1 AiEngineError 枚举

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AiEngineError {
    #[error(\"模型加载失败: {0}\")]
    ModelLoadFailed(String),

    #[error(\"模型文件校验失败: 期望 {expected}, 实际 {actual}\")]
    ChecksumMismatch { expected: String, actual: String },

    #[error(\"推理失败: {0}\")]
    InferenceFailed(String),

    #[error(\"模型未加载\")]
    ModelNotLoaded,

    #[error(\"输入形状不匹配: 期望 {expected:?}, 实际 {actual:?}\")]
    InputShapeMismatch {
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    #[error(\"输出形状不匹配\")]
    OutputShapeMismatch,

    #[error(\"RKNN Runtime 错误: {0}\")]
    RknnError(String),

    #[error(\"模型版本不兼容: {0}\")]
    VersionMismatch(String),

    #[error(\"在线微调失败: {0}\")]
    OnlineUpdateFailed(String),

    #[error(\"数据融合失败: {0}\")]
    FusionFailed(String),

    #[error(\"模式切换失败: {0}\")]
    ModeSwitchFailed(String),

    #[error(\"动作校验失败: {0}\")]
    ActionValidationFailed(String),

    #[error(\"数据源过期: {0}\")]
    DataSourceStale(String),

    #[error(\"NPU 温度过高: current={current}°C, limit={limit}°C\")]
    NpuOverheating { current: f32, limit: f32 },

    #[error(\"奖励计算错误: {0}\")]
    RewardCalculationError(String),

    // ---- v3.0 预测增强管线新增 ----
    #[error(\"VMD 分解失败: {0}\")]
    VmdFailed(String),

    #[error(\"VMD 迭代不收敛 (max_iter={max_iter}, 最终误差={final_error})\")]
    VmdNotConverged { max_iter: usize, final_error: f64 },

    #[error(\"Attention 层退化 (所有权重相等)\")]
    AttentionDegraded,

    #[error(\"误差修正失败: {0}\")]
    ErrorCorrectionFailed(String),

    #[error(\"模型校验失败: model={model_path}, reason={reason}\")]
    ModelValidationFailed { model_path: String, reason: String },

    #[error(\"残差缓冲不足: filled={filled}/{capacity}\")]
    ResidualBufferInsufficient { filled: usize, capacity: usize },
}
```

### 12.2 错误分类与处理策略

| 错误类别 | 错误变体 | 恢复策略 |
|----------|----------|----------|
| 模型加载 | `ModelLoadFailed`, `VersionMismatch`, `ModelValidationFailed` | 拒绝启动，记录 ERROR，触发降级 |
| 推理运行时 | `InferenceFailed`, `RknnError`, `InputShapeMismatch`, `OutputShapeMismatch` | 重试 1 次，失败后记录 ERROR，连续 3 次后触发 NPU 降级 |
| 资源状态 | `ModelNotLoaded`, `ResidualBufferInsufficient` | 等待模型加载完成；残差缓冲不足时零填充或拒绝推理 |
| 数据异常 | `FusionFailed`, `DataSourceStale` | 按缺失数据处理策略填充，连续 10 周期后触发降级 |
| 运维操作 | `ModeSwitchFailed`, `ActionValidationFailed`, `OnlineUpdateFailed` | 记录 WARN，操作回滚 |
| 硬件异常 | `NpuOverheating` | 降频保护，连续 5 周期正常后恢复 |
| **预测增强** | `VmdFailed`, `VmdNotConverged`, `AttentionDegraded`, `ErrorCorrectionFailed` | VMD 失败自动降级至无 VMD 模式；连续 5 次成功后自动升级；误差修正失败跳过修正、主预测值直出、连续 3 次失败自动禁用 |

## 13. 消息总线集成

### 13.1 Topic 定义

| Topic | 发布者 | 订阅者 | 数据格式 | 频率 |
|-------|--------|--------|----------|------|
| `ai/fused_state` | DataFusionEngine | RLModel, RewardCalculator | FusedSystemState (JSON) | 1Hz |
| `ai/action_output` | ModelManager | strategy-engine, intercore | ActionOutput (JSON) | 1Hz |
| `ai/reward_value` | RewardCalculator | OnlineUpdater, Web UI | RewardValue (JSON) | 1Hz |
| `ai/model_status` | ModelManager | Web UI, 告警模块 | ModelStatus (JSON) | 1Hz |
| `ai/mode_switch` | ModeSelector | RewardCalculator, strategy-engine, Web UI, 审计日志 | ModeSwitchEvent (JSON) | 事件驱动 |
| `ai/current_mode` | ModeSelector | Web UI（心跳查询） | RunningMode (JSON) | 按需查询 |
| `price/real_time` | data-processing | DataFusionEngine (PriceAdapter) | ElectricityPrice (JSON) | 15min / 事件 |
| `weather/forecast` | data-processing | DataFusionEngine (WeatherAdapter) | WeatherData (JSON) | 15min |
| `demand/current` | data-processing | DataFusionEngine | DemandData (JSON) | 1Hz |

### 13.2 消息格式定义

```rust
/// 奖励值消息
#[derive(Debug, Clone, Serialize)]
pub struct RewardValue {
    pub mode: RunningMode,
    pub reward: f64,
    pub components: Vec<(String, f64)>,  // 各子项贡献值
    pub timestamp: i64,
}

/// 模式切换事件（已定义于 mode_selector.rs）
pub struct ModeSwitchEvent {
    pub previous: RunningMode,
    pub current: RunningMode,
    pub source: SwitchSource,
    pub timestamp: i64,
}

/// 电价消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricityPrice {
    pub current_price: f64,
    pub next_period_price: f64,
    pub tariff_id: u8,
    pub peak_price: f64,
    pub valley_price: f64,
    pub timestamp: i64,
}

/// 气象数据消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub solar_irradiance: f64,
    pub temperature: f64,
    pub timestamp: i64,
}

/// 需量数据消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandData {
    pub current_demand: f64,
    pub contract_demand: f64,
    pub peak_demand_this_month: f64,
    pub timestamp: i64,
}
```

## 14. 技术决策记录

### 14.1 ADR-001: INT8 量化模型部署

**决策：** 所有 AI 模型（LSTM, RL）训练后统一经 ONNX 导出，由 rknn-toolkit2 量化为 INT8 精度，以 .rknn 格式部署到 RK3588 NPU。

**理由：**
- INT8 量化后模型大小 <= 5MB（满足 <= 5MB 要求）
- NPU 推理延迟 < 100ms（INT8 推理速度是 FP32 的 4~8 倍）
- 精度损失在可接受范围内（MAPE 增加 < 2%）
- RK3588 NPU 原生支持 INT8 推理

### 14.2 ADR-002: tokio::spawn_blocking 异步封装

**决策：** 所有 FFI 调用（rknn_init, rknn_inputs_set, rknn_run, rknn_outputs_get, rknn_destroy）通过 `tokio::task::spawn_blocking` 在后台线程执行。

**理由：**
- librknnrt.so C API 均为同步阻塞调用
- 直接在 async 上下文中调用会阻塞整个 Tokio runtime，导致其他任务饥饿
- spawn_blocking 将阻塞任务分配到专用的阻塞线程池，不占用 async worker 线程

### 14.3 ADR-003: ModeSelector（互斥模式选择器）替代 SceneClassifier（自动分类器）

**决策：** 场景确定方式从 AI 自动识别改为调度主站远程控制或策略管理员本地选择，ModeSelector 使用 `tokio::sync::Mutex` 保证互斥。

**理由：**
- 电力系统运行场景应由调度人员明确指定，而非 AI 自动推断
- 自动分类器在边界工况下可能误判，导致奖励函数和优化目标错误
- 调度主站具有全局电网调度视角，可主动下发场景切换指令
- 互斥保证（Mutex）比自动分类（概率输出）的确定性更高

### 14.4 ADR-004: 10 大类状态空间 + 78 维输入向量

**决策：** 状态空间从最初 6 大类 48 维扩展至当前 10 大类（D1 实时数据含三相电压, D2 预测数据, D3 电价, D4 需量, D5 气象, D6 调度指令, D7 实时模块, D8 季节时段, D9 安全覆盖, D10 概率负荷），序列化为 78 维固定长度输入向量。

**理由：**
- 多类别按数据源组织便于缺失数据处理和配置管理
- 固定长度简化 RKNN Runtime 输入形状校验
- 预测向量超出时截断、不足时补零，保证维度一致性
- D1 中三相电压标幺值 (voltage_phase_a/b/c) 使 AI 引擎能感知台区电压水平，执行 P/Q 协同控制
- v2.5~v2.14 扩展：逐步增加 q_realtime_margin、季节/时段编码、安全覆盖状态、概率负荷预测等维度

### 14.5 ADR-005: 4 条双参数动作约束规则 + clamp 限幅

**决策：** AI 模型输出 2 维动作（p_ref, k_droop）后，经 4 条双参数校验规则（ACT-DUAL-01~04）校验，违反约束时自动 clamp 到安全边界，并记录 WARN 日志。load_shedding 和 pv_limit 的约束下沉至 strategy-engine（需量控制/防逆流策略内置边界检查），confidence 保留在 ModelOutput 中用于内部校验。

**理由：**
- AI 模型输出不能直接下发给物理设备，必须经过安全校验
- 变化率约束 (ACT-DUAL-03) 保护电池设备免受功率突变损害
- 值域约束 (ACT-DUAL-01/02) 保证动作值在安全边界内
- 调度约束 (ACT-DUAL-04) 保证不超出调度指令权限
- clamp（截断）比拒绝动作更鲁棒：拒绝动作会导致控制中断，clamp 保留有效部分
- v2.15 将 load_shedding/pv_limit 下沉至策略引擎，使 AI 引擎专注于核心 P-Q 协同控制

### 14.6 ADR-006: A/B 双缓冲模型热加载

**决策：** 模型更新时采用双缓冲模式：新模型加载到独立上下文中，加载期间旧模型继续服务；新模型加载完成后原子切换，旧模型延迟释放。

**理由：**
- 不中断推理服务
- 模型切换时间 < 30ms
- 支持模型版本回滚（保留上一个稳定版本）

### 14.7 ADR-007: 三相电压从 D1 实时数据中采集（非 D5 电能质量独立分类）

**决策：** 三相电压幅值（voltage_phase_a/b/c）作为 D1 实时数据的子字段，直接进入 RL 输入向量，用于电压感知 P/Q 控制。

**理由：**
- 电压幅值是 P/Q 控制策略的必要输入（过电压 -> 吸收无功，低电压 -> 释放无功）
- 与三相不平衡治理（不涉及电池充放电，由实时控制核心独立处理）是不同用途
- 归入 D1 实时数据使数据流更简洁（无需单独的电能质量适配器），同一 intercore TCP 通道获取

### 14.8 ADR-008: VMD 纯 Rust 实现

**决策：** VMD 分解器采用纯 Rust（rustfft + nalgebra），备选 FFI 调用 C libvmd。

**理由：** 无 FFI 跨编译链依赖，aarch64-openEuler 目标直接 cross build；VMD 核心是 ADMM 迭代（FFT + 矩阵运算），Rust 生态已成熟；去掉 unsafe 边界。

**位置：** §2.3.1

### 14.9 ADR-009: Attention ONNX 图内嵌

**决策：** Attention 机制由 MUPC-AI2 训练管线在 ONNX 导出时嵌入计算图，不 Rust 侧实现。

**理由：** Rust 侧零代码改动；Attention 计算在 NPU 上执行，延迟增量 <= 15%；降低 Rust 侧维护负担。

**位置：** §2.3.2

### 14.10 ADR-010: MIC 离线 Python

**决策：** MIC 特征筛选由 MUPC-AI2 项目中独立 Python 脚本完成（minepy 库），JSON 输出。

**理由：** MIC 仅在训练阶段离线执行，不进入 RK3588 部署路径；Python minepy 是 MIC 计算的权威实现；Rust 生态无成熟 MIC crate。

**位置：** §2.8

### 14.11 ADR-011: BiLSTM 双模型文件 + Go/No-Go 准入

**决策：** 训练管线导出两个独立的 .rknn 模型（`lstm_attn.rknn` + `bilstm_attn.rknn`），硬件验证准入。

**理由：** 推理路径零分支；独立文件便于 OTA 独立升级/回滚；ONNX metadata_props 交叉校验模型类型与配置一致性。

**位置：** §2.3.4

### 14.12 ADR-012: 误差修正独立模型 + 独立 Runtime

**决策：** 误差修正 BiLSTM 作为独立 .rknn 模型（`error_correction.rknn`），拥有独立的 RknnRuntime 实例，与主预测模型完全分离。

**理由：** 模型独立性（输入维度不同）、运行时独立性（独立 OTA 升级/降级）、RknnRuntime 实例化成本低（< 50ms）、完全隔离保证误差修正模型故障不影响主预测。

**位置：** §2.3.5

### 14.13 ADR-013: 误差修正 RknnRuntime 挂在 PredictionPipeline

**决策：** 误差修正 RknnRuntime (#2) 挂在 `PredictionPipeline` 而非 `ModelManager` 下。

**理由：** 关注点分离 -- 误差修正与主预测在 PredictionPipeline 内串行编排，同一生命周期管理更简单；`ModelManager` 不应感知误差修正的内部实现细节。

**位置：** §2.6

### 14.14 ADR-014: BiLSTM/误差修正/单向 LSTM 三模型独立 OTA

**决策：** 任意一个模型的升级/回滚不影响其他两个模型。

**理由：** 降低 OTA 耦合风险 -- 单向 LSTM 回滚不应影响误差修正，BiLSTM 升级不应需要同步升级误差修正。`ModelManager` 通过版本号元数据追踪各模型当前版本。

**位置：** §2.6

### 14.15 ADR-015: data_fusion.rs 不修改

**决策：** MIC 离线筛选后 ONNX 维度固定，Rust 侧无需适配。

**理由：** 特征增减在训练管线一侧（MUPC-AI2）处理，ONNX 模型使用固定特征集；Rust 侧按 ONNX 输入 shape 构造 LstmInput 即可；`FusedSystemState` 序列化逻辑不感知特征维度变化。

**位置：** §2.8

### 14.16 ADR-016: MSSA 算法参数与编码

**决策：** N=30, p_d=0.2, p_s=0.1, 佳点集初始化, 13 维混合编码（离散索引+浮点+One-hot枚举）对应 10 维逻辑超参, epsilon=1e-4 收敛。

**位置：** §15

### 14.17 ADR-017: MSSA 目标函数与缓存

**决策：** 加权 MAPE = 0.5*MAPE_pv + 0.5*MAPE_load, 惩罚分 1e6, SHA256 指纹缓存跨运行持久化, training_data_fingerprint 自动失效。

**位置：** §15

### 14.18 ADR-018: IPSO 降级路径

**决策：** 配置 `algorithm: "IPSO"` 一键切换, JSON Schema 保持一致, 用于 MSSA 超时或不收敛时快速收敛。

**位置：** §15

### 14.19 关键实现文件

These are the most critical files that need to be created or significantly modified to implement this design:

- `e:\MUPC2\mupc\crates\ai-engine\src\data_fusion.rs` (new: DataFusionEngine, DataSourceAdapter trait, 5 adapter implementations, FusedSystemState with to_input_vector())
- `e:\MUPC2\mupc\crates\ai-engine\src\rl_model.rs` (refactor: replace SystemState with FusedSystemState, replace old 8-field ActionOutput with new 5-field ActionOutput, add parse_action_output, add 78-dim input support)
- `e:\MUPC2\mupc\crates\ai-engine\src\reward_calculator.rs` (new: RewardCalculator with 5 scene formulas, SceneWeights lookup)
- `e:\MUPC2\mupc\crates\ai-engine\src\action_validator.rs` (new: ActionValidator with 5 constraint rules ACT-01~05, clamp logic, ViolationRecord)
- `e:\MUPC2\mupc\crates\ai-engine\src\model_manager.rs` (refactor: add full_decision_cycle(), wire in DataFusionEngine, RewardCalculator, ActionValidator)
## 15. MSSA 超参优化工具设计

> 本章已迁移至 MUPC-AI2 训练管线设计文档。
> MSSA 是离线训练阶段 Python 工具，非 RK3588 Rust 运行时组件。
> 详见：`MUPC-AI2/docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-设计文档.md` §11

---
## 附录：版本演进

> 正文已整合全部历史补丁，本表仅作演进追溯。

| 版本 | 主要变更 |
|------|----------|
| v2.3 | 恢复 SCENE-01 电压质量惩罚（P/Q 协同控制可主动调节电压幅值） |
| v2.5 | FusedSystemState 新增 q_realtime_margin 与季节/时段编码，输入向量扩展至 56 维 |
| v2.7 | 双参数动作空间（p_ref + k_droop），时间尺度解耦 |
| v2.8 | P-Q 协同度奖励替代电压硬惩罚，新增下垂系数平滑惩罚 |
| v2.10 | 安全覆盖惩罚 + 影子模型验证/折扣累积奖励/场景平滑过渡 |
| v2.14 | SafetyOverride 奖励重构 + FusedSystemState 扩展至 78 维 |
| v2.15 | 动作空间精简 5→2 维，load_shedding/pv_limit 下沉策略引擎 |
| v3.0 | 合并预测增强分层混合架构（VMD+Attention+BiLSTM+误差修正+MSSA） |
