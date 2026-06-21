---

# MUPC AI 引擎 - 模块设计文档

[DESIGN_APPROVED] — v3.1 第 2 章精简 + 预测增强管线统一描述

| 版本 | 日期       | 作者   | 状态 |
| ---- | ---------- | ------ | ---- |
| v3.1 | 2026-06-21 | 架构师 | [DESIGN_APPROVED] — 第 2 章精简（移除过时字段与 TCN 引用，新增第 14 章交叉引用），对齐 PRD v3.1 |
| v3.0 | 2026-06-21 | 架构师 | [DESIGN_APPROVED] — 合并预测增强分层混合架构（~1876行） |
| v2.15 | 2026-06-17 | 架构师 | [DESIGN_APPROVED] |
| v2.14 | 2026-06-15 | 架构师 | [DESIGN_APPROVED] |
| v2.13 | 2026-06-14 | 架构师 | [DESIGN_APPROVED] |
| v2.12 | 2026-06-14 | 架构师 | [DESIGN_APPROVED] |
| v2.12 | 2026-06-14 | 架构师 | 规划中（R-04~R-07 中优先级） |
| v2.11 | 2026-06-14 | 架构师 | [DESIGN_APPROVED] |
| v2.9 | 2026-06-14 | 架构师 | 历史版本 |
| v2.8 | 2026-06-13 | 架构师 | 历史版本 |
| v2.7 | 2026-06-13 | 架构师 | 历史版本 |
| v2.6 | 2026-06-10 | 架构师 | 历史版本 |

**对应 PRD:** `docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md` v3.1 (`[REVIEWED: PASS]`)
**合并来源:** `docs/superpowers/plans/2026-06-21-预测增强分层混合架构-DESIGN.md` v3.0 (`[DESIGN_APPROVED]`，覆盖三轮：VMD+Attention / BiLSTM+误差修正 / MSSA)

---

## 目录

1. [模块架构](#1-模块架构)
2. [LSTM 模型设计](#2-lstm-模型设计)
3. [多源数据融合设计](#3-多源数据融合设计)
4. [强化学习模型设计](#4-强化学习模型设计)
   - 4.1 功能概述
   - 4.2 分层控制架构（v2.4）
   - 4.3 算法选择
   - 4.4 完整状态空间表（10 大类，78 维，v2.14）
   - 4.5 完整动作空间表（2 维，v2.15）
   - 4.6 ActionOutput 结构体 & 解析
   - 4.7 RLModel 结构体
   - 4.8 ActionValidator 约束规则校验
   - 4.9 场景切换平滑过渡（v2.10 R3）
5. [奖励函数计算模块](#5-奖励函数计算模块)
   - 5.1 功能概述 & RewardCalculator 结构体
   - 5.2 SCENE-01：台区季节性负荷模式
   - 5.3 SCENE-B1：自主套利
   - 5.4 SCENE-B2：需量控制
   - 5.5 SCENE-B3：虚拟电厂
   - 5.6 SCENE-B5：极致绿色
   - 5.7 SceneWeights 映射表
   - 5.8 折扣累积奖励机制（v2.10 R2）
   - 5.9 冲击负荷概率预测（v2.11）
   - 5.10 变压器过载分段惩罚（v2.12 R-04）
   - 5.11 电压斜率惩罚动态权重（v2.12 R-05）
   - 5.12 冲击负荷响应奖励（v2.12 R-06）
   - 5.13 P-Q 协同度阈值可配置化（v2.12 R-07）
   - 5.14 v2.13 奖励函数精细化改进
   - 5.15 v2.17 安全 RL 包装器（Safety RL Wrapper）
6. [RKNN Runtime 设计](#6-rknn-runtime-设计)
7. [ModelManager 统一调度设计](#7-modelmanager-统一调度设计)
   - 7.1 功能概述
   - 7.2 结构体
   - 7.3 full_decision_cycle() 完整流程
   - 7.4 影子模型验证+渐进式切换（v2.10 R1）
   - 7.5 自适应权重优化器（v2.11）
8. [与策略引擎集成设计](#8-与策略引擎集成设计)
9. [文件结构](#9-文件结构)
10. [配置结构](#10-配置结构)
11. [错误类型](#11-错误类型)
12. [消息总线集成](#12-消息总线集成)
13. [技术决策记录](#13-技术决策记录)
14. [LSTM 预测增强管线设计](#14-lstm-预测增强管线设计) — **v3.0 合并新增**
   - 14.1 设计目标
   - 14.2 技术选型（VMD纯Rust / Attention ONNX内嵌 / MIC离线 / BiLSTM双模型+Go/No-Go / MSSA离线 / 误差修正BiLSTM）
   - 14.3 模块划分（新增vmd.rs/prediction_pipeline.rs/pipeline_config.rs/residual_buffer.rs/model_validator.rs）
   - 14.4 数据流设计（VMD+Attention / 误差修正 / 降级路径）
   - 14.5 接口定义（VmdDecomposer / PredictionPipeline / 配置结构体 / 模型文件管理）
   - 14.6 性能预算（单模块/组合/准入条件/模型大小/内存）
   - 14.7 配置设计（prediction_enhancement YAML段 / 热加载策略）
   - 14.8 错误处理与降级（8级降级层级 / 自动升降级逻辑）
   - 14.9 测试策略
   - 14.10 风险与缓解
15. [MSSA 超参优化工具设计](#15-mssa-超参优化工具设计) — **v3.0 合并新增**
   - 15.1 模块定位
   - 15.2 文件结构
   - 15.3 MSSA 算法设计（佳点集/三群体/反向学习/Corsi变异）
   - 15.4 目标函数设计
   - 15.5 搜索空间映射
   - 15.6 收敛与终止
   - 15.7 JSON 输出格式
   - 15.8 配置文件设计
附录A. [修订记录](#附录a-修订记录)

---

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

> **v2.4 数据流说明：** Q_batt由实时电压调节器闭环控制，不经过RL模型

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

---

## 2. LSTM 模型设计

> **本章定义 LSTM 网络架构和基础推理接口。完整的预测管线设计（VMD 分解、Attention、误差修正、降级层级）见第 14 章「LSTM 预测增强管线设计」。**

### 2.1 功能概述

LSTM 时序预测模型负责预测未来 225 分钟（15 步 × 15 分钟）的光伏出力和负荷功率，为 RL 决策模型提供前瞻性输入。模型通过 ONNX 导出并部署为 .rknn，在 RK3588 NPU 上执行 INT8 推理。

### 2.2 预测规格

| 项目 | 规格 |
|------|------|
| 预测目标 | 光伏出力 (PV forecast)、负荷功率 (Load forecast) |
| 负荷分类 | 基荷（基础用电）、可调负荷（柔性负荷）、冲击负荷（概率预测） |
| 预测范围 | 225 分钟（15 步 × 15 分钟） |
| 采样间隔 | 15 分钟（900s step_seconds，与 MUPC-AI2 训练管线对齐） |
| 输入窗口 | 6 小时（24 步 × 15 分钟），MSSA 可搜索 {12, 24, 36} |
| 输出窗口 | 225 分钟（15 步 × 15 分钟），固定不可配 |
| 模型格式 | ONNX（训练）→ INT8 量化 → .rknn（部署）|
| 精度要求 | 光伏 MAPE ≤ 8.5%（R1）/ ≤ 7.5%（R2）；负荷 MAPE ≤ 13%（R1）/ ≤ 12%（R2），详见 PRD §3.11 |

### 2.3 接口定义

```rust
/// LSTM 模型输入（经 MIC 筛选后的特征序列）
pub struct LstmInput {
    /// 输入窗口步数（默认 24，由 input_window_secs / step_seconds 计算）
    pub window_size: usize,
    /// 筛选后的特征维度（默认 7，由 MIC top_k 确定）
    pub num_features: usize,
}

/// 模型预测输出
pub struct LstmOutput {
    pub pv_forecast: Vec<f32>,              // 光伏 P50 点预测（15 维）
    pub load_forecast: Vec<f32>,             // 负荷 P50 点预测（15 维）
    pub load_forecast_quantiles: Vec<f32>,   // 负荷 P90 分位数（15 维）
    pub shock_load_probability: f32,         // 冲击负荷概率
    pub base_load: f32,                      // 基础负荷（第 1 步 P50）
}
```

> **注意：** `confidence` 字段已删除（原算法基于预测序列方差，数学上无意义）。策略引擎的 confidence 来自 `ModelOutput` 元数据字段，与 LSTM 无关。

### 2.4 模型结构体

```rust
/// LSTM 预测模型
pub struct LstmModel {
    config: LstmConfig,
    runtime: RknnRuntime,
}

impl LstmModel {
    pub fn new(config: LstmConfig) -> Result<Self, AiEngineError>;
    pub async fn load(&mut self) -> Result<(), AiEngineError>;
    pub async fn predict(&self, input: &LstmInput) -> Result<LstmOutput, AiEngineError>;
    pub fn model_type(&self) -> ModelType;
    pub fn input_window_secs(&self) -> u64;
    pub fn output_horizon_secs(&self) -> u64;
}
```

### 2.5 ONNX 到 RKNN 量化流程

```
训练阶段 (x86 服务器, PyTorch):
  1. 定义 LSTM 模型 + AdditiveAttention（第 14 章）
  2. 训练至收敛
  3. torch.onnx.export() 导出 ONNX 模型（含 metadata_props 10 键）
  4. rknn-toolkit2 加载 ONNX 模型
  5. 校准数据集 INT8 量化
  6. rknn.build() 生成 .rknn 模型文件（lstm_attn ≤ 8MB / bilstm_attn ≤ 12MB / error_correction ≤ 3MB）

部署阶段 (RK3588):
  1. 加载 .rknn 文件到 RKNN Runtime
  2. NPU 执行 INT8 整数推理
  3. 输出 f32 预测值
```

### 2.6 预测向量长度处理

预测输出向量长度由 `LstmConfig.output_horizon_secs / step_seconds` 计算（15 步，固定）。当实际输出长度与配置不符时：
- 超出部分：截断（取前 N 个值）
- 不足部分：**返回 `OutputShapeMismatch` 错误**（不静默补零）

```rust
let output_size = self.config.output_horizon_secs as usize / self.config.step_seconds as usize;
// = 22500 / 900 = 15
if output.len() < output_size {
    return Err(AiEngineError::OutputShapeMismatch {
        expected: output_size,
        actual: output.len(),
    });
}
```

---

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

### 3.5 FusedSystemState 结构体（v2.14：34 字段，78 维输入向量）

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

### 3.6 to_input_vector() -- 78 维序列化（v2.14）

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

---

## 4. 强化学习模型设计

### 4.1 功能概述

RLModel 使用 MADDPG（多智能体深度确定性策略梯度）或 PPO（近端策略优化）算法，基于融合状态向量、LSTM 预测值和运行场景权重，输出 2 维动作空间（p_ref + k_droop，v2.15）的最优控制指令。load_shedding 和 pv_limit 已下沉至策略引擎（需量控制/防逆流策略独立执行），confidence 保留在 ModelOutput 内部用于校验，不再作为动作维度。

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

### 4.4 完整状态空间表（10 大类，78 维，v2.14）

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

**输入向量维度（v2.15 修正）：** 78 维 = 9(D1) + 30(D2) + 3(D3) + 3(D4) + 2(D5) + 1(D6) + 1(D7) + 8(D8) + 4(D9) + 17(D10)。（v2.15 修正：D1 10→9，q_realtime_margin 仅计入 D7，不再重复）

> **注：** v2.14 D9 新增 `safety_override_consecutive` 和 `safety_override_ratio` 字段（2 维），用于精细化 SafetyOverride 惩罚计算。D9 从 2 维扩展至 4 维，输入向量从 76 维扩展至 78 维。

> **历史说明：** PRD v2.10/v2.11 中 59 维的描述不准确，实际应为 61 维（v2.10）和 76 维（v2.11）。

**电压感知 P/Q 协同控制策略（v2.7 双参数模式）：**

| 场景 | 电压特征 | P 控制 (p_ref) | Q 控制（实时模块闭环） |
|------|----------|---------------------|---------------------|
| 光伏超发 | 电压 > 1.05 p.u. | 充电 (p_ref<0) → 吸收有功消纳光伏 | 实时控制模块根据电压自动调节 |
| 台区季节性负荷 | 电压 < 0.95 p.u. | 放电 (p_ref>0) → 释放有功补充缺口 | 实时控制模块根据电压自动调节 |
| 末端低电压 | 电压 < 0.95 p.u. | 放电 (p_ref>0) — 仅当 Q 裕度不足时 | 实时控制模块优先调节 Q |

> **注：** v2.7 双参数模式将 Q 控制完全交给实时控制模块，RL 仅输出 P 控制指令（P_ref + k_droop），实现时间尺度解耦。

### 4.5 完整动作空间表（2 维，v2.15）

> **符号约定（v2.15 统一声明）：p_ref > 0 = 放电（向电网注入功率），p_ref < 0 = 充电（从电网吸收功率）。**
> 此约定与实时控制模块、MUPC-AI2 训练管线三方一致。

| 维度 | 字段名 | 类型 | 取值范围 | 单位 | 说明 | 分发路径 |
|------|--------|------|----------|------|------|----------|
| A1 | p_ref | f64 | [-50.0, 50.0] | kW | 有功基准点（负=充电，正=放电） | 核间→实时控制模块 |
| A2 | k_droop | f64 | [0.0, 30.0] | kW/V | 电压-有功下垂系数 | 核间→实时控制模块 |

> **v2.15 下沉说明：** load_shedding 下沉至 strategy-engine（需量控制策略独立执行），pv_limit 下沉至 strategy-engine（防逆流策略独立执行），confidence 保留在 ModelOutput 中（action_validator 内部校验使用）。AI 引擎仅通过核间通信下发 p_ref + k_droop 至实时控制模块。

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

### 4.6.1 旧版 ActionOutput（legacy，v2.6 及之前）

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

### 4.6.2 parse_action_output 双参数解析（v2.15）

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

### 4.6.3 parse_action_output()（v2.15 最终版）

从 RKNN Runtime 推理输出的 f32 向量解析为 ActionOutput 结构体，并在解析阶段执行 clamp 限幅。v2.15 将动作空间从 5 维精简为 2 维（p_ref + k_droop）。

```rust
/// 解析 RL 模型输出向量为 ActionOutput（v2.15，2 维动作）
///
/// 输出格式: [p_ref, k_droop]
/// v2.15: load_shedding/pv_limit/confidence 已从动作空间移除
pub fn parse_action_output(raw: &[f32], config: &ActionSpaceConfig) -> Option<ActionOutput> {
    if raw.len() < 2 {
        return None;
    }
    Some(ActionOutput {
        p_ref:    (raw[0] as f64).clamp(-config.max_batt_discharge_power, config.max_batt_charge_power),
        k_droop:  (raw[1] as f64).clamp(0.0, 30.0),  // 范围由实时控制模块提供
    })
}
```

### 4.8 ActionValidator — 约束规则校验（v2.15：4 条规则 ACT-DUAL-01~04，load_shedding/pv_limit/confidence 下沉）

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

**4 条双参数校验规则（ACT-DUAL-01 ~ ACT-DUAL-04，v2.15）：**

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

### 4.9 v2.10 短期实现：场景切换平滑过渡（R3）

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

---

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

> **v2.10 核心变更（安全覆盖惩罚）：**
> - 新增 R_safety_override 惩罚项，当 safety_override_active=true 时触发
> - AI 引擎感知被实时控制模块覆盖事件，学习避免触发覆盖的策略

> **v2.9 核心变更（RobustnessManager 集成）：**
> - dispatch_ai_decision 前进行异常检测
> - 存在异常时使用应急策略（不经过 RL 模型）

> **v2.8 核心变更（P-Q 协同度奖励）：**
> - 移除"电压硬惩罚"P_voltage_deviation，改为"行为奖励"R_PQ_coordination
> - AI 仅控制 P（p_ref），但需感知 Q 裕度做最优决策
> - 核心原则：Q 有裕度时"偷懒"省电池；Q 饱和时正确出手（低压放电/高压充电）
> - 弃光场景差异化：高电压时检查 AI 动作方向而非简单置零
> - 新增下垂系数平滑惩罚 R_smooth，防止 k_droop 极大化导致系统震荡

> **v2.5 分层架构原则（继续适用）：**
> - AI 仅在实时模块无功耗尽时才对电压偏差负责（q_realtime_margin <= 10% + 越限连续 2 步）
> - 实时模块有裕度时，电压问题由实时模块自行处理，AI 不因"旁观"被惩罚
> - 自适应损耗系数 α(s) ∈ {1.0, 0.2, 3.0} 区分"常规调度"与"应急处置"的电池损耗价值差异

**v2.10 奖励公式：**

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

**P-Q 协同度奖励 R_PQ_coordination（v2.8 新增）：**

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

**弃光奖励差异化（v2.8 改进）：**

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

**下垂系数平滑惩罚 R_smooth（v2.8 新增）：**

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

**安全覆盖惩罚 R_safety_override（v2.14 重构 + v2.15 修正）：**

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

**v2.15 修正说明：** D9 字段表已无 `safety_override_reason_code`（仅 4 维：active/p_ref/consecutive/ratio），样本不足时无 reason 数据可用，故删除 v2.13 引入的 reason 差异化惩罚分支。固定惩罚 -3.33 取原 voltage_violation 档位（最常见原因）。

**系数说明（v2.14 校准）：**

| 系数 | 值 | 说明 |
|------|------|------|
| k_override | 5.0 | 覆盖比例惩罚系数 |
| k_consecutive | 10.0 | 连续触发次数惩罚系数 |
| min_sample_threshold | 10 | 最小样本阈值 |
| norm_divisor | 15.0 | 归一化除数 |
| cold_start_penalty | 3.33 | 样本不足时固定惩罚（v2.15 新增，替代原 reason 差异化） |

**互斥惩罚逻辑（v2.14 新增）：**

当 `safety_override_active = true` 时，跳过该步的 **P-Q 协同度惩罚**，避免同一次事件双重惩罚：

```rust
let r_pq = if state.safety_override_active {
    0.0  // 互斥：SafetyOverride 事件不重复惩罚
} else {
    self.calc_pq_coordination(state, action.p_ref)
};
```

**权重表（v2.10）：**

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

**Rust 代码实现（v2.8）：**

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

**RewardCalculator 结构体变更（v2.8）：**

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

### 5.9 v2.10 短期实现：折扣累积奖励机制（R2）

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

### 5.10 v2.11 中期实现：冲击负荷概率预测

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

**ProbabilisticLoadOutput（v2.16 重构）：**

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

**FusedSystemState 扩展（v2.11）：**

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

### 5.11 v2.12 中优先级实现：变压器过载分段惩罚（R-04）

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

### 5.12 v2.12 中优先级实现：电压斜率惩罚动态权重（R-05）

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

### 5.13 v2.12 中优先级实现：冲击负荷响应奖励（R-06）

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

### 5.14 v2.12 中优先级实现：P-Q 协同度阈值可配置化（R-07）

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

### 5.15 v2.13 奖励函数精细化改进

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

---

### 5.16 v2.17 高优先级实现：安全 RL 包装器（Safety RL Wrapper）

> **状态：** `[DESIGN_APPROVED]` | **设计日期：** 2026-06-18 | **批准轮次：** 第三轮（修复 D-01/D-02/D-03）

> **来源**：`docs/TODO/安全RL包装器.md` + `docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md §3.7`

#### 5.16.1 需求描述

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

#### 5.16.2 数据结构

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

#### 5.16.3 核心算法：check_and_fallback 流程

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

#### 5.16.4 ModelManager 集成

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

#### 5.16.5 配置结构

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

#### 5.16.6 Web API 设计（SSE 推送为主，HTTP API 仅用于状态查询）

**架构变更**（v2.17 设计修订）：
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

**Web UI 集成**（修改 §5.16.7）：
- 使用 `EventSource` API 订阅 SSE
- 收到 `SafetyWrapperUpdate` 事件即更新 UI
- 无需任何轮询代码
```

#### 5.16.7 Web UI 设计

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

**API 调用**（v2.17 设计修订：SSE EventSource 订阅模式，**无轮询**）：

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

#### 5.16.8 错误处理

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

#### 5.16.9 测试策略

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

#### 5.16.10 影响文件

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

> **v2.17 设计修订（D-01/D-02/D-03 修复）**：
> 1. 事件流采用 `tokio::sync::broadcast`（AI 引擎 → Web API → SSE → Web UI），不依赖 HTTP 轮询
> 2. AI 引擎 `event_sender: Option<SafetyEventSender>` 字段，main.rs 注入 Sender
> 3. Web API 订阅 broadcast Receiver，转发到现有 `SsePushService`
> 4. Web UI 用 `EventSource` 订阅 SSE 端点，零轮询开销
> 5. storage 持久化作为审计通道（与 broadcast 并行，独立存在）

#### 5.16.11 设计决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 物理模型选择 | 线性灵敏度（戴维南等效）| 5ms 预算下最简单可靠的模型 |
| 线路阻抗来源 | ai.toml 配置 | 跨台区可配，避免硬编码 |
| 检查失败时回退目标 | `last_safe_action`（上一周期有效动作）| 与现有"通信中断保持最后参数"一致 |
| 与 RobustnessManager 顺序 | SafetyRLWrapper 在前 | 事前预测先于事中应急 |
| 违规日志存储 | 消息总线 + storage 双重 | 实时性 + 持久化查询 |
| Web UI 技术 | 原生 HTML + JS（无框架）| 单页面简单，无构建工具链 |
| 拒绝率告警阈值 | 20%（可配置）| 经验值，需现场调优 |

---

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

---

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

### 7.4 影子模型验证+渐进式切换（v2.10 R1）

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

#### 7.5.2 接口定义

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

#### 7.5.3 核心结构

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

### 7.5 自适应权重优化器（v2.11）

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

#### 7.4.4 AdaptiveWeightOptimizer 详细设计

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

#### 7.4.5 ParetoWeightOptimizer (NSGA-II) 详细设计

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

#### 7.4.6 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| 元学习器推理失败 | 使用上一有效权重，回退告警 |
| 权重边界违规 | 强制裁剪到合法范围 |
| 性能数据缺失 | 跳过本轮优化，使用当前权重 |
| NSGA-II 收敛失败 | 返回空 Pareto 前沿，不更新权重 |

#### 7.4.7 测试策略

| 测试项 | 验收条件 | 测试方法 |
|--------|---------|---------|
| AWO-01 | 加载配置 | 配置正确解析，无 panic |
| AWO-04 | 权重约束 | 权重为正，归一化和正确 |
| AWO-05 | 调整幅度限制 | 单次变化不超过 20% |
| AWO-07 | 权重更新后 RL 推理 | 推理延迟 < 1s |

---

## 8. 与策略引擎集成设计

### 8.1 集成架构

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

### 8.2 AiIntegrator 扩展

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

### 8.3 AiCommandValidator 扩展

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

### 8.4 兜底策略联动

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

---

## 9. 文件结构

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

### 9.1 lib.rs 模块导出

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

---

## 10. 配置结构

### 10.1 AiEngineConfig

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

### 10.1b 预测增强配置 (v3.0 新增)

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

### 10.1c YAML 配置 -- prediction_enhancement 段 (v3.0 新增)

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

### 10.2 子配置结构定义

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

**RewardThresholdConfig（v2.5）结构定义：**

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

### 10.3 枚举类型定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum QuantizationType { FP32, FP16, INT8 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType { LSTM, MADDPG, PPO }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlAlgorithm { MADDPG, PPO }
```

---

## 11. 错误类型

### 11.1 AiEngineError 枚举

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

### 11.2 错误分类与处理策略

| 错误类别 | 错误变体 | 恢复策略 |
|----------|----------|----------|
| 模型加载 | `ModelLoadFailed`, `VersionMismatch`, `ModelValidationFailed` | 拒绝启动，记录 ERROR，触发降级 |
| 推理运行时 | `InferenceFailed`, `RknnError`, `InputShapeMismatch`, `OutputShapeMismatch` | 重试 1 次，失败后记录 ERROR，连续 3 次后触发 NPU 降级 |
| 资源状态 | `ModelNotLoaded`, `ResidualBufferInsufficient` | 等待模型加载完成；残差缓冲不足时零填充或拒绝推理 |
| 数据异常 | `FusionFailed`, `DataSourceStale` | 按缺失数据处理策略填充，连续 10 周期后触发降级 |
| 运维操作 | `ModeSwitchFailed`, `ActionValidationFailed`, `OnlineUpdateFailed` | 记录 WARN，操作回滚 |
| 硬件异常 | `NpuOverheating` | 降频保护，连续 5 周期正常后恢复 |
| **v3.0 预测增强** | `VmdFailed`, `VmdNotConverged`, `AttentionDegraded`, `ErrorCorrectionFailed` | VMD 失败自动降级至无 VMD 模式；连续 5 次成功后自动升级；误差修正失败跳过修正、主预测值直出、连续 3 次失败自动禁用 |

---

## 12. 消息总线集成

### 12.1 Topic 定义

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

### 12.2 消息格式定义

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

---

## 13. 技术决策记录

### 13.1 ADR-001: INT8 量化模型部署

**决策：** 所有 AI 模型（LSTM, RL）训练后统一经 ONNX 导出，由 rknn-toolkit2 量化为 INT8 精度，以 .rknn 格式部署到 RK3588 NPU。

**理由：**
- INT8 量化后模型大小 <= 5MB（满足 <= 5MB 要求）
- NPU 推理延迟 < 100ms（INT8 推理速度是 FP32 的 4~8 倍）
- 精度损失在可接受范围内（MAPE 增加 < 2%）
- RK3588 NPU 原生支持 INT8 推理

### 13.2 ADR-002: tokio::spawn_blocking 异步封装

**决策：** 所有 FFI 调用（rknn_init, rknn_inputs_set, rknn_run, rknn_outputs_get, rknn_destroy）通过 `tokio::task::spawn_blocking` 在后台线程执行。

**理由：**
- librknnrt.so C API 均为同步阻塞调用
- 直接在 async 上下文中调用会阻塞整个 Tokio runtime，导致其他任务饥饿
- spawn_blocking 将阻塞任务分配到专用的阻塞线程池，不占用 async worker 线程

### 13.3 ADR-003: ModeSelector（互斥模式选择器）替代 SceneClassifier（自动分类器）

**决策：** 场景确定方式从 AI 自动识别改为调度主站远程控制或策略管理员本地选择，ModeSelector 使用 `tokio::sync::Mutex` 保证互斥。

**理由：**
- 电力系统运行场景应由调度人员明确指定，而非 AI 自动推断
- 自动分类器在边界工况下可能误判，导致奖励函数和优化目标错误
- 调度主站具有全局电网调度视角，可主动下发场景切换指令
- 互斥保证（Mutex）比自动分类（概率输出）的确定性更高

### 13.4 ADR-004: 10 大类状态空间 + 78 维输入向量（v2.14 更新）

**决策：** 状态空间从最初 6 大类 48 维扩展至当前 10 大类（D1 实时数据含三相电压, D2 预测数据, D3 电价, D4 需量, D5 气象, D6 调度指令, D7 实时模块, D8 季节时段, D9 安全覆盖, D10 概率负荷），序列化为 78 维固定长度输入向量。

**理由：**
- 多类别按数据源组织便于缺失数据处理和配置管理
- 固定长度简化 RKNN Runtime 输入形状校验
- 预测向量超出时截断、不足时补零，保证维度一致性
- D1 中三相电压标幺值 (voltage_phase_a/b/c) 使 AI 引擎能感知台区电压水平，执行 P/Q 协同控制
- v2.5~v2.14 扩展：逐步增加 q_realtime_margin、季节/时段编码、安全覆盖状态、概率负荷预测等维度

### 13.5 ADR-005: 4 条双参数动作约束规则 + clamp 限幅（v2.15 更新）

**决策：** AI 模型输出 2 维动作（p_ref, k_droop）后，经 4 条双参数校验规则（ACT-DUAL-01~04）校验，违反约束时自动 clamp 到安全边界，并记录 WARN 日志。load_shedding 和 pv_limit 的约束下沉至 strategy-engine（需量控制/防逆流策略内置边界检查），confidence 保留在 ModelOutput 中用于内部校验。

**理由：**
- AI 模型输出不能直接下发给物理设备，必须经过安全校验
- 变化率约束 (ACT-DUAL-03) 保护电池设备免受功率突变损害
- 值域约束 (ACT-DUAL-01/02) 保证动作值在安全边界内
- 调度约束 (ACT-DUAL-04) 保证不超出调度指令权限
- clamp（截断）比拒绝动作更鲁棒：拒绝动作会导致控制中断，clamp 保留有效部分
- v2.15 将 load_shedding/pv_limit 下沉至策略引擎，使 AI 引擎专注于核心 P-Q 协同控制

### 13.6 ADR-006: A/B 双缓冲模型热加载

**决策：** 模型更新时采用双缓冲模式：新模型加载到独立上下文中，加载期间旧模型继续服务；新模型加载完成后原子切换，旧模型延迟释放。

**理由：**
- 不中断推理服务
- 模型切换时间 < 30ms
- 支持模型版本回滚（保留上一个稳定版本）

### 13.7 ADR-007: 三相电压从 D1 实时数据中采集（非 D5 电能质量独立分类）

**决策：** 三相电压幅值（voltage_phase_a/b/c）作为 D1 实时数据的子字段，直接进入 RL 输入向量，用于电压感知 P/Q 控制。

**理由：**
- 电压幅值是 P/Q 控制策略的必要输入（过电压 -> 吸收无功，低电压 -> 释放无功）
- 与三相不平衡治理（不涉及电池充放电，由实时控制核心独立处理）是不同用途
- 归入 D1 实时数据使数据流更简洁（无需单独的电能质量适配器），同一 intercore TCP 通道获取

---

### 13.8 ADR-008: VMD 纯 Rust 实现（v3.0 新增）

**决策：** VMD 分解器采用纯 Rust（rustfft + nalgebra），备选 FFI 调用 C libvmd。

**理由：** 无 FFI 跨编译链依赖，aarch64-openEuler 目标直接 cross build；VMD 核心是 ADMM 迭代（FFT + 矩阵运算），Rust 生态已成熟；去掉 unsafe 边界。

**位置：** [14.2.1](#1421-vmd-分解纯-rust-实现)

### 13.9 ADR-009: Attention ONNX 图内嵌（v3.0 新增）

**决策：** Attention 机制由 MUPC-AI2 训练管线在 ONNX 导出时嵌入计算图，不 Rust 侧实现。

**理由：** Rust 侧零代码改动；Attention 计算在 NPU 上执行，延迟增量 <= 15%；降低 Rust 侧维护负担。

**位置：** [14.2.2](#1422-attention-机制onnx-图内嵌)

### 13.10 ADR-010: MIC 离线 Python（v3.0 新增）

**决策：** MIC 特征筛选由 MUPC-AI2 项目中独立 Python 脚本完成（minepy 库），JSON 输出。

**理由：** MIC 仅在训练阶段离线执行，不进入 RK3588 部署路径；Python minepy 是 MIC 计算的权威实现；Rust 生态无成熟 MIC crate。

**位置：** [14.2.3](#1423-mic-特征筛选离线-python-脚本)

### 13.11 ADR-011: BiLSTM 双模型文件 + Go/No-Go 准入（v3.0 新增）

**决策：** 训练管线导出两个独立的 .rknn 模型（`lstm_attn.rknn` + `bilstm_attn.rknn`），硬件验证准入。

**理由：** 推理路径零分支；独立文件便于 OTA 独立升级/回滚；ONNX metadata_props 交叉校验模型类型与配置一致性。

**位置：** [14.2.5](#1425-bilstm-可选性双模型文件--gono-go-准入条件)

### 13.12 ADR-012: 误差修正独立模型 + 独立 Runtime（v3.0 新增）

**决策：** 误差修正 BiLSTM 作为独立 .rknn 模型（`error_correction.rknn`），拥有独立的 RknnRuntime 实例，与主预测模型完全分离。

**理由：** 模型独立性（输入维度不同）、运行时独立性（独立 OTA 升级/降级）、RknnRuntime 实例化成本低（< 50ms）、完全隔离保证误差修正模型故障不影响主预测。

**位置：** [14.2.7](#1427-误差修正-bilstm独立模型--独立-runtime)

### 13.13 ADR-013: 误差修正 RknnRuntime 挂在 PredictionPipeline（v3.0 新增）

**决策：** 误差修正 RknnRuntime (#2) 挂在 `PredictionPipeline` 而非 `ModelManager` 下。

**理由：** 关注点分离 -- 误差修正与主预测在 PredictionPipeline 内串行编排，同一生命周期管理更简单；`ModelManager` 不应感知误差修正的内部实现细节。

**位置：** [14.5.4](#1454-模型文件管理r2-新增)

### 13.14 ADR-014: BiLSTM/误差修正/单向 LSTM 三模型独立 OTA（v3.0 新增）

**决策：** 任意一个模型的升级/回滚不影响其他两个模型。

**理由：** 降低 OTA 耦合风险 -- 单向 LSTM 回滚不应影响误差修正，BiLSTM 升级不应需要同步升级误差修正。`ModelManager` 通过版本号元数据追踪各模型当前版本。

**位置：** [14.5.4](#1454-模型文件管理r2-新增)

### 13.15 ADR-015: data_fusion.rs 不修改（v3.0 新增）

**决策：** MIC 离线筛选后 ONNX 维度固定，Rust 侧无需适配。

**理由：** 特征增减在训练管线一侧（MUPC-AI2）处理，ONNX 模型使用固定特征集；Rust 侧按 ONNX 输入 shape 构造 LstmInput 即可；`FusedSystemState` 序列化逻辑不感知特征维度变化。

**位置：** [14.3.3](#1433-不修改的模块)

### 13.16 ADR-016: MSSA 算法参数与编码（v3.0 新增）

**决策：** N=30, p_d=0.2, p_s=0.1, 佳点集初始化, 13 维混合编码（离散索引+浮点+One-hot枚举）对应 10 维逻辑超参, epsilon=1e-4 收敛。

**位置：** [15.3](#153-mssa-算法设计)

### 13.17 ADR-017: MSSA 目标函数与缓存（v3.0 新增）

**决策：** 加权 MAPE = 0.5*MAPE_pv + 0.5*MAPE_load, 惩罚分 1e6, SHA256 指纹缓存跨运行持久化, training_data_fingerprint 自动失效。

**位置：** [15.4](#154-目标函数设计)

### 13.18 ADR-018: IPSO 降级路径（v3.0 新增）

**决策：** 配置 `algorithm: "IPSO"` 一键切换, JSON Schema 保持一致, 用于 MSSA 超时或不收敛时快速收敛。

**位置：** [15.1](#151-模块定位), [15.8](#158-配置文件设计)

---

### 13.19 关键实现文件

These are the most critical files that need to be created or significantly modified to implement this design:

- `e:\MUPC2\mupc\crates\ai-engine\src\data_fusion.rs` (new: DataFusionEngine, DataSourceAdapter trait, 5 adapter implementations, FusedSystemState with to_input_vector())
- `e:\MUPC2\mupc\crates\ai-engine\src\rl_model.rs` (refactor: replace SystemState with FusedSystemState, replace old 8-field ActionOutput with new 5-field ActionOutput, add parse_action_output, add 48-dim input support)
- `e:\MUPC2\mupc\crates\ai-engine\src\reward_calculator.rs` (new: RewardCalculator with 5 scene formulas, SceneWeights lookup)
- `e:\MUPC2\mupc\crates\ai-engine\src\action_validator.rs` (new: ActionValidator with 5 constraint rules ACT-01~05, clamp logic, ViolationRecord)
- `e:\MUPC2\mupc\crates\ai-engine\src\model_manager.rs` (refactor: add full_decision_cycle(), wire in DataFusionEngine, RewardCalculator, ActionValidator)


---

## 14. LSTM 预测增强管线设计

> **v3.0 合并新增** | **来源：** `docs/superpowers/plans/2026-06-21-预测增强分层混合架构-DESIGN.md` v3.0
> **覆盖轮次：** 第一轮 VMD + Attention (R1) / 第二轮 BiLSTM + 误差修正 (R2) / 第三轮 MSSA (独立章节 15)
> **PRD 对应：** `docs/superpowers/specs/2026-06-21-预测增强分层混合架构-PRD.md`

### 14.1 设计目标

在现有 `mupc-ai-engine` LSTM 时序预测管线之上，以分层叠加方式集成 VMD 信号分解 + Attention 注意力机制 + 可选 BiLSTM + 残差修正管线，提升光伏出力与台区负荷的预测精度。

**第一轮代码状态：** VMD 纯 Rust 分解器（`vmd.rs`）、预测增强管线编排器（`prediction_pipeline.rs`）、增强配置（`pipeline_config.rs`）均已实现并通过设计评审，Attention 由 MUPC-AI2 训练管线嵌入 ONNX 计算图。

**第二轮新增代码（本设计覆盖）：** BiLSTM 双模型文件加载与 Go/No-Go 准入、误差修正 BiLSTM 独立推理与残差修正管线、扩展 PipelineHealth 为模块级健康状态数组、YAML 配置扩展。

### 14.2 技术选型

#### 14.2.1 VMD 分解：纯 Rust 实现

| 维度 | 决策 |
|------|------|
| **方案** | 纯 Rust，依赖 `rustfft`（FFT）+ `nalgebra`（矩阵 SVD/特征分解） |
| **理由** | (1) 无 FFI 跨编译链依赖，aarch64-openEuler 目标直接 `cross build`；(2) VMD 核心是 ADMM 迭代（FFT + 矩阵运算），Rust 生态已成熟；(3) 去掉 `unsafe` 边界，`cargo test` 全量覆盖 |
| **备选** | FFI 调用 C `libvmd`：优势是已有验证实现，劣势是 aarch64 交叉编译 fragile、unsafe FFI 边界、调试困难。若 Rust 实现数值稳定性不达标或性能超标，可切换为 FFI 路径 |
| **Python 预处理** | 不适用 -- 推理阶段需要实时 VMD，不能预计算 |
| **性能风险** | 若单次 VMD > 50ms，启用 `rayon` 并行化 K 个 IMF 的 ADMM 迭代（每个 IMF 独立求解，天然可并行） |

**依赖 crate：** `rustfft = "6.2"` (aarch64 NEON 优化)、`nalgebra = "0.33"` (矩阵运算)；`rayon = "1.10"` 可选并行。

#### 14.2.2 Attention 机制：ONNX 图内嵌

| 维度 | 决策 |
|------|------|
| **方案** | MUPC-AI2 训练管线在 ONNX 导出时将 Attention 节点嵌入计算图（MatMul + Softmax + ReduceSum 均为 ONNX 标准算子），RKNN Toolkit 2 一并转换为 .rknn |
| **理由** | (1) Rust 侧零代码改动 -- 现有 `RknnRuntime::run()` 直接消费含 Attention 的 .rknn 模型；(2) Attention 计算在 NPU 上执行，延迟增量 <= 15%；(3) 降低 Rust 侧维护负担 |
| **备选** | Rust 后处理：ONNX 仅输出 LSTM hidden states，Rust 侧 CPU 计算 Attention。劣势：数据 NPU->CPU 搬运开销 + Rust 侧额外计算 + 延迟超标风险。仅保留为调试/可视化用途 |
| **Attention 权重导出** | ONNX 模型增加一个输出节点 `attention_weights`（shape=[input_window]），Rust 侧可选读取并记录日志用于可视化分析 |

#### 14.2.3 MIC 特征筛选：离线 Python 脚本

| 维度 | 决策 |
|------|------|
| **方案** | MUPC-AI2 项目中独立 Python 脚本，使用 `minepy` 库计算 MIC，输出 JSON 结果文件 |
| **理由** | (1) MIC 仅在训练阶段离线执行，不进入 RK3588 部署路径；(2) Python `minepy` 是 MIC 计算的权威实现；(3) Rust 生态无成熟 MIC crate |
| **与 Rust 的接口** | JSON 文件。Rust 侧 `PredictionEnhancementConfig` 可引用 `mic_top_k` 值以确定特征维度；实际特征筛选在训练阶段完成，ONNX 输入维度已固定 |

#### 14.2.4 MSSA 超参搜索：纯 Python 离线工具（概述）

| 维度 | 决策 |
|------|------|
| **方案** | MUPC 项目 `tools/mssa_optimizer/` 目录下的纯 Python 模块，实现 MSSA（多策略麻雀搜索算法），输出符合 PRD Section 7.4.2 JSON Schema 的最优超参配置，供 MUPC-AI2 训练管线消费 |
| **Python 版本** | Python >= 3.9，仅依赖 numpy >= 1.24、scipy >= 1.10、pyyaml >= 6.0 |
| **备选（IPSO 降级路径）** | IPSO 作为备选方案。当 MSSA 搜索时间超预期或收敛不稳定时，通过配置文件 `algorithm: "IPSO"` 切换 |
| **与 Rust 的接口** | JSON 文件。Rust 侧不消费 MSSA 结果 -- MSSA 输出直接由训练管线读取，映射为 ONNX export 参数。Rust 推理端仅感知最终 ONNX 模型维度，对搜索过程透明 |

> 完整的 MSSA 算法设计、目标函数、搜索空间映射、JSON 输出格式和配置文件设计见 [第 15 章](#15-mssa-超参优化工具设计)。

#### 14.2.5 BiLSTM 可选性：双模型文件 + Go/No-Go 准入条件

| 维度 | 决策 |
|------|------|
| **方案** | 训练管线导出两个独立的 .rknn 模型：`lstm_attn.rknn`（单向 LSTM + Attention）和 `bilstm_attn.rknn`（BiLSTM + Attention）。`ModelManager` 根据 `PredictionEnhancementConfig.bilstm.enabled` 和准入条件共同决定加载哪个模型 |
| **理由** | (1) 推理路径零分支 -- 选中模型后 `RknnRuntime::run()` 逻辑完全不变；(2) 独立文件便于 OTA 独立升级/回滚；(3) 通过 ONNX `metadata_props` 中的 `mupc_model_type` 校验模型类型与配置一致性 |

**双模型文件命名约定与校验：**

| 文件名 | 模型类型 | 要求 | 用途 |
|--------|----------|------|------|
| `lstm_attn.rknn` | 单向 LSTM + Attention | 必须存在 | 第一轮部署 + BiLSTM No-Go 回退 |
| `bilstm_attn.rknn` | BiLSTM + Attention | 第二轮按需发布 | Go 路径下替换单向模型 |

**ONNX metadata_props 交叉校验：** `mupc_model_type` ("lstm"/"bilstm")、`mupc_with_attention` ("true")、`mupc_hidden_size`、`mupc_num_layers`、`mupc_input_window`、`mupc_direction` ("forward"/"bidirectional")、`mupc_version`。启动时与配置交叉校验，不一致时记录 WARN 并自动回退。

**Go/No-Go 准入条件 -- 双重门控：**

| 门控 | 配置项 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| **配置门** | `bilstm.enabled` | `bool` | `false` | 运维人员在配置文件中主动开启 |
| **硬件验证门** | `bilstm.gate_passed` | `bool` | `false` | 使用 BiLSTM 原型 .rknn 在 RK3588 上完成 P99 延迟摸底后，由运维手动设为 `true` |

**硬件验证条件：** 使用 BiLSTM + Attention 原型模型在 RK3588 NPU 上运行 >= 1000 次连续推理，P99 推理延迟 < 900ms（为 VMD 50ms + 误差修正 200ms 留 750ms 裕度 —— 注意 900ms 门限是**全管线**延迟的预留上限，单 BiLSTM 推理本身应远小于此值）。

**准入失败处理（No-Go）：** 若 P99 >= 900ms，操作步骤：
1. 保持 `gate_passed = false`（永不为 `true`）
2. `enabled` 配置无效，系统始终加载单向 LSTM 模型
3. 第二轮仅部署误差修正管线（`error_correction.rknn`），跳过 BiLSTM 双向替换
4. 运维在变更日志中记录 No-Go 原因和基准测试数据

**模型加载选择逻辑：**

```rust
// ModelManager::load_models() 中的选择逻辑（伪代码）
fn select_prediction_model(config: &PredictionEnhancementConfig) -> ModelSelection {
    let gate_passed = config.bilstm.gate_passed;
    let bilstm_enabled = config.bilstm.enabled;

    match (bilstm_enabled, gate_passed) {
        // Go 路径：配置启用 + 硬件验证通过 → 加载 BiLSTM 模型
        (true, true) => {
            let path = config.bilstm.model_path
                .as_deref()
                .unwrap_or_else(|| Path::new("/etc/mupc/models/bilstm_attn.rknn"));
            ModelSelection::BiLstm(path.to_path_buf())
        }
        // No-Go 路径 1：配置启用但硬件未验证 → 回退单向，记录 WARN
        (true, false) => {
            tracing::warn!("BiLSTM 配置启用但 gate_passed=false（未通过 RK3588 延迟摸底），回退到单向 LSTM");
            let path = Path::new("/etc/mupc/models/lstm_attn.rknn");
            ModelSelection::UniLstm(path.to_path_buf())
        }
        // No-Go 路径 2：默认单向
        (false, _) => {
            let path = Path::new("/etc/mupc/models/lstm_attn.rknn");
            ModelSelection::UniLstm(path.to_path_buf())
        }
    }
}
```

#### 14.2.6 Attention 配置校验

| 维度 | 决策 |
|------|------|
| **方案** | `AttentionConfig.enabled` 是调试开关：当 ONNX 模型含 Attention 层时，运维可临时设为 `false` 禁用 Attention（退化到等权重模式）。若模型不含 Attention 但配置 `enabled=true`，启动时校验 metadata 后记录 WARN 并自动回退 |
| **校验时机** | `ModelManager` 加载模型后，查询 `mupc_with_attention` 元数据，与 `AttentionConfig.enabled` 交叉校验 |
| **不做的事** | Rust 侧不实现 Attention 计算，不实现所有权重相等检测（退化由训练管线负责） |

#### 14.2.7 误差修正 BiLSTM：独立模型 + 独立 Runtime

| 维度 | 决策 |
|------|------|
| **方案** | 误差修正 BiLSTM 作为独立 .rknn 模型 (`error_correction.rknn`)，拥有独立的 `RknnRuntime` 实例，与主预测模型完全分离 |
| **理由** | (1) **模型独立性**：误差修正 BiLSTM 输入为历史残差序列（维度不同于主预测模型输入的特征向量），必须独立建模；(2) **运行时独立性**：主模型和修正模型可能在不同时机加载/卸载、独立 OTA 升级、独立降级；(3) `RknnRuntime` 实例化成本低（< 50ms），两个实例同时驻留在 NPU 内存中是 RK3588 支持的标准操作；(4) 完全隔离意味着误差修正模型的任何故障不影响主预测管线输出 |
| **备选** | 方案 A -- ONNX 图内嵌：将误差修正节点嵌入主预测 ONNX 计算图。劣势：输入维度不同，需要 ONNX 分支处理，RKNN 对动态分支支持不确定。方案 B -- 同一 RknnRuntime 实例切换模型：需频繁 `rknn_destroy`/`rknn_init` 来回切换，延迟累积不可接受 |
| **参数量约束** | <= 主预测 LSTM 参数的 50%（PRD ERR-02），INT8 模型文件大小 <= 3MB |

**残差输入构建：**

误差修正 BiLSTM 的输入 = 最近 T 步的观测残差序列（实际值 - 预测值）。`T = residual_window_steps`，默认 24 步（与主预测 input_window 对齐，可通过配置调整）。设计将输入窗口设为可配置的 `residual_window_steps`（默认 24），与主预测 `input_window` 对齐以保持架构一致性；误差修正的 output horizon 为 15 步，与主预测匹配。

```rust
/// 残差滑动窗口缓冲
pub struct ResidualBuffer {
    /// 容量 = residual_window_steps（默认 24）
    capacity: usize,
    /// 循环缓冲（FIFO）
    buffer: VecDeque<f32>,
    /// 是否已填满（未填满时使用零向量填充 → zero_init=true）
    filled: bool,
}
```

**输入构建规则：**

| 场景 | 输入内容 | 说明 |
|------|----------|------|
| 缓冲已满（>= T 步残差） | 最近 T 步残差值 `[e_{t-T+1}, ..., e_t]` | 正常推理路径 |
| 缓冲未满（冷启动/模型刚加载）且 `zero_init=true` | T 步零向量 `[0.0; T]` | 默认行为，不产生修正（y_corrected = y_pred + 0 = y_pred） |
| 缓冲未满且 `zero_init=false` | 拒绝推理，返回 `ErrorCorrectionFailed` | 保守模式（生产环境不推荐） |
| 残差序列含 NaN/Inf | 替换为该位置零值，记录 WARN | 鲁棒性保护 |

**在线残差更新：** 每次主预测完成后，等待下一周期实际值到达时更新 `ResidualBuffer`。每个预测对象（PV、Load）各维护一个独立的 `ResidualBuffer`。

**与主预测管线的并行/串行关系：** 误差修正推理与主预测推理是**严格串行**关系：

```
Step 1: 主预测推理 (NPU, RknnRuntime #1)
  └─→ y_pred [15 维]

Step 2: 残差输入构建 (CPU)
  └─→ ResidualBuffer::build_input() → e_history [T 维]

Step 3: 误差修正推理 (NPU, RknnRuntime #2)
  └─→ e_pred [15 维]

Step 4: 修正输出 (CPU)
  └─→ y_corrected = y_pred + e_pred [15 维]
```

**不可并行的理由：** NPU 同一时刻只能执行一个推理任务（RK3588 NPU 单任务），两个 RknnRuntime 的推理必须串行。误差修正总延迟 ≈ 两次修正推理（PV + Load），<= 200ms。

### 14.3 模块划分

#### 14.3.1 新增模块

| 模块 | 文件 | 职责 | 轮次 |
|------|------|------|------|
| **VMD 分解器** | `ai-engine/src/vmd.rs` | 纯 Rust VMD 算法实现 | R1 |
| **预测增强管线** | `ai-engine/src/prediction_pipeline.rs` | 串联 VMD + NPU 推理 + IMF 重构 + 误差修正的编排器；统一管理增强模块的启用/降级状态 | R1（VMD 编排）+ R2（误差修正编排） |
| **增强配置** | `ai-engine/src/pipeline_config.rs` | VMD/Attention/BiLSTM/误差修正/特征筛选的配置结构体 | R1（VMD+Attention）+ R2（BiLSTM/ErrorCorrection 扩展） |
| **残差缓冲** | `ai-engine/src/residual_buffer.rs` | 残差滑动窗口缓冲管理（R2 新增） | R2 |
| **模型文件校验器** | `ai-engine/src/model_validator.rs` | ONNX metadata_props 读取 + 与配置交叉校验（R2 新增） | R2 |

#### 14.3.2 修改模块

| 模块 | 文件 | 改动内容 | 轮次 |
|------|------|----------|------|
| **LSTM 配置** | `ai-engine/src/config.rs` | `LstmConfig` 新增 `prediction_enhancement: Option<PredictionEnhancementConfig>` 字段 | R1 |
| **LSTM 模型** | `ai-engine/src/lstm_model.rs` | 新增 `predict_with_vmd()` 方法；VMD 状态管理 | R1 |
| **模型管理器** | `ai-engine/src/model_manager.rs` | `full_decision_cycle()` 集成 PredictionPipeline；**R2 新增**：多 RknnRuntime 管理（1-3 个 .rknn 模型文件）、误差修正状态管理 | R1 + R2 |
| **错误类型** | `ai-engine/src/error.rs` | R1 新增 `VmdFailed`/`VmdNotConverged`/`AttentionDegraded`/`ErrorCorrectionFailed`；**R2 新增** `ModelValidationFailed`（metadata 校验失败） | R1 + R2 |
| **公共接口** | `ai-engine/src/lib.rs` | 导出新增模块 | R1 + R2 |

#### 14.3.3 不修改的模块

| 模块 | 理由 |
|------|------|
| `mupc-common` | 特征序列化逻辑在 `ai-engine/data_fusion.rs`，预测增强对下游透明 |
| `mupc-strategy-engine` | `FusedSystemState` 接口不变，`AiIntegrator` 调用 `ModelManager` 接口不变 |
| `rknn_runtime.rs` | ONNX 内嵌 Attention = 对 Runtime 透明，推理调用不变；R2 误差修正使用同一 RknnRuntime 抽象（R2 新增第二个实例） |
| `data_fusion.rs` | **不修改**。理由：MIC 离线筛选完成于训练阶段，筛选后的特征维度已固定在 ONNX 输入 shape 中。Rust 侧 `FusedSystemState` 序列化逻辑不感知特征维度变化——特征增减在训练管线一侧（MUPC-AI2）处理，ONNX 模型使用固定特征集，Rust 侧按 ONNX 输入 shape 构造 LstmInput 即可。与 PRD Section 7.1 的差异：PRD 原文提及 `data_fusion.rs` 需适配，但经设计评审确认 MIC 离线筛选 → ONNX 输入维度固定 → `data_fusion.rs` 和 `lstm_model.rs` 无需适配（特征集变更在训练阶段完成，Rust 部署侧所见 ONNX 已是固定维度） |

### 14.4 数据流设计

#### 14.4.1 第一轮全功能数据流（VMD + Attention）

```
+-----------------------------------------------------------------------------+
|  PredictionPipeline::execute(state, config)                                  |
|                                                                              |
|  Step 1: 特征提取                                                            |
|    lstm_history --> [pv_history_24, load_history_24]                        |
|                                                                              |
|  Step 2: VMD 分解 (CPU, <= 50ms)                                            |
|    pv_history_24 --> VMD(K_pv, alpha, tol, max_iter)                        |
|                    --> [IMF_1, IMF_2, ..., IMF_K]  (K 个 24 维向量)          |
|    load_history_24 --> VMD(K_load, ...) --> [IMF_1, ..., IMF_K]            |
|                                                                              |
|  Step 3: 逐 IMF NPU 推理 (K 次, NPU)                                         |
|    for each IMF_i:                                                           |
|      LstmInput { history: IMF_i } --> RknnRuntime::run()                   |
|      --> LstmOutput { predictions: [15] }  // 含 Attention 加权              |
|                                                                              |
|  Step 4: 重构 (CPU)                                                          |
|    PV_pred = Sigma IMF_predictions  (逐元素求和，15 维)                       |
|    Load_pred = Sigma IMF_load_predictions (15 维)                           |
|                                                                              |
|  Step 5: 分位数后处理 (CPU, 复用现有逻辑)                                      |
|    Load_pred --> predict_quantiles() --> ProbabilisticLoadOutput            |
|                                                                              |
|  Step 6: [R2 新增出口] 误差修正入口                                           |
|    PV_pred / Load_pred --> ErrorCorrectionPipeline (见 14.4.3)               |
|                                                                              |
|  Step 7: 注入 FusedSystemState                                               |
|    FusedSystemState.pv_forecast_15min = PV_final (f64)                      |
|    FusedSystemState.load_forecast_15min = Load_final (f64)                  |
|    FusedSystemState.D10 字段 = step_5 输出                                    |
+-----------------------------------------------------------------------------+
```

#### 14.4.2 VMD 分解失败时的降级数据流

```
    VMD 失败 (不收敛 / NaN / 超时)
      |
      +--> 记录 WARN 日志 + 递增失败计数
      +--> 使用原始序列 (未分解) 直接送入 LSTM
      +--> 后续推理同 Step 3-7（单次推理，无 IMF 循环）
      +--> 连续 5 次成功后自动升回 VMD 模式
```

#### 14.4.3 误差修正数据流（R2 实现）

```
    主预测 PV/Load_pred (Step 4 输出)
      |
      +--> 直接输出 (error_correction.enabled = false)
      |
      +--> [error_correction.enabled = true AND residual_buffer.filled = true]
            |
            | Step EC-1: 残差输入构建 (CPU, < 1ms)
            |   ResidualBuffer::build_input()
            |     ├── 缓冲已满 (>= T 步): 取出最近 T 步残差值
            |     └── 缓冲未满 (< T 步): zero_init=true → 零向量填充
            |   --> e_history [T 维 float32]
            |
            | Step EC-2: 误差修正推理 (NPU, RknnRuntime #2, <= 100ms × 2)
            |   for target in [PV, Load]:
            |     LstmInput { history: e_history_target }
            |     --> error_correction.rknn (轻量 BiLSTM)
            |     --> e_pred [15 维 float32]
            |
            | Step EC-3: 修正输出 (CPU, < 1ms)
            |   for target in [PV, Load]:
            |     y_corrected[target] = y_pred[target] + e_pred[target]
            |
            | Step EC-4: 残差缓冲更新 (下个周期)
            |   等待本周期实际值 y_actual 到达后:
            |     e_new = y_actual - y_pred (每个预测步独立计算)
            |     ResidualBuffer::push(e_new)
            |
            +--> y_corrected [15 维] --> Step 7 注入 FusedSystemState
```

**误差修正降级数据流（残差缓冲不完整）：**

```
    error_correction.enabled = true AND residual_buffer 未满 AND zero_init = true
      |
      +--> e_history = 零向量 [0.0; T]
      +--> e_pred = [0.0; 15] (理论输出，实际推理可跳过)
      +--> y_corrected = y_pred + 0 = y_pred (等效于直接输出)
      +--> 记录 INFO: "残差缓冲未满 ({filled}/{capacity})，零填充跳过修正"
```

**误差修正失败降级数据流：**

```
    误差修正推理失败 (RknnRuntime #2 错误 / 输出 NaN / 超时)
      |
      +--> 记录 ERROR: "误差修正推理失败: {reason}"
      +--> error_correction_consecutive_failures += 1
      +--> y_corrected = y_pred (使用主预测值，跳过修正)
      +--> 连续 3 次失败后持久化禁用 (error_correction.enabled = false, 写入 DB)
      +--> 恢复: OTA 下发新版 error_correction.rknn 后手动重新启用
```

#### 14.4.4 与现有 `full_decision_cycle()` 的集成点

当前 `ModelManager::full_decision_cycle()` 中 LSTM 预测调用链为：

```rust
// 现有路径 (model_manager.rs:204-207)
let (pv_forecast, load_forecast, load_quantiles) = self
    .run_lstm_predict_with_quantiles()
    .await
    .unwrap_or_else(|_| (vec![0.0; 15], vec![0.0; 15], None));
```

增强后切换为：

```rust
// 增强路径 (model_manager.rs: 替换上述调用)
let pipeline = self.prediction_pipeline.read().await;
let forecast_result = pipeline
    .as_ref()
    .map(|p| p.execute().await)
    .unwrap_or_else(|| self.run_lstm_predict_with_quantiles().await); // 降级
```

**R2 扩展：** `p.execute()` 内部自动处理误差修正（根据 `ErrorCorrectionConfig.enabled` 决定是否调用 `execute_error_correction()`）。

### 14.5 接口定义

#### 14.5.1 VmdDecomposer

```rust
/// VMD 分解器
pub struct VmdDecomposer {
    config: VmdConfig,
}

pub struct VmdConfig {
    /// 模态数 K（光伏 4~6，负荷 5~8）
    pub k: usize,
    /// 惩罚因子
    pub alpha: f64,
    /// 噪声容忍度（Lagrangian 更新步长，tau=0.0 不做双升更新）
    pub tau: f64,
    /// 收敛容差
    pub tol: f64,
    /// 最大迭代次数
    pub max_iter: usize,
}

/// 单次 VMD 分解结果
pub struct VmdResult {
    /// K 个子模态，每个长度 = 输入序列长度
    pub imfs: Vec<Vec<f32>>,
    /// 重构序列（所有 IMF 求和）
    pub reconstructed: Vec<f32>,
    /// 重构误差 (RMSE)
    pub reconstruction_error: f64,
    /// 实际迭代次数
    pub iterations: usize,
    /// 是否收敛
    pub converged: bool,
}

impl VmdDecomposer {
    pub fn new(config: VmdConfig) -> Self;
    pub fn decompose(&self, signal: &[f32]) -> Result<VmdResult, AiEngineError>;
}
```

#### 14.5.2 PredictionPipeline（R2 扩展）

```rust
/// 预测增强管线（R2 扩展：新增误差修正推理、ResidualBuffer 集成）
pub struct PredictionPipeline {
    // --- R1 字段 ---
    vmd_pv: Option<VmdDecomposer>,
    vmd_load: Option<VmdDecomposer>,
    lstm_model: Arc<RwLock<Option<LstmModel>>>,
    lstm_history: Arc<RwLock<VecDeque<(f64, f64)>>>,
    input_size: usize,
    config: PredictionEnhancementConfig,
    health: RwLock<PipelineHealth>,

    // --- R2 新增字段 ---
    /// 误差修正 RknnRuntime（独立实例，与主模型 Runtime 隔离）
    error_correction_runtime: Option<RknnRuntime>,
    /// PV 残差缓冲
    residual_buffer_pv: Option<RwLock<ResidualBuffer>>,
    /// Load 残差缓冲
    residual_buffer_load: Option<RwLock<ResidualBuffer>>,
}

/// EnhancedForecastResult
pub struct EnhancedForecastResult {
    pub pv_forecast: Vec<f64>,
    pub load_forecast: Vec<f64>,
    pub load_quantiles: Option<ProbabilisticLoadOutput>,
    pub enhancement_level: EnhancementLevel,
    pub vmd_degraded: bool,
    /// R2 新增：误差修正是否生效（true = 误差修正成功执行且产生非零修正）
    pub error_correction_applied: bool,
}

/// 增强等级（降级追踪）
///
/// 注意：Level 0-3 由 PredictionPipeline 内部管理；
/// Level 4 (全降级/v2.16 基线) 和 Level 5 (全零预测/安全兜底)
/// 由 ModelManager 调用方处理。见 14.8.1 降级层级边界表格。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EnhancementLevel {
    FullVmdAttentionCorrection = 0,  // VMD + (Bi)LSTM/Attention + 误差修正
    BiLstmVmdAttention = 1,          // BiLSTM + VMD + Attention (R2 新增)
    VmdAttention = 2,                // VMD + LSTM/Attention
    AttentionOnly = 3,               // LSTM/Attention (无 VMD)
    Baseline = 4,                    // LSTM 基线 (无 VMD, 无 Attention)
    // Level 5: 全零预测 — 由 ModelManager 调用方处理，不在此枚举
}
```

**EnhancementLevel 枚举重新编号说明（R2 变更）：**
- v1.1 中 Level 0-3 为连续编号，v2.0 因 BiLSTM 中间层插入，重新编号为 0-4
- `BiLstmVmdAttention (1)` 在 VMD+Attention 之上、误差修正之下，代表"BiLSTM 替换单向 LSTM"这一独立增强维度
- 若 BiLSTM 为 No-Go（`gate_passed = false`），降级路径中 Level 1 不存在，直接从 Level 0 降级到 Level 2

**PipelineHealth 模块级健康状态（R2 扩展）：**

```rust
/// 管线模块健康状态（R2 扩展：VMD + 误差修正 双模块追踪）
///
/// 首轮仅 VMD 需硬降级追踪（仅使用 vmd_* 字段），
/// R2 扩展为模块级健康状态数组，逐一追踪每个模块的降级/升级。
#[derive(Debug, Clone)]
pub struct PipelineHealth {
    // VMD 模块
    pub vmd_consecutive_failures: u32,
    pub vmd_consecutive_successes: u32,
    // 误差修正模块（R2 新增）
    pub ec_consecutive_failures: u32,
    pub ec_consecutive_successes: u32,
    // 当前增强等级
    pub current_level: EnhancementLevel,
}
```

#### 14.5.3 PredictionEnhancementConfig

```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PredictionEnhancementConfig {
    #[serde(default)]
    pub vmd: VmdEnhancementConfig,
    #[serde(default)]
    pub attention: AttentionConfig,
    /// R2 扩展：BiLSTM 配置
    #[serde(default)]
    pub bilstm: BiLstmConfig,
    /// R2 扩展：误差修正配置
    #[serde(default)]
    pub error_correction: ErrorCorrectionConfig,
    #[serde(default)]
    pub feature_selection: FeatureSelectionConfig,
}
```

R2 新增字段定义详见 [10.1b 预测增强配置](#101b-预测增强配置-v30-新增)。

#### 14.5.4 模型文件管理（R2 新增）

**模型文件清单与校验规则：**

| 文件名 | 轮次 | 必须存在 | 加载的 RknnRuntime | 用途 |
|--------|------|----------|-------------------|------|
| `lstm_attn.rknn` | R1 | 是 | Runtime #1（主预测） | 单向 LSTM + Attention 主预测 |
| `bilstm_attn.rknn` | R2 | 否（Go/No-Go 按需） | Runtime #1（主预测，替换 lstm_attn） | BiLSTM + Attention 主预测 |
| `error_correction.rknn` | R2 | 否（error_correction.enabled 按需） | Runtime #2（误差修正，独立实例） | 轻量 BiLSTM 残差修正 |

**ModelManager 中 RknnRuntime 实例管理策略（R2 设计决策）：**

```
ModelManager
  |-- lstm_model: Arc<RwLock<Option<LstmModel>>>   // 持有 RknnRuntime #1（主预测）
  |                                                    //    可能加载 lstm_attn.rknn 或 bilstm_attn.rknn
  |-- prediction_pipeline: Arc<RwLock<Option<PredictionPipeline>>>
       |-- error_correction_runtime: Option<RknnRuntime>  // RknnRuntime #2（误差修正，独立）
```

**设计决策：** 误差修正 RknnRuntime (#2) 挂在 `PredictionPipeline` 而非 `ModelManager` 下，理由：
- 误差修正与主预测在 PredictionPipeline 内串行编排（14.4.3），同一生命周期管理更简单
- `ModelManager` 不应感知误差修正的内部实现细节（关注点分离）
- 若未来误差修正需要独立于主预测管线加载（如 OTA 热加载），可将字段提升至 `ModelManager`

**文件校验规则：**

| 校验项 | 时机 | 方法 | 失败处理 |
|--------|------|------|----------|
| 文件存在性 | 模型加载（启动时） | `std::fs::metadata` + 路径存在检查 | 记录 ERROR，若为可选模型（bilstm/error_correction）则跳过；若为必须模型（lstm_attn）则拒绝启动 |
| SHA256 完整性 | 模型加载 | 与 OTA 下发的校验值比对（预期哈希值存储在 OTA manifest JSON 或 DB `model_versions` 表中） | 触发 OTA 备份恢复流程 |
| ONNX metadata_props 一致性 | 模型加载 | RKNN runtime 查询 + 与 config 交叉比对（14.2.5 表格） | 记录 ERROR，拒绝部署；若为可选模型则跳过 |
| RKNN Runtime 版本兼容 | rknn_init | 检查返回码（-4 = SDK 版本不匹配） | 拒绝加载，等待 RKNN Runtime 升级 |
| 输入/输出维度匹配 | rknn_query | 检查 input/output 数量与期望一致 | 拒绝加载 |

**OTA 升级/回滚策略：**

| 模型文件 | 升级策略 | 回滚策略 |
|----------|----------|----------|
| `lstm_attn.rknn` | OTA 下发新版本 → SHA256 校验 → 原子替换（rename） → 下次推理周期生效 | 保留上一版本备份（`lstm_attn.rknn.bak`），SHA256 校验失败时自动恢复 |
| `bilstm_attn.rknn` | 同上；首次下发后须完成 14.2.5 硬件验证才设 `gate_passed = true` | 同 lstm_attn；可独立回滚（不影响单向 LSTM）。BiLSTM 模型版本变更（`mupc_version` 变化）后 `gate_passed` 自动重置为 `false`，需重新执行硬件验证 |
| `error_correction.rknn` | OTA 下发 → 校验 → `PredictionPipeline::reload_error_correction()` 热加载 | 回滚到上一版本或直接禁用（`enabled = false`，系统自动降级） |

**三种模型的 OTA 独立性：** 任意一个模型的升级/回滚不影响其他两个模型。`ModelManager` 通过版本号元数据（`mupc_version`）追踪各模型的当前版本。

**BiLSTM 运行时模型热切换机制：** 当 BiLSTM OTA 升级后 `gate_passed` 重置为 `false`，系统继续使用单向 LSTM。完成硬件验证并设 `gate_passed = true` 后，下一推理周期起 `ModelManager::load_models()` 重新执行 `select_prediction_model()` 切换到 BiLSTM。热切换通过预留双 Runtime（主预测 Runtime #1 保持单向 LSTM 服务，新 BiLSTM 模型加载到独立 Context 中，加载完成后原子替换 `lstm_model` 中的 Runtime 引用）实现无感切换，推理不中断。若无法预留双 Runtime（NPU 内存不足），则 fallback 到 destroy+init 路径（短暂推理中断 < 200ms）。

### 14.6 性能预算

#### 14.6.1 单模块延迟预算表

| 阶段 | 位置 | 单次预算 | 测量方法 | 轮次 |
|------|------|----------|----------|------|
| VMD 分解（PV） | CPU | <= 25ms | 1000 次 benchmark | R1 |
| VMD 分解（Load） | CPU | <= 25ms | 1000 次 benchmark | R1 |
| VMD 合计 | CPU | <= 50ms | 串行 PV + Load | R1 |
| LSTM/Attention 主推理（单次） | NPU | <= 40ms | 1000 次 rknn_run | R1 |
| IMF NPU 推理 K 次合计 | NPU | <= 600ms | K_pv + K_load 次推理 | R1 |
| 分位数后处理 | CPU | <= 10ms | 现有路径 | R1 |
| **BiLSTM 主推理（单次，Go 路径）** | **NPU** | **<= 80ms** | **1000 次 rknn_run** | **R2** |
| **误差修正推理（PV，单次）** | **NPU** | **<= 100ms** | **1000 次 rknn_run** | **R2** |
| **误差修正推理（Load，单次）** | **NPU** | **<= 100ms** | **1000 次 rknn_run** | **R2** |
| **误差修正合计** | **NPU** | **<= 200ms** | **串行 PV + Load** | **R2** |
| **残差缓冲更新** | **CPU** | **<= 1ms** | **FIFO push** | **R2** |

#### 14.6.2 组合预算表

**Go 路径（BiLSTM 通过硬件验证 + 误差修正启用，全功能）：**

| 阶段 | 预算 | 累计 |
|------|------|------|
| VMD 分解（PV + Load） | <= 50ms | 50ms |
| IMF BiLSTM 推理（K_pv + K_load 次, NPU） | <= 600ms | 650ms |
| 分位数后处理 | <= 10ms | 660ms |
| 误差修正推理（PV + Load） | <= 200ms | 860ms |
| **端到端总延迟（Go 路径，P99）** | **<= 860ms** | **满足 < 1s** |
| Go 路径终端余量 | **140ms** | -- |

**Level 1A（Go 路径无误差修正 = BiLSTM + VMD，error_correction.enabled=false 直接进入）：**

| 阶段 | 预算 | 累计 |
|------|------|------|
| VMD 分解 | <= 50ms | 50ms |
| IMF BiLSTM 推理 | <= 600ms | 650ms |
| 分位数后处理 | <= 10ms | 660ms |
| **端到端总延迟（Level 1A，P99）** | **<= 660ms** | **满足 < 1s** |

**No-Go 路径 A（BiLSTM No-Go + 误差修正启用）：**

| 阶段 | 预算 | 累计 |
|------|------|------|
| VMD 分解（PV + Load） | <= 50ms | 50ms |
| IMF LSTM/Attention 推理（K_pv + K_load 次, NPU） | <= 490ms | 540ms |
| 分位数后处理 | <= 10ms | 550ms |
| 误差修正推理（PV + Load） | <= 200ms | 750ms |
| **端到端总延迟（No-Go A，P99）** | **<= 750ms** | **满足 < 1s** |

**No-Go 路径 B（BiLSTM No-Go + 误差修正禁用 = 第一轮 VMD + Attention）：**

| 阶段 | 预算 | 累计 |
|------|------|------|
| VMD 分解（PV + Load） | <= 50ms | 50ms |
| IMF LSTM/Attention 推理（K_pv + K_load 次, NPU） | <= 490ms | 540ms |
| 分位数后处理 | <= 10ms | 550ms |
| **端到端总延迟（No-Go B，P99）** | **<= 550ms** | **满足 < 1s** |

**Level 3（AttentionOnly，无 VMD/EC）：**

| 阶段 | 预算 | 累计 |
|------|------|------|
| LSTM/Attention 推理（单次 PV + Load） | <= 80ms | 80ms |
| 分位数后处理 | <= 10ms | 90ms |
| **端到端总延迟（Level 3，P99）** | **<= 90ms** | **满足 < 1s** |

**基线（全降级，v2.16 基线）：**

| 阶段 | 预算 | 累计 |
|------|------|------|
| LSTM 基线推理（单次 PV + Load） | <= 80ms | 80ms |
| 分位数后处理 | <= 10ms | 90ms |
| **端到端总延迟（基线，P99）** | **<= 90ms** | **满足 < 1s** |

#### 14.6.3 BiLSTM 准入条件的延迟裕度分析

以 Go 路径全功能总延迟 860ms 为基准：

| 边界条件 | 值 | 说明 |
|----------|-----|------|
| 全管线延迟硬上限 | < 1s (1000ms) | 系统级约束 |
| Go 路径预算 | <= 860ms | 剩余 140ms 裕度 |
| BiLSTM 准入门限 | P99 < 900ms | PRD Section 4.2 定义，为全管线留 100ms 裕度 |
| BiLSTM 单次推理上限 | <= 80ms | 与单向 LSTM (<= 40ms) 之比 = 2.0x，满足 <= 2.2x 参数量约束 |

**分析结论：** Go 路径 P99 <= 860ms，距离 900ms 准入门限有 40ms 裕度，距离 1s 硬上限有 140ms 裕度。裕度来源于：IMF 推理未并行批处理（K 次串行），实际延迟可能低于 600ms（若支持 batch>1 可降至约 40ms）；误差修正的 200ms 预算中，实际单次修正推理预期 <= 80ms；VMD 的 50ms 预算中，可启用 `rayon` 并行降低至 ~25ms。

#### 14.6.4 模型大小预算

| 模型文件 | 大小 | 轮次 |
|----------|------|------|
| `lstm_attn.rknn`（LSTM + Attention, INT8） | <= 8MB（基线 ~5MB + Attention 增 ~3MB） | R1 |
| `bilstm_attn.rknn`（BiLSTM + Attention, INT8） | <= 12MB（参数量 <= 2.2x lstm_attn） | R2 |
| `error_correction.rknn`（误差修正 BiLSTM, INT8） | <= 3MB（参数量 <= 主预测的 50%） | R2 |
| **三个模型文件合计** | **<= 23MB** | -- |

#### 14.6.5 内存预算

| 组件 | 内存 | 轮次 |
|------|------|------|
| VMD 分解器工作内存（nalgebra 矩阵） | <= 5MB | R1 |
| VMD IMF 缓冲（K * input_window * f32） | <= 2MB | R1 |
| RknnRuntime #1（主预测模型） | <= 200MB | R1 |
| RknnRuntime #2（误差修正模型） | <= 80MB（独立 NPU 内存段） | R2 |
| 残差缓冲（2 * T * f32 ≈ 192B） | < 1KB | R2 |
| **推理运行时总内存（含增强）** | **<= 350MB**（基线 200MB + R1 50MB + R2 100MB） | -- |

### 14.7 配置设计

#### 14.7.1 YAML 配置结构（R2 扩展）

在 `mupc/config/mupc_env_config.yaml` 中新增 `prediction_enhancement` 段：

```yaml
# ============================================================================
# 预测增强配置（v3.0，2026-06-21）
# 缺失时系统运行于 v2.16 基线模式（全部增强功能禁用）
# ============================================================================
prediction_enhancement:
  # ==========================================================================
  # 第一轮配置项（VMD + Attention）— R1 实现
  # ==========================================================================
  vmd:
    enabled: true               # 启用 VMD 分解
    k_pv: 5                     # 光伏模态数 [2, 10]
    k_load: 6                   # 负荷模态数 [2, 10]
    alpha: 2000.0               # 惩罚因子 [100, 5000]
    tol: 1.0e-6                 # 收敛容差 [1e-7, 1e-5]
    max_iter: 500               # 最大迭代次数 [100, 2000]
    tau: 0.0                    # 噪声容忍度（Lagrangian 更新步长，0.0 = 标准 VMD）

  attention:
    enabled: true               # Attention 启用（需 ONNX 模型含 Attention 层）
    score_type: "additive"      # additive | dot | general
    export_weights: false       # 是否导出注意力权重到日志

  # ==========================================================================
  # 第二轮配置项（BiLSTM + 误差修正）— R2 实现
  # ==========================================================================
  bilstm:
    enabled: false              # [R2] 是否启用 BiLSTM 双向替换（需硬件验证通过）
    gate_passed: false          # [R2] Go/No-Go 准入标志：RK3588 硬件延迟摸底通过后设为 true
    model_path: "/etc/mupc/models/bilstm_attn.rknn"  # [R2] BiLSTM 模型文件路径
    hidden_size_override: null  # [R2] 隐状态维度覆盖（null = 使用模型内建值，仅调试用途）
    fallback_on_failure: true   # [R2] BiLSTM 推理失败时是否自动回退单向 LSTM（默认 true）

  error_correction:
    enabled: false              # [R2] 是否启用误差修正 BiLSTM（主预测偏差 > 3% 时启用）
    model_path: "/etc/mupc/models/error_correction.rknn"  # [R2] 误差修正模型文件路径
    residual_window_steps: 24   # [R2] 残差窗口步数（默认 24，与主预测 input_window 对齐）
    zero_init: true             # [R2] 冷启动/缓冲未满时是否零向量填充（true = 零填充跳过修正，false = 拒绝推理）
    auto_disable_after_failures: 3  # [R2] 连续失败 N 次后自动禁用误差修正（0 = 不自动禁用）
    enable_bias_check: true     # [R2] 是否启用系统性偏差检测（主预测 |Bias| > 3% MAPE 才启用修正）

  # ==========================================================================
  # 特征筛选配置（离线 MIC，R1 定义 + 跨轮次不变）
  # ==========================================================================
  feature_selection:
    mic_top_k: 7                # MIC 筛选 Top-K 特征数
```

#### 14.7.2 配置项归属（第一轮 vs 第二轮）

| 配置段 | 配置项 | 轮次 | 说明 |
|--------|--------|------|------|
| `vmd` | 全部 | R1 | VMD 分解器参数 |
| `attention` | 全部 | R1 | Attention 调试开关 |
| `bilstm.enabled` | R2 | R2 | BiLSTM 启用开关 |
| `bilstm.gate_passed` | R2 | R2 | 硬件验证准入标志 |
| `bilstm.model_path` | R2 | R2 | 模型文件路径 |
| `bilstm.hidden_size_override` | R2 | R2 | 调试覆盖（生产不应使用） |
| `bilstm.fallback_on_failure` | R2 | R2 | 失败回退策略 |
| `error_correction.enabled` | R2 | R2 | 误差修正启用开关 |
| `error_correction.model_path` | R2 | R2 | 修正模型文件路径 |
| `error_correction.residual_window_steps` | R2 | R2 | 残差窗口长度 |
| `error_correction.zero_init` | R2 | R2 | 冷启动策略 |
| `error_correction.auto_disable_after_failures` | R2 | R2 | 自动禁用阈值 |
| `error_correction.enable_bias_check` | R2 | R2 | 偏差检测开关 |
| `feature_selection` | 全部 | R1 | 离线 MIC 引用 |

#### 14.7.3 配置默认值策略

| 场景 | 行为 |
|------|------|
| `prediction_enhancement` 段完全缺失 | 所有增强功能禁用，运行 v2.16 基线模式 |
| `vmd` 子段缺失 | `vmd.enabled = false` |
| `attention` 子段缺失 | `attention.enabled = false` |
| `bilstm` 子段缺失 | 所有 bilstm 字段使用默认值（`enabled = false`, `gate_passed = false`） |
| `error_correction` 子段缺失 | 所有 error_correction 字段使用默认值（`enabled = false`, `zero_init = true`） |
| 参数值超出范围（如 `k_pv = 0`） | 使用硬编码默认值（k_pv=5, k_load=6）并记录 WARN |
| `residual_window_steps < 1` | 使用默认值 24，记录 WARN |
| `auto_disable_after_failures < 0` | 使用默认值 3，记录 WARN |

#### 14.7.4 配置热加载

配置变更通过 `DynamicConfigLoader` 周期轮询（与现有一致），增强模块支持运行时重新加载：

- `vmd.enabled` 由 true -> false：下一次推理周期起跳过 VMD
- `vmd.enabled` 由 false -> true：下一次推理周期起尝试 VMD（VmdDecomposer 重新初始化）
- `attention.enabled` 切换：仅当模型含 Attention 层时生效，否则记录 WARN
- `bilstm.enabled` 切换：需重新加载模型文件（OTA 场景），运行时切换记录 INFO 但推迟到下次模型加载；`gate_passed` 变更需重启生效（防止运行时突然切换模型导致推理中断）
- `error_correction.enabled` / `residual_window_steps` / `zero_init` 切换：下一次推理周期起生效（`residual_window_steps` 变更需重建 ResidualBuffer）
- `error_correction.model_path` 变更：需重新加载模型文件（OTA 场景），下一次推理周期起生效

### 14.8 错误处理与降级

#### 14.8.1 降级层级（R2 扩展）

```
Level 0: VMD → BiLSTM/Attention → 误差修正         [全功能, Go 路径]
Level 1A: VMD → BiLSTM/Attention (无误差修正)       [误差修正降级，或 error_correction.enabled=false 直接进入]
Level 1B: VMD → LSTM/Attention → 误差修正            [BiLSTM No-Go / BiLSTM 降级]
Level 2: VMD → LSTM/Attention (无误差修正)           [BiLSTM 降级 + 误差修正降级]
Level 3: LSTM/Attention (无 VMD, 无误差修正)         [VMD 降级]
Level 4: LSTM 基线 (无 VMD, 无 Attention)            [Attention 降级 = v2.16 基线]
Level 5: 全零预测 (安全兜底)                          [模型加载完全失败，由 ModelManager 处理]
```

**降级层级边界说明：**

| EnhancementLevel 枚举值 | 对应降级层级 | 管理者 | 说明 |
|-------------------------|-------------|--------|------|
| `FullVmdAttentionCorrection` (0) | Level 0 | PredictionPipeline | R2 实现真正的误差修正，不再直接降级 |
| `BiLstmVmdAttention` (1) | Level 1A | PredictionPipeline | Go 路径下误差修正失败降到此层，或 error_correction.enabled=false 直接进入 |
| `VmdAttention` (2) | Level 1B / 2 | PredictionPipeline | No-Go 路径下误差修正启用时从 Level 0 降到此层 |
| `AttentionOnly` (3) | Level 3 | PredictionPipeline | VMD 失败 |
| `Baseline` (4) | Level 4 | PredictionPipeline | Attention 失败/未配置 |
| (无枚举值) | Level 5 | **ModelManager** | `run_lstm_predict_with_quantiles()` 也失败时，返回全零向量 |

**注意：** Level 1A 和 Level 1B 都是 `EnhancementLevel` 枚举中的一个值，但代表不同的降级路径：
- 1A = BiLSTM (Go) + VMD，误差修正失败
- 1B = LSTM + VMD + 误差修正，BiLSTM No-Go 或 BiLSTM 推理失败

这两者在当前枚举设计中共用 `VmdAttention` 值（`EnhancementLevel::VmdAttention`），通过 `BiLstmConfig.enabled` 和 `ErrorCorrectionConfig.enabled` 的组合在日志中区分降级原因，不创建两个仅名称不同的枚举值（保持枚举精简），但降级日志中明确标注降级原因：

```
[WARN] 降级至 VmdAttention (Level 2): BiLSTM 推理失败，已回退单向 LSTM
[WARN] 降级至 VmdAttention (Level 2): 误差修正连续 3 次失败，已禁用
```

**BiLSTM 降级层级过渡（R2 新增）：**

```
BiLSTM 推理失败（单次）或连续 3 次 P99 > 80ms
  |
  +--> 自动回退到单向 LSTM + Attention（加载 lstm_attn.rknn）
  +--> 保留 VMD + 误差修正功能（若启用）
  +--> 降级原因记录到日志: "BiLSTM -> 单向LSTM"
  +--> 恢复: 运维修复后手动设 gate_passed=true 并重启（或 OTA 下发新版 bilstm_attn.rknn）
```

#### 14.8.2 自动升降级逻辑（R2 扩展）

```rust
// prediction_pipeline.rs 核心逻辑 (R2 扩展)

impl PredictionPipeline {
    pub async fn execute(&self) -> Result<EnhancedForecastResult, AiEngineError> {
        let mut level = self.health.read().await.current_level;

        loop {
            match level {
                EnhancementLevel::FullVmdAttentionCorrection => {
                    // R2: 执行真正的 VMD + (Bi)LSTM/Attention + 误差修正
                    match self.execute_full_with_correction().await {
                        Ok(r) => {
                            self.health.write().await.on_success_both();
                            return Ok(r);
                        }
                        Err(e) => {
                            // 区分错误来源：主预测失败 vs 误差修正失败
                            if e.is_error_correction_failure() {
                                tracing::warn!("误差修正失败: {}, 降级至 BiLSTM+VMD", e);
                                self.health.write().await.on_failure_ec();
                                level = EnhancementLevel::BiLstmVmdAttention;
                            } else {
                                tracing::warn!("主预测失败: {}, 降级至 VMD+Attention", e);
                                self.health.write().await.on_failure_vmd();
                                level = EnhancementLevel::VmdAttention;
                            }
                        }
                    }
                }
                EnhancementLevel::BiLstmVmdAttention => {
                    // R2: BiLSTM + VMD（跳过误差修正）
                    match self.execute_vmd_attention().await {
                        Ok(mut r) => {
                            r.error_correction_applied = false;
                            return Ok(r);
                        }
                        Err(e) => { /* 继续降级 */ }
                    }
                }
                EnhancementLevel::VmdAttention => {
                    match self.execute_vmd_attention().await {
                        Ok(r) => {
                            self.try_promote().await;
                            return Ok(r);
                        }
                        Err(e) => {
                            tracing::warn!("VMD+Attention 失败: {}, 降级至 Attention", e);
                            self.health.write().await.on_failure_vmd();
                            level = EnhancementLevel::AttentionOnly;
                        }
                    }
                }
                EnhancementLevel::AttentionOnly => { /* 同 v1.1 */ }
                EnhancementLevel::Baseline => { /* 同 v1.1 */ }
            }
        }
    }

    /// R2 扩展：连续成功升级逻辑支持多模块
    ///
    /// VMD 连续 5 次成功 → 可升回 VMD 层级
    /// 误差修正连续 5 次成功 → 可升回误差修正层级
    /// 每个模块独立追踪，不互相阻塞
    async fn try_promote(&self) {
        let mut health = self.health.write().await;

        // VMD 升级（连续 5 次成功升一级）
        if health.vmd_consecutive_successes >= 5 {
            // Level 3 → 2, Level 2 → 1A/1B
        }

        // R2 新增：误差修正升级（连续 5 次成功升一级）
        if health.ec_consecutive_successes >= 5 {
            // Level 1A → 0（若 BiLSTM Go），Level 1B → 0（若 BiLSTM No-Go）
        }
    }
}
```

#### 14.8.3 错误变体

预测增强管线新增的错误变体已定义于 [11.1 AiEngineError 枚举](#111-aiengineerror-枚举)：

| 错误变体 | 用途 | 轮次 |
|----------|------|------|
| `VmdFailed(String)` | VMD 分解失败 | R1 |
| `VmdNotConverged { max_iter, final_error }` | VMD 迭代不收敛 | R1 |
| `AttentionDegraded` | Attention 层退化 | R1 |
| `ErrorCorrectionFailed(String)` | 误差修正失败 | R1 |
| `ModelValidationFailed { model_path, reason }` | 模型 metadata 校验失败 | R2 |
| `ResidualBufferInsufficient { filled, capacity }` | 残差缓冲不足 | R2 |

### 14.9 测试策略

#### 14.9.1 单元测试（R2 扩展）

| 模块 | 测试项 | 验证标准 | 轮次 |
|------|--------|----------|------|
| `vmd.rs` | VMD 分解对 24 步正弦波 + 噪声的合成信号，输出 K=4 个 IMF | VMD-01: IMF 长度 = 输入长度 | R1 |
| `vmd.rs` | 所有 IMF 求和与原始信号 RMSE <= 1e-4 | VMD-02: 重构保真度 | R1 |
| `vmd.rs` | max_iter=1 时返回 VmdNotConverged | VMD-06: 不收敛处理 | R1 |
| `vmd.rs` | 输入含 NaN 时返回 VmdFailed | VMD 异常处理 | R1 |
| `prediction_pipeline.rs` | VMD 失败时自动降级到 baseline | 降级逻辑 | R1 |
| `prediction_pipeline.rs` | 连续 5 次成功后自动升级 | 升级逻辑 | R1 |
| `config.rs` | EnhancementConfig 缺失时全部 default | 配置兼容性 | R1 |
| `residual_buffer.rs` | 缓冲已满时提取最近 T 步残差 | ERR-08: 历史残差输入 | R2 |
| `residual_buffer.rs` | 缓冲未满 + zero_init=true 时返回零向量 | ERR-08: 冷启动零填充 | R2 |
| `residual_buffer.rs` | 缓冲未满 + zero_init=false 时返回错误 | ERR-08: 保守模式 | R2 |
| `model_validator.rs` | metadata 校验通过 (mupc_model_type="bilstm" + config.enabled=true) | 模型加载校验 | R2 |
| `model_validator.rs` | metadata 不一致时返回 ModelValidationFailed | 模型加载校验 | R2 |
| `model_validator.rs` | gate_passed=false + enabled=true 时选择单向模型 | 14.2.5 逻辑 | R2 |
| `prediction_pipeline.rs` | 误差修正启用 + 缓冲已满 → error_correction_applied=true | ERR 集成测试 | R2 |
| `prediction_pipeline.rs` | 误差修正启用 + 缓冲未满 + zero_init=true → 跳过修正 | ERR 集成测试 | R2 |
| `prediction_pipeline.rs` | 误差修正连续 3 次失败 → 自动禁用 | ERR 降级逻辑 | R2 |
| `prediction_pipeline.rs` | BiLSTM 失败 → 回退单向 LSTM + 保留 VMD/误差修正 | BiLSTM 降级逻辑 | R2 |

#### 14.9.2 性能测试

```rust
// 单模块性能基准 (R2 扩展)

#[test]
fn test_error_correction_inference_latency() {
    // R2: 误差修正推理延迟 <= 100ms (单次 NPU)
    let runtime = RknnRuntime::new("error_correction.rknn")?;
    let input = vec![0.0_f32; 24]; // 零向量输入（冷启动场景）
    let start = std::time::Instant::now();
    let output = runtime.run(&input)?;
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() <= 100,
        "误差修正推理超时: {}ms", elapsed.as_millis());
}

#[test]
fn test_residual_buffer_update_latency() {
    // R2: 残差缓冲更新延迟 <= 1ms
    let mut buf = ResidualBuffer::new(24, true);
    let start = std::time::Instant::now();
    for _ in 0..100 {
        buf.push(0.5);
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() <= 1,
        "残差缓冲更新超时: {}ms", elapsed.as_millis());
}
```

#### 14.9.3 集成测试（R2 扩展）

| 测试场景 | 预期行为 | 轮次 |
|----------|----------|------|
| 启动时 `prediction_enhancement` 缺失 | 运行于 baseline，日志 INFO | R1 |
| VMD + Attention 全功能路径 | `enhancement_level = VmdAttention` | R1 |
| VMD 参数非法 (k_pv=0) | 使用默认值，WARN 日志 | R1 |
| 模型不含 Attention 但配置 `attention.enabled=true` | 自动降级，WARN 日志 | R1 |
| VMD 连续 3 次失败 | `PipelineHealth.current_level` 降级 | R1 |
| **BiLSTM gate_passed=true + 模型加载成功** | **enhancement_level = FullVmdAttentionCorrection (若误差修正也启用)** | **R2** |
| **BiLSTM gate_passed=false + enabled=true** | **加载 lstm_attn.rknn，记录 WARN "gate_passed=false"** | **R2** |
| **BiLSTM gate_passed=true + 推理失败** | **回退 lstm_attn.rknn，保留 VMD 和误差修正** | **R2** |
| **误差修正启用 + 残差缓冲已满** | **y_corrected = y_pred + e_pred** | **R2** |
| **误差修正启用 + 残差缓冲未满 + zero_init=true** | **y_corrected = y_pred，不抛错** | **R2** |
| **误差修正推理失败** | **跳过修正，主预测值直接输出，连续 3 次后禁用** | **R2** |
| **全功能 Go 路径端到端** | **VMD + BiLSTM/Attention + 误差修正，延迟 < 1s** | **R2** |

### 14.10 风险与缓解

| 风险 | 概率 | 影响 | 缓解 | 轮次 |
|------|------|------|------|------|
| VMD Rust 实现数值不稳定 | 中 | 预测精度不达标 | 与 Python VMD (vmdpy) 输出对比验证；若不达标切换 FFI | R1 |
| VMD 延迟 > 50ms | 低 | 总延迟超标 | `rayon` 并行 K 个 IMF；减少 K 值；切换 FFI | R1 |
| RKNN Toolkit 不支持 Attention ONNX 算子 | 低 | Attention 无法 NPU 加速 | MatMul/Softmax/ReduceSum 均为基础算子，RKNN Toolkit 2 已支持 | R1 |
| IMF 逐个推理导致 NPU 调用次数膨胀 | 中 | 延迟超标 | 若 RKNN 支持 batch>1，拼接 K 个 IMF 为 batch 一次推理 | R1 |
| VMD 模块引入 unsafe 代码 | 极低 | 安全审计不通过 | `rustfft` 和 `nalgebra` 均为 pure Rust；若引入 FFI 则走 review | R1 |
| **BiLSTM 参数量超标（> 2.2x 单向）** | **低** | **模型无法加载** | **metadata 校验阶段拦截；训练管线在导出前验证参数量** | **R2** |
| **BiLSTM 推理延迟超标（P99 > 900ms）** | **中** | **BiLSTM No-Go** | **PRD Section 4.2 准入条件：延迟摸底 P99 >= 900ms → 跳过 BiLSTM，仅保留误差修正** | **R2** |
| **误差修正推理与主预测 NPU 资源争抢** | **低** | **总延迟超标** | **两个 RknnRuntime 串行调用；NPU 不支持并行推理，设计上已保证串行** | **R2** |
| **残差缓冲与实际值不同步** | **中** | **误差修正方向错误** | **每个预测周期结束后，等待本周期实际值到达后再更新缓冲；配置 `enable_bias_check` 持续监控修正方向** | **R2** |
| **误差修正引入负修正（修正后比修正前更差）** | **中** | **预测精度反降** | **`enable_bias_check` 检测修正后 MAPE 是否劣于修正前；连续 3 次劣化自动禁用** | **R2** |
| **双 RknnRuntime 实例内存超标** | **低** | **OOM** | **误差修正为轻量 BiLSTM（<= 主预测 50% 参数量），内存增量 <= 80MB（14.6.5）** | **R2** |

### 14.11 与现有文档的关系

| 文档 | 关系 |
|------|------|
| `2026-06-21-预测增强分层混合架构-PRD.md` (v1.1) | 本设计文档的输入需求 |
| `modules/05-MUPC-AI引擎-PRD.md` | AI 引擎基线 PRD，本设计在其上增强 |
| `论文吸收-预测增强.md` | 方法论背景 |
| `technical-debt.md` | 增强完成后需更新 Phase 3C 增强状态 |
| `pipeline_config.rs` | 本设计 14.5.3 的代码实现（R1 已实现，R2 将扩展） |
| `prediction_pipeline.rs` | 本设计 14.5.2 的代码实现（R1 已实现核心编排，R2 将扩展误差修正） |
| `model_manager.rs` | 本设计 14.4.4 集成点（R1 已集成，R2 将扩展多 Runtime 管理） |

### 14.12 跨项目接口（与 MUPC-AI2 训练管线）

完全遵循 PRD Section 7.4 定义的 JSON Schema：

| 接口 | 方向 | 格式 | Schema 位置 |
|------|------|------|-------------|
| MIC 分析结果 | MUPC-AI2 -> MUPC | JSON | PRD Section 7.4.1 |
| MSSA 搜索最优超参 | MUPC-AI2 -> MUPC-AI2 | JSON | PRD Section 7.4.2 |
| ONNX 模型（含 Attention） | MUPC-AI2 -> .rknn 转换 -> MUPC | .rknn | PRD Section 7.4.3 |
| ONNX 模型元数据 | 内嵌于 ONNX | metadata_props | PRD Section 7.4.3 表格 |

**Rust 侧职责：**
- 读取 ONNX 模型导出时写入的 `metadata_props`（通过 RKNN Toolkit 转后在运行时查询）
- 校验 `mupc_model_type`、`mupc_with_attention`、`mupc_with_vmd`、`mupc_direction` 与 `PredictionEnhancementConfig` 一致性
- 含 VMD 模型时校验 `mupc_with_vmd == "true"`，并在 K 维求和重构
- R2 新增：校验 `mupc_hidden_size` 以间接验证 BiLSTM 参数量约束（<= 2.2x 单向）

---

## 15. MSSA 超参优化工具设计

> **v3.0 合并新增** | **来源：** `docs/superpowers/plans/2026-06-21-预测增强分层混合架构-DESIGN.md` v3.0, Section 12
> **PRD 对应：** Section 3.5 (F5: 超参自动优化) + Section 7.4.2 (JSON Schema)
> **与 Rust 的关系：** 零。搜索结果通过 JSON 被训练管线消费，最终体现为 ONNX 模型维度变化。Rust 推理端不感知搜索过程。

### 15.1 模块定位

| 维度 | 说明 |
|------|------|
| **模块名称** | `tools/mssa_optimizer/` |
| **语言** | Python 3.9+，纯 Python（无 C 扩展） |
| **依赖** | numpy >= 1.24、scipy >= 1.10、pyyaml >= 6.0（详见 Python 依赖表） |
| **运行环境** | MUPC-AI2 训练服务器（Linux x86_64），不在 RK3588 上运行 |
| **触发方式** | 命令行 `python -m tools.mssa_optimizer --config mssa_search_config.yaml` |
| **对 Rust 的影响** | 零。搜索结果通过 JSON 被训练管线消费，最终体现为 ONNX 模型维度变化。Rust 推理端不感知搜索过程 |
| **PRD 对应** | Section 3.5 (F5: 超参自动优化) + Section 7.4.2 (JSON Schema) |

**设计原则：**
- **无状态**：每次运行独立，不依赖 DB 或外部服务（缓存文件除外）
- **幂等**：相同搜索配置 + 相同训练数据 → 确定性结果（固定 random seed）
- **可中断**：支持 Ctrl+C 优雅退出，输出当前最优解（通过 signal handler 捕获 SIGINT）
- **可观测**：每次迭代输出 `[INFO]` 级别日志（当前迭代数、最优 MAPE、种群多样性指标）

**备选（IPSO 降级路径）：** IPSO（Improved Particle Swarm Optimization，改进粒子群优化）作为备选方案。当 MSSA 搜索时间超预期（单次搜索 > 2 小时）或收敛不稳定（连续 3 次重启后最优解 MAPE > 人工基线 MAPE * 1.1）时，通过配置文件 `algorithm: "IPSO"` 切换。IPSO 实现简单、收敛快（通常 <= 30 次迭代），但全局搜索能力弱于 MSSA。切换时输出 JSON 中 `search_metadata.algorithm` 字段自动变更为 `"IPSO"`，其余 JSON Schema 保持一致。IPSO 降级时 algorithm 字段值为 `"IPSO"`（与 PRD v1.1 Schema 的 const 约束存在已知偏差，将在 PRD 下一版本修正为 enum）。

**Python 依赖说明：**

| 依赖 | 版本 | 用途 |
|------|------|------|
| `numpy` | >= 1.24 | 种群矩阵运算（向量化操作）、伪随机数生成（PCG64 或 MT19937）、数值稳定性保护 |
| `scipy` | >= 1.10 | `scipy.stats` 统计检验（Mann-Whitney U）、`scipy.spatial.distance` 种群多样性度量、`scipy.optimize` 辅助（仅调试用） |
| `pyyaml` | >= 6.0 | 搜索配置文件解析 |
| `json` | 标准库 | JSON 输出（对齐 PRD Section 7.4.2 JSON Schema） |
| `hashlib` | 标准库 | 超参组合 SHA256 指纹（缓存键生成） |
| `subprocess` | 标准库 | 调用训练脚本 |
| `time` | 标准库 | 超时控制 |
| `pathlib` | 标准库 | 临时配置路径管理、缓存文件路径 |
| `tempfile` | 标准库 | 临时训练配置文件创建（原子写入 + 自动清理） |

**与 MUPC-AI2 训练管线的集成点：**

```
tools/mssa_optimizer/                         MUPC-AI2 训练管线
+----------------------------------+         +-----------------------------------+
| mssa.py         (算法核心)        |         |                                   |
| search_space.py (搜索空间定义)     |  JSON   |  读取 best_hyperparameters       |
| objective.py    (目标函数) ──subprocess──→ |  train.py --config <tmp_config>  |
|                                   |  (调用)  |  export_onnx.py                  |
| output.py       (JSON 输出)       |         |                                   |
| config.py       (配置加载/校验)    |         |                                   |
| test_mssa.py    (单元测试)        |         |                                   |
+----------------------------------+         +-----------------------------------+

数据流：
1. mssa.py 生成一组超参 → objective.py
2. objective.py 写临时训练配置 YAML → 调用 subprocess 运行 train.py
3. train.py 训练 → 输出验证集 MAPE → objective.py 解析 stdout/结果文件
4. objective.py 返回加权目标函数值 → mssa.py 用于种群更新
5. 搜索结束 → output.py 写 mssa_search_result.json → MUPC-AI2 训练管线消费
```

### 15.2 文件结构

```
tools/mssa_optimizer/
├── __init__.py                # 包初始化，导出公共 API (~5 行)
├── mssa.py                   # MSSA 算法核心 (预计 ~300 行)
│   ├── class MssaOptimizer   #   主控制器：种群初始化、迭代循环、终止判定
│   ├── class Population      #   种群管理：位置矩阵 (N, D)、适应度向量 (N,)
│   ├── class Discoverer      #   发现者更新逻辑
│   ├── class Joiner          #   加入者更新逻辑
│   ├── class Scout           #   侦察者更新逻辑
│   ├── good_point_set_init() #   佳点集初始化函数
│   ├── opposition_learning() #   反向学习增强函数
│   └── corsi_mutation()      #   Corsi 变异扰动函数
├── search_space.py            # 搜索空间定义与编码/解码 (~150 行)
│   ├── class SearchSpace      #   10 维超参搜索空间定义
│   ├── class HyperParam      #   单个超参定义（名称/类型/范围/步长）
│   ├── encode() / decode()    #   混合编码/解码（离散→整数索引，连续→浮点，枚举→one-hot）
│   └── random_sample()        #   搜索空间内均匀随机采样
├── objective.py               # 目标函数 (~200 行)
│   ├── class ObjectiveFunc    #   目标函数包装
│   ├── class TrainingRunner  #   子进程调用封装（train.py）
│   ├── class CacheManager    #   评估缓存管理（mssa_cache.json）
│   └── class ResultParser    #   训练输出解析（提取 MAPE_pv、MAPE_load）
├── config.py                  # 配置加载/校验 (~100 行)
│   ├── class MssaConfig       #   搜索配置结构体（种群参数、终止条件、算法选择）
│   ├── load_config()          #   YAML 加载 + schema 校验
│   └── validate()             #   参数合法性校验（范围检查、互斥检查）
├── output.py                  # JSON 输出 (~80 行)
│   ├── class SearchOutput     #   搜索输出构建器
│   ├── to_json()              #   序列化为 PRD 7.4.2 兼容 JSON
│   └── validate_output()      #   JSON Schema 自校验
├── mssa_cache.json            # 评估缓存（运行时自动生成，.gitignore 排除）
└── test_mssa.py               # 单元测试 (~200 行)
    ├── test_good_point_set()  #   佳点集初始化均匀性测试
    ├── test_encode_decode()   #   编解码往返一致性测试
    ├── test_objective_cache() #   缓存命中/未命中测试
    ├── test_convergence()     #   收敛条件触发测试（模拟目标函数）
    ├── test_output_schema()   #   输出 JSON 对 PRD 7.4.2 schema 合规性
    └── test_ipso_fallback()   #   IPSO 降级路径测试
```

**文件总行数预估：**

| 文件 | 预估行数 | 说明 |
|------|----------|------|
| `__init__.py` | ~5 | 包导出 |
| `mssa.py` | ~300 | 算法核心（种群 + 三群体 + 增强策略） |
| `search_space.py` | ~150 | 10 维搜索空间 + 混合编解码 |
| `objective.py` | ~200 | 目标函数 + 训练子进程 + 缓存 |
| `config.py` | ~100 | YAML 配置加载 + 校验 |
| `output.py` | ~80 | JSON 输出构建 + schema 自校验 |
| `test_mssa.py` | ~200 | 6 项单元测试 |
| `mssa_search_config.yaml` | ~60 | 配置模板（含注释） |
| **合计** | **~1095** | |

### 15.3 MSSA 算法设计

#### 15.3.1 算法总览

MSSA 模拟麻雀群体觅食行为，将超参搜索问题映射为群体在 D 维搜索空间中的位置优化：

```
算法流程（伪代码）：

Input:  搜索空间 Ω (D=10 维), 种群大小 N (默认 30),
        发现者比例 p_d=0.2, 侦察者比例 p_s=0.1,
        最大迭代 T_max (默认 50), 收敛阈值 ε=1e-4,
        停滞容忍 S_max (默认 10), 总时间上限 T_wall (默认 7200s)

Output: 最优超参 x_best, 最优目标值 f_best, 收敛曲线, 搜索轨迹

1. 初始化:
   1a. 佳点集生成 N 个初始个体位置 X = {x_1, ..., x_N}  (15.3.3)
   1b. 评估所有个体 f_i = ObjectiveFunc(x_i)
   1c. 排序: 按 f_i 升序排列 (最小化问题)
   1d. 记录全局最优: x_best = x_1, f_best = f_1
   1e. 初始化停滞计数: stagnation = 0, prev_best = f_best
   1f. 初始化缓存: CacheManager.load()

2. 主循环 (for iter = 1 to T_max):
   2a. 群体角色分配:
       - 前 p_d*N 个体 = 发现者 (Discoverer)
       - 中间 (1-p_d-p_s)*N 个体 = 加入者 (Joiner)
       - 后 p_s*N 个体 = 侦察者 (Scout)

   2b. 发现者更新 (15.3.4)
   2c. 加入者更新 (15.3.5)
   2d. 侦察者更新 (15.3.6)
   2e. 边界处理: 裁剪到搜索空间 Ω
   2f. 评估所有新个体 f_i_new = ObjectiveFunc(x_i_new)  (带缓存)
   2g. 精英保留: 若 min(f_new) > f_best, 保留上一代最优个体
   2h. 更新全局最优

   2i. 反向学习增强 (15.3.7): 每 5 次迭代执行一次
   2j. Corsi 变异扰动 (15.3.8): 若 stagnation >= S_max, 对停滞个体施加强变异

   2k. 收敛判定 (15.6):
       - 若 abs(f_best - prev_best) < ε: stagnation += 1
       - 否则: stagnation = 0
       - 若 stagnation >= S_max: 终止 (收敛)
       - 若 elapsed > T_wall: 终止 (超时)

   2l. 保存收敛曲线: convergence_curve[iter] = f_best
       保存参数轨迹: trajectory[param][iter] = x_best[param]

   2m. 检查 Ctrl+C 中断信号 → 优雅退出

3. 输出:
   3a. 构建 SearchOutput (符合 PRD 7.4.2 JSON Schema)
   3b. 写 mssa_search_result.json
   3c. 持久化缓存 (mssa_cache.json)
```

#### 15.3.2 种群规模与角色分配

| 参数 | 符号 | 默认值 | 可配置范围 | 说明 |
|------|------|--------|------------|------|
| 种群大小 | N | 30 | [10, 100] | 种群个体总数。N 越大搜索越充分，但目标函数评估次数 = N * T，训练成本高。若搜索空间较大或训练成本允许，可适当增大 N 至 50~60 以提升全局搜索能力 |
| 发现者比例 | p_d | 0.20 | [0.10, 0.30] | 全局勘探主力，数量少以保证快速收敛 |
| 侦察者比例 | p_s | 0.10 | [0.05, 0.20] | 预警个体，感知危险后引导群体逃离局部最优 |
| 加入者比例 | p_j = 1-p_d-p_s | 0.70 | -- | 跟随发现者在局部精细搜索 |

**角色分配规则：** 每代按适应度排序后：
- 前 `N_d = ceil(p_d * N)` 个体 = 发现者（适应度最优，负责全局勘探）
- 后 `N_s = ceil(p_s * N)` 个体 = 侦察者（适应度最差，负责预警和全局逃离）
- 中间 `N_j = N - N_d - N_s` 个体 = 加入者（中等适应度，负责局部开发）

默认配置：N=30, p_d=0.20, p_s=0.10 → N_d=6, N_s=3, N_j=21。

#### 15.3.3 佳点集初始化

传统随机初始化可能导致个体在搜索空间中聚集，降低种群多样性。佳点集（Good Point Set）方法生成均匀分布的初始种群：

```
x_{i,j} = (i * golden_ratio^j) mod 1 * (ub_j - lb_j) + lb_j

其中:
  i ∈ [1, N]:      个体索引
  j ∈ [1, D]:      维度索引 (D=10)
  golden_ratio:    黄金分割比 φ = (sqrt(5) - 1) / 2 ≈ 0.6180339887
  ub_j, lb_j:      第 j 维搜索上下界
```

**为什么用黄金分割比：** φ 是无理数，{i * φ^j mod 1} 序列在 [0,1] 上具有低偏差（low discrepancy），比伪随机序列更均匀。

**验证标准：** 单元测试验证佳点集生成的 N 个个体在 10 维空间中的成对最小欧氏距离 >= 随机初始化的 1.2 倍（种群多样性指标）。

#### 15.3.4 发现者位置更新

发现者负责在全局范围内搜索更优区域。位置更新公式：

```
x_{i,j}^{t+1} = x_{i,j}^t * exp(-i / (alpha * T_max))   若 R_2 < ST  (安全：广泛探索)
x_{i,j}^{t+1} = x_{i,j}^t + Q * L                        若 R_2 >= ST (危险：飞离当前位置)

其中:
  t:          当前迭代
  alpha:      (0, 1] 均匀随机数
  R_2:        [0, 1] 预警值 (每代随机生成)
  ST:         安全阈值 (默认 0.8)
  Q:          标准正态分布随机数 N(0, 1)
  L:          1×D 全 1 向量
```

#### 15.3.5 加入者位置更新

加入者跟随发现者，在发现者占领的中优区域进行局部搜索：

```
x_{i,j}^{t+1} = Q * exp((x_worst_j^t - x_{i,j}^t) / i^2)    若 i > N/2  (低排名加入者：飞向最优发现者)
x_{i,j}^{t+1} = x_best_j^t + |x_{i,j}^t - x_best_j^t| * A^+ * L   若 i <= N/2 (高排名加入者：在最优附近局部搜索)

其中:
  x_worst_j^t:   第 t 代全局最差个体的第 j 维
  x_best_j^t:    第 t 代全局最优个体的第 j 维
  A^+:           随机 1×D 向量，每个元素 = 1 或 -1 各 50% 概率
  A^+ = A^T * (A * A^T)^{-1}
```

#### 15.3.6 侦察者位置更新

侦察者感知危险，引导种群逃离局部最优：

```
x_{i,j}^{t+1} = x_best_j^t + beta * |x_{i,j}^t - x_best_j^t|    若 f_i > f_best  (当前个体比最优差)
x_{i,j}^{t+1} = x_{i,j}^t + K * (|x_{i,j}^t - x_worst_j^t| / (|f_i - f_worst| + 1e-8))    若 f_i == f_best  (当前个体在最优位置)

其中:
  beta:   标准正态分布随机数 N(0, 1)，控制步长
  K:      [-1, 1] 均匀随机数，控制方向
  f_i:    第 i 个体的适应度值
```

#### 15.3.7 反向学习增强

每 5 次迭代对全体个体执行一次反向学习（Opposition-Based Learning），生成每个个体的"反向解"并保留更优者：

```
x_opposition_{i,j} = lb_j + ub_j - x_{i,j}

若 f(x_opposition_i) < f(x_i):
    替换 x_i = x_opposition_i
否则:
    保留 x_i
```

#### 15.3.8 Corsi 变异扰动

当连续 S_max 次迭代（默认 10 次）全局最优无改善时，Corsi 变异针对收敛停滞的个体施加自适应强度变异：

```
停滞检测:
  若 stagnation_count >= S_max (默认 10):
    对适应度最差的后 50% 个体施加 Corsi 变异

Corsi 变异公式:
  x_{i,j}^{new} = x_{i,j} + Corsi * (ub_j - lb_j) * randn()

  Corsi 系数:
    Corsi = C_0 * exp(-beta_corsi * stagnation / S_max) * diversity

  diversity = mean(pairwise_euclidean(所有个体)) / max_pairwise_euclidean(初始种群)

其中:
  C_0:            初始变异强度 (默认 0.1)
  beta_corsi:     衰减因子 (默认 2.0)
  stagnation:     当前停滞次数
  diversity:      当前种群多样性分数 (0~1)，多样性越低变异越强
```

**自适应特性：**
- 停滞时间越长，Corsi 系数越小（变异幅度衰减，防止永远不收敛）
- 种群多样性越低，Corsi 系数越大（强变异打破同质化）
- 变异仅作用于后 50% 个体（保留精英个体不受扰动）

**恢复机制：** Corsi 变异后，若全局最优在 3 次迭代内更新，重置 `stagnation = 0`；否则继续增加 stagnation 计数。

### 15.4 目标函数设计

#### 15.4.1 目标函数定义

```
f(x) = weighted_mape = 0.5 * MAPE_pv + 0.5 * MAPE_load  (最小化)

其中:
  x = [hidden_size, num_layers, attn_score, vmd_k, vmd_alpha,
       lr, batch_size, dropout, optimizer, input_window]  (10 维向量)

  MAPE_pv  = 训练脚本输出: 光伏测试集 MAPE (第 1 步)
  MAPE_load = 训练脚本输出: 负荷测试集 MAPE (第 1 步)
```

**权重说明：** 光伏和负荷在当前系统中等权重（0.5 : 0.5）。`MssaConfig` 中提供 `objective_weights: [pv_weight, load_weight]` 配置项，允许后续调整为非等权。

#### 15.4.2 目标函数执行流程

```
ObjectiveFunc::evaluate(x) → float:

Step 1: 超参解码 (CPU, < 1ms)
  decode(x) → hyperparams_dict
    - hidden_size: int(floor(x[0] + 0.5)) → 取最近的离散值 (32, 64, 96, 128)
    - num_layers: int(floor(x[1] + 0.5)) → 取最近的离散值 (1, 2, 3)
    - attn_score: argmax(x[2:5]) → {"additive", "dot", "general"}
    - vmd_k: int(clip(floor(x[5] + 0.5), 2, 10))
    - vmd_alpha: float(clip(x[6], 100, 5000))
    - lr: 10^clip(x[7], -4, -2)  (log 空间编码)
    - batch_size: int(floor(x[8] + 0.5)) → 取最近的离散值 (16, 32, 64, 128)
    - dropout: float(clip(x[9], 0.0, 0.5))
    - optimizer: argmax(x[10:13]) → {"Adam", "AdamW", "RMSprop"}
    - input_window: int(floor(x[13] + 0.5)) → 取最近的离散值 (12, 24, 36)
  注: 编码维度 D = 2(离散) + 3(one-hot枚举) + 2(连续/整数) + 3(one-hot枚举) + 3(离散) = 13 维
      但逻辑搜索空间维度 = 10 (见 15.5 详细映射)
  注: 使用 floor(x + 0.5) 代替 round()，确保 .5 边界始终向上舍入，消除银行家舍入的非确定性

Step 2: 缓存查找 (CPU, < 1ms)
  fingerprint = sha256(json.dumps(hyperparams_dict, sort_keys=True))[:16]
  若 fingerprint ∈ cache:
    return cache[fingerprint]  ← 跳过训练

Step 3: 写临时训练配置 (CPU, < 10ms)
  temp_config = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
  将 hyperparams_dict 注入训练配置模板 → 写 temp_config

Step 4: 调用训练脚本 (外部进程, 变长)
  cmd = [sys.executable, "train.py", "--config", temp_config.name]
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
  若 proc.returncode != 0:
    return PENALTY_SCORE  (1e6)

Step 5: 解析验证集 MAPE (CPU, < 10ms)
  result = ResultParser::parse(proc.stdout)
    - 正则提取: "MAPE_pv: <float>"
    - 正则提取: "MAPE_load: <float>"
    - 若解析失败: 返回 PENALTY_SCORE

Step 6: 计算加权分数 + 写入缓存
  score = 0.5 * MAPE_pv + 0.5 * MAPE_load
  cache[fingerprint] = score
  若 score < PENALTY_SCORE:
    记录为有效解
  返回 score

Step 7: 清理临时文件
  finally: os.unlink(temp_config.name)  // 确保训练脚本执行完毕后（无论成功/失败/超时）均删除临时配置文件
```

#### 15.4.3 失败处理与惩罚分数

| 场景 | 检测条件 | 返回值 | 处理 |
|------|----------|--------|------|
| 训练脚本崩溃 | `returncode != 0` | `PENALTY_SCORE = 1e6` | 标记该个体为"无效解"，不参与精英保留 |
| 训练超时 | `subprocess.TimeoutExpired` (默认 600s) | `1e6` | 同上 |
| 输出解析失败 | 正则未匹配到 MAPE 值 | `1e6` | 同上 |
| MAPE 异常（NaN/Inf） | `math.isnan(mape) or math.isinf(mape)` | `1e6` | 同上 |
| MAPE 超出合理范围 | MAPE > 100% | `1e6` | 同上（模型不收敛或数据问题） |

**无效解在种群中的处理：**
- 排序时无效解排在末尾（f = 1e6 >> 正常 MAPE 范围）
- 下一代更新时，无效解对应的个体被加入者和侦察者更新覆盖
- 记录无效解总数到 `search_metadata.invalid_solutions`

#### 15.4.4 评估缓存

**缓存文件：** `tools/mssa_optimizer/mssa_cache.json`

```json
{
  "cache_version": "1.0",
  "created": "2026-06-21T10:00:00",
  "training_data_fingerprint": "sha256-of-training-csv",
  "entries": {
    "a1b2c3d4e5f6a7b8": {
      "hyperparams": {
        "hidden_size": 64, "num_layers": 2, "attn_score": "additive",
        "vmd_k": 5, "vmd_alpha": 2000.0, "lr": 0.001,
        "batch_size": 32, "dropout": 0.2, "optimizer": "Adam", "input_window": 24
      },
      "mape_pv": 0.078, "mape_load": 0.121,
      "weighted_mape": 0.0995,
      "evaluated_at": "2026-06-21T10:05:30"
    }
  }
}
```

**缓存策略：**
- 缓存键 = 超参组合的 SHA256 前 16 位十六进制
- 跨运行持久化 -- 相同训练数据 + 相同超参 = 跳过训练
- `training_data_fingerprint` 校验 -- 训练数据变更时自动失效整个缓存
- 缓存大小预估：50 次迭代 * 30 个体 = 最多 1500 条记录 * ~200B = ~300KB

### 15.5 搜索空间映射

对应 PRD Section 3.5.2 的 10 维超参搜索空间：

| 维度 | 超参名 | 类型 | 编码方式 | 搜索范围 | 编码后表示 |
|------|--------|------|----------|----------|------------|
| 1 | `hidden_size` | 离散 | 整数索引 | {32, 64, 96, 128} | x[0] in [0, 3], 解码: values[floor(x[0]+0.5)] |
| 2 | `num_layers` | 离散 | 整数索引 | {1, 2, 3} | x[1] in [0, 2], 解码: values[floor(x[1]+0.5)] |
| 3 | `attn_score` | 枚举 | One-hot (3 维) | {additive, dot, general} | x[2:5] ∈ R^3, 解码: argmax |
| 4 | `vmd_k` | 整数 | 连续浮点 → 取整 | [2, 10] | x[5] ∈ [2, 10], 解码: clip(floor(x[5]+0.5), 2, 10) |
| 5 | `vmd_alpha` | 连续 | 浮点直接编码 | [100, 5000] | x[6] ∈ [100, 5000], 解码: clip(x[6], 100, 5000) |
| 6 | `lr` | log-连续 | log10 编码 | [1e-4, 1e-2] | x[7] ∈ [-4, -2], 解码: 10^x[7] |
| 7 | `batch_size` | 离散 | 整数索引 | {16, 32, 64, 128} | x[8] ∈ [0, 3], 解码: values[floor(x[8]+0.5)] |
| 8 | `dropout` | 连续 | 浮点直接编码 | [0.0, 0.5] | x[9] ∈ [0, 0.5], 解码: clip(x[9], 0, 0.5) |
| 9 | `optimizer` | 枚举 | One-hot (3 维) | {Adam, AdamW, RMSprop} | x[10:13] ∈ R^3, 解码: argmax |
| 10 | `input_window` | 离散 | 整数索引 | {12, 24, 36} | x[13] ∈ [0, 2], 解码: values[floor(x[13]+0.5)] |

**编码维度总计：** 离散(1+1+1+1=4) + One-hot枚举(3+3=6) + 连续/整数(1+1+1=3) = **13 维实际编码**，对应 **10 维逻辑超参**。

**解码器往返一致性：** `encode(decode(x)) == project(x)`（投影到有效超参空间），`decode(encode(h)) == h`。单元测试 `test_encode_decode()` 验证该性质。

**搜索空间配置化：** 每个超参的搜索范围和离散值集合可通过 `mssa_search_config.yaml` 覆盖（见 15.8），例如缩小 `vmd_alpha` 范围到 `[500, 3000]` 以加速搜索。

**舍入策略：** 统一使用 `floor(x + 0.5)` 代替 Python `round()`，确保 .5 边界始终向上舍入，消除银行家舍入（ties to even）可能引入的非确定性。

### 15.6 收敛与终止

对应 PRD Section 3.5.2 的三个终止条件：

| 序号 | 终止条件 | 配置项 | 默认值 | 判断逻辑 |
|------|----------|--------|--------|----------|
| 1 | 达到最大迭代次数 | `max_iterations` | 50 | `iteration >= max_iterations` |
| 2 | 连续无改善 | `no_improvement_rounds` | 10 | `stagnation_count >= no_improvement_rounds`，其中 `stagnation_count` 在 `|f_best_new - f_best_old| < epsilon` 时 +1，否则归零 |
| 3 | 总搜索时间超限 | `max_wall_time_seconds` | 7200 (2h) | `time.monotonic() - start_time >= max_wall_time_seconds` |

**收敛阈值：** `epsilon = 1e-4`，即加权 MAPE 改善小于 0.01 个百分点视为无改善。此阈值考虑 MAPE 测量的有效精度（通常在 0.1% 左右）、训练随机性（相同超参两次训练的 MAPE 波动约 0.05%-0.15%），epsilon 设为 0.01% 可过滤掉由训练随机性引起的虚假改善。

**终止时行为：**
- 若因条件 1 或 2 终止：`convergence_reason = "max_iter"` 或 `"no_improvement"`
- 若因条件 3 终止：`convergence_reason = "timeout"`，输出当前最优解
- 若因 Ctrl+C 终止：同 timeout 处理，`convergence_reason = "timeout"`

**收敛曲线：** `convergence_curve` 数组按迭代次序记录每次迭代后的全局最优目标函数值。收敛曲线单调不增（精英保留保证）。

**重启策略（非默认）：** 若 `convergence_reason = "no_improvement"` 且最优 MAPE > 人工基线 MAPE，运维可通过 `restart_on_stagnation: true` 配置自动重启动 MSSA（最多 3 次），每次使用不同的 `random_seed` 和佳点集偏移量。若 3 次重启后最优 MAPE 仍 > 人工基线 MAPE * 1.1，`quality_flag = "unusable"`，建议降级为 IPSO 或人工调参。

### 15.7 JSON 输出格式

严格对齐 PRD Section 7.4.2 JSON Schema。`output.py::SearchOutput::to_json()` 输出以下结构：

```json
{
  "search_metadata": {
    "algorithm": "MSSA",
    "start_time": "2026-06-21T10:00:00",
    "end_time": "2026-06-21T11:45:30",
    "total_iterations": 38,
    "convergence_reason": "no_improvement",
    "elapsed_seconds": 6330.5,
    "population_size": 30,
    "discoverer_ratio": 0.20,
    "scout_ratio": 0.10,
    "invalid_solutions": 3,
    "cache_hits": 187,
    "total_evaluations": 842
  },
  "best_hyperparameters": {
    "hidden_size": 64, "num_layers": 2, "attn_score": "additive",
    "vmd_k": 5, "vmd_alpha": 1800.0, "lr": 0.0015,
    "batch_size": 32, "dropout": 0.25, "optimizer": "Adam",
    "input_window": 24
  },
  "best_objective": {
    "weighted_mape": 0.0955, "mape_pv": 0.076, "mape_load": 0.115
  },
  "convergence_curve": [0.132, 0.124, 0.118, 0.112, 0.107, ...],
  "per_parameter_trajectory": {
    "hidden_size": [1, 1, ...],
    "num_layers": [1, 1, ...],
    "attn_score": [0, 0, ...],
    "vmd_k": [6, 6, 6, 5, 5, ...],
    "vmd_alpha": [2500, 2500, 2200, ..., 1800],
    "lr": [0.003, 0.003, 0.0025, ..., 0.0015],
    "batch_size": [2, 2, ...],
    "dropout": [0.3, 0.3, 0.25, ...],
    "optimizer": [0, 0, ...],
    "input_window": [1, 1, ...]
  },
  "quality_flag": "usable",
  "additional_info": {
    "population_size": 30, "discoverer_count": 6,
    "joiner_count": 21, "scout_count": 3,
    "opposition_learning_frequency": 5, "corsi_stagnation_threshold": 10,
    "corsi_initial_strength": 0.1, "corsi_decay_factor": 2.0,
    "random_seed": 42, "invalid_solutions": 3,
    "cache_hits": 187, "total_evaluations": 842,
    "stopped_early": false, "final_stagnation_count": 10,
    "final_diversity": 0.12
  }
}
```

**`per_parameter_trajectory` 值编码说明：**

| 超参 | 轨迹数组值含义 | 示例 |
|------|--------------|------|
| `hidden_size` | 离散选项索引 (0=32, 1=64, 2=96, 3=128) | `[1, 1, ...]` = 64 |
| `num_layers` | 离散选项索引 (0=1, 1=2, 2=3) | `[1, 1, ...]` = 2 |
| `attn_score` | One-hot argmax 索引 (0=additive, 1=dot, 2=general) | `[0, 0, ...]` = additive |
| `vmd_k` | 实际 K 值 (整数 2~10) | `[5, 5, ...]` = K=5 |
| `vmd_alpha` | 实际 alpha 值 (浮点) | `[1800.0, ...]` |
| `lr` | 实际学习率 (浮点) | `[0.0015, ...]` |
| `batch_size` | 离散选项索引 (0=16, 1=32, 2=64, 3=128) | `[2, 2, ...]` = 64 |
| `dropout` | 实际 dropout 率 (浮点) | `[0.25, ...]` |
| `optimizer` | One-hot argmax 索引 (0=Adam, 1=AdamW, 2=RMSprop) | `[0, 0, ...]` = Adam |
| `input_window` | 离散选项索引 (0=12, 1=24, 2=36) | `[1, 1, ...]` = 24 |

> **注（M-09 跟踪）：** 轨迹采用混合编码（连续型存储实际值，离散/枚举型存储索引），下游消费者需根据编码说明表区分哪些字段需二次解码。建议在实现阶段对 `per_parameter_trajectory` 同时提供 `_index` 和 `_value` 两个版本，或统一存储解码后的可读值。

**自校验：** `output.py::validate_output()` 在写入文件前自动执行以下校验：
1. `best_hyperparameters` 的所有必填键存在且类型正确
2. `best_objective.weighted_mape ≈ 0.5 * mape_pv + 0.5 * mape_load`（容差 1e-6）
3. `len(convergence_curve) == total_iterations`
4. `per_parameter_trajectory` 的每个键对应长度 = `total_iterations`
5. `convergence_reason` 为合法枚举值
6. `quality_flag` 为合法枚举值
7. 校验失败则写文件但标记 `quality_flag = "unusable"` 并记录 ERROR 日志

### 15.8 配置文件设计

**配置文件路径：** `tools/mssa_optimizer/config/mssa_search_config.yaml`

```yaml
# ============================================================================
# MSSA/IPS 超参搜索配置文件 (v1.0, 2026-06-21)
# 用于 MUPC 第三轮 MSSA/IPS 超参自动优化
# ============================================================================

# ---- 算法选择 ----
algorithm: "MSSA"             # "MSSA" (推荐) 或 "IPSO" (备选降级)
                              # IPSO 在搜索速度优先时使用，全局搜索能力弱于 MSSA

# ---- 随机种子（用于确定性复现）----
random_seed: 42               # 任意整数，相同 seed + 相同数据 → 确定性结果

# ---- 种群参数 ----
population:
  size: 30                    # 种群个体总数 [10, 100]。
                              # 若搜索空间较大或训练成本允许，可适当增大 N 至 50~60 以提升全局搜索能力
  discoverer_ratio: 0.20      # 发现者比例 [0.10, 0.30]
  scout_ratio: 0.10           # 侦察者比例 [0.05, 0.20]
  # 加入者比例 = 1 - discoverer_ratio - scout_ratio (自动计算)

# ---- 增强策略（仅 MSSA 生效，IPSO 忽略）----
enhancement:
  good_point_set: true        # 佳点集初始化
  opposition_learning: true   # 反向学习增强
  opposition_frequency: 5     # 反向学习执行频率（每 N 次迭代）
  corsi_mutation: true        # Corsi 变异扰动
  corsi_stagnation: 10        # 触发 Corsi 的停滞阈值（连续无改善次数）
  corsi_strength: 0.1         # Corsi 变异初始强度 C_0
  corsi_decay: 2.0            # Corsi 衰减因子 beta

# ---- 终止条件 ----
termination:
  max_iterations: 50          # 最大迭代次数
  no_improvement_rounds: 10   # 连续无改善轮次（|delta_MAPE| < epsilon）
  epsilon: 1.0e-4             # 收敛阈值（加权 MAPE 绝对改善）
  max_wall_time_seconds: 7200 # 总搜索时间硬上限 (2 小时)
  restart_on_stagnation: false  # 停滞时是否自动重启（最多 3 次）[默认 false]

# ---- 目标函数 ----
objective:
  pv_weight: 0.5              # 光伏 MAPE 权重
  load_weight: 0.5            # 负荷 MAPE 权重
  penalty_score: 1000000.0    # 训练失败惩罚分数
  training_timeout_seconds: 600  # 单次训练超时 (10 分钟)
  cache_enabled: true         # 是否启用评估缓存
  cache_path: "mssa_cache.json"  # 缓存文件路径（相对 mssa_optimizer/）

# ---- 训练脚本调用 ----
training:
  script_path: "../../train.py"  # 训练脚本路径（相对 mssa_optimizer/ 或绝对路径）
  python_executable: null      # Python 解释器 (null = sys.executable)
  extra_args: []               # 额外命令行参数 (如 ["--gpu", "0"])

# ---- 搜索空间覆盖（可选，用于缩小搜索范围）----
# 未指定的超参使用 PRD Section 3.5.2 的默认搜索范围
search_space_overrides:
  # hidden_size: [32, 64, 96, 128]           # 离散值列表
  # num_layers: [1, 2, 3]                     # 离散值列表
  # attn_score: ["additive", "dot", "general"] # 枚举值列表
  # vmd_k: {min: 2, max: 10}                  # 整数范围
  # vmd_alpha: {min: 100, max: 5000}           # 连续范围
  # lr: {min: 1.0e-4, max: 1.0e-2}            # 连续范围（线性空间，内部 log 编码）
  # batch_size: [16, 32, 64, 128]             # 离散值列表
  # dropout: {min: 0.0, max: 0.5}             # 连续范围
  # optimizer: ["Adam", "AdamW", "RMSprop"]    # 枚举值列表
  # input_window: [12, 24, 36]                # 离散值列表

# ---- 输出 ----
output:
  result_path: "mssa_search_result.json"  # 搜索结果输出路径
  verbose: true                            # 是否输出每次迭代的详细日志
  log_level: "INFO"                        # 日志级别: DEBUG / INFO / WARNING / ERROR
```

**配置校验规则（`config.py::validate()`）：**

| 校验项 | 条件 | 失败处理 |
|--------|------|----------|
| `algorithm` | 必须为 `"MSSA"` 或 `"IPSO"` | 拒绝执行 |
| `population.size` | `10 <= size <= 100` | 拒绝执行 |
| `discoverer_ratio + scout_ratio` | `< 1.0` | 拒绝执行（加入者数量 = 0 不可接受） |
| `max_iterations` | `>= 1` | 拒绝执行 |
| `max_wall_time_seconds` | `>= 300` (至少 5 分钟) | 拒绝执行 |
| `pv_weight + load_weight` | `≈ 1.0` (容差 1e-6) | 自动归一化并记录 WARN |
| `penalty_score` | `> max_possible_mape` (即 > 1.0) | 使用默认值 1e6 |
| `training_timeout_seconds` | `>= 60` | 使用默认值 600 |
| `training.script_path` | 必须指向存在的文件 | 拒绝执行 |
| 搜索空间覆盖值 | 每个超参须合法（离散值 >= 2 个选项、连续值 min < max） | 拒绝执行 |

**IPSO 降级路径配置切换：**

当 `algorithm: "IPSO"` 时：
- `enhancement.*` 全部忽略（IPSO 不使用佳点集、反向学习、Corsi 变异）
- `population` 参数沿用（IPSO 同样使用种群概念）
- 输出 JSON 中 `search_metadata.algorithm = "IPSO"`，`additional_info` 中不含 MSSA 特有的 `opposition_learning_frequency`、`corsi_*` 等字段
- 其余 JSON Schema 完全一致（PRD Section 7.4.2 兼容）



## 附录A：修订记录

### v2.3 修订记录 (2026-06-07)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | SCENE-01 恢复电压质量惩罚 | 5.3 奖励函数 | 增加 w4 * P_voltage_deviation 项；三相电压幅值通过 P/Q 协同控制可主动调节，三相不平衡维持实时控制核心独立处理 |
| 2 | SceneWeights 表更新 | 5.8 | MODE-01 新增 w4=1.0（电压质量）列 |
| 3 | 版本号更新 | 文档头部 | v2.2 → v2.3 |

**修订依据：** MODE-01 农网灌溉场景中，高电压（光伏超发）和低电压（灌溉抽水导致）均需 AI 引擎主动干预。
原 v2.1 以"三相不平衡不涉及电池充放"为由完全移除电压奖励属误判——电压幅值偏差与电池充放电、无功输出密切相关，
AI 引擎可通过 p_batt/q_batt 协同控制主动调节。v2.3 仅恢复电压幅值惩罚，三相不平衡治理维持实时控制核心独立处理。

### v2.4 修订记录 (2026-06-07)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 分层控制架构 | 1.1/4.1a/5.3 | 动作空间4维→2维；Q控制交实时控制核心，RL仅输出P_batt+Load_shedding |
| 2 | 电压死区机制 | 5.3 | ±5%死区，越限连续2步触发惩罚 |
| 3 | 功率变化率惩罚R_ramp | 5.3 | R_ramp = λ·|ΔP_batt| |
| 4 | SceneWeights表更新 | 5.8 | MODE-01新增w5列 |
| 5 | 版本号更新 | 文档头部 | v2.3 → v2.4 |

**修订依据：** 300kW错峰启停产生高频随机脉冲，Q属ms级快变量应由实时控制闭环；AI专注能量管理；电压死区防高频脉冲过度响应；变化率惩罚延长电池寿命。

### v2.5 修订记录 (2026-06-08)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | FusedSystemState 新增 q_realtime_margin | 3.1/D1 | 实时模块剩余无功容量比例 [0.0, 1.0]，默认 0.5 |
| 2 | FusedSystemState 新增 season_encoding | 3.1/D7 | 季节 one-hot (6维)，默认 [0,0,0,1,0,0] 常规季 |
| 3 | FusedSystemState 新增 time_period_encoding | 3.1/D7 | 时段 one-hot (2维)，默认 [1,0] 白天 |
| 4 | to_input_vector 扩展至 56 维 | 3.3 | D7 (8字段) 追加至向量末尾 |
| 5 | RewardThresholdConfig 结构体 | 10.3 | voltage_deadband, q_margin_threshold, voltage_high_limit, soc_critical, voltage_penalty_high/low |
| 6 | calc_agri_v2_5 替代 calc_agri | 5.3 | 条件触发电压惩罚 + 自适应损耗系数 α(s) + 弃光电压前置 |
| 7 | compute_alpha 三态 | 5.3 | α(s) ∈ {1.0, 0.2, 3.0}，优先级：SOC极低 > 电压支撑 > 常规 |
| 8 | conditional_voltage_penalty | 5.3 | 仅 q_margin <= 10% 且越限2步时触发 |
| 9 | new_with_thresholds 工厂方法 | 5.1 | 支持自定义阈值配置 |
| 10 | 版本号更新 | 文档头部 | v2.4 → v2.5 |

**修订依据：** 专家评审指出状态空间缺少实时模块能力边界反馈 + 奖励函数未体现有功边际贡献。v2.5 通过 q_realtime_margin 实现 AI 对实时模块裕度的感知，通过条件触发电压惩罚实现分层控制原则。

### v2.6 修订记录 (2026-06-10)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 农网台区参数更新 | config.rs / reward_calculator.rs / data_fusion.rs / strategy-engine | 变压器容量 500kVA→200kVA，电池容量 200kWh→100kWh，最大充放电功率 100kW→50kW，合同需量 500kW→200kW |
| 2 | 版本号更新 | 文档头部 | v2.5 → v2.6 |

**修订依据：** 农网台区新规格落地：变压器 200kVA、光伏 150kW、储能 50kW/100kWh、居民负荷 60kW、农业冲击负荷最高 120kW。代码默认值已同步更新。

### v2.7 修订记录 (2026-06-13)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 双参数 ActionOutput | 4.5a | p_batt_set → p_ref + k_droop 双参数结构 |
| 2 | parse_action_output 双参数解析 | 4.5b | 输出格式 [p_ref, k_droop, load_shedding, pv_limit, confidence] |
| 3 | ActionValidator 双参数模式 | 4.8 | 新增 droop_range 和 dual_mode 字段 |
| 4 | validate_dual 方法 | 4.8 | 实现 ACT-DUAL-01~05 校验规则 |
| 5 | k_droop 范围动态更新 | 4.8 | update_droop_range / get_droop_range 方法 |
| 6 | TCP 帧 v2.0 格式 | protocol.rs / tcp_server.rs | ControlCmdPayloadV2 双参数传输 |
| 7 | 通信中断降级逻辑 | ai_integration.rs | IntercoreConnectionState 保持最后有效参数 |
| 8 | 版本号更新 | 文档头部 | v2.6 → v2.7 |

**修订依据：** 动作空间重构设计文档（2026-06-13）已合并。双参数模式实现时间尺度解耦：AI 负责稳态全局优化（P_ref），执行器负责毫秒级暂态调节（k_droop × ΔV）。通信中断时执行器保持最后有效参数，继续下垂控制，保障本质安全不停机。

### v2.8 修订记录 (2026-06-13)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | P-Q 协同度奖励 R_PQ_coordination | 5.3 SCENE-01 | 替代原 P_voltage_deviation 电压惩罚；Q 有裕度时奖励"偷懒"，Q 饱和时奖励正确出手 |
| 2 | 弃光奖励差异化 | 5.3 SCENE-01 | 高电压（v_avg >= 1.05）时根据 AI 动作方向差异化：充电消纳给奖励，放电给惩罚 |
| 3 | 下垂系数平滑惩罚 R_smooth | 5.3 SCENE-01 | R_smooth = -\|Δk_droop\| - λ·max(0, k_droop - K_MAX) |
| 4 | RewardCalculator 结构体扩展 | 5.2 | 新增 last_k_droop 字段用于 R_smooth 计算 |
| 5 | calc_pq_coordination 方法 | 5.3 | 实现 P-Q 协同度奖励逻辑 |
| 6 | calc_smooth_penalty 方法 | 5.3 | 实现下垂系数平滑惩罚逻辑 |
| 7 | 权重表更新 | 5.3 | w4 改为 P-Q 协同度，新增 w7 下垂平滑惩罚 |
| 8 | 版本号更新 | 文档头部 | v2.7 → v2.8 |

**修订依据：** 专家评审指出原有电压惩罚设计会使 AI 在 Q 饱和时"两难"——调了也没用还被罚，最终退化到零策略。P-Q 协同度奖励将考核从"结果惩罚"转变为"行为奖励"，更符合强化学习正向激励原理，同时新增 R_smooth 防止 AI 设置极大 k_droop 导致系统震荡。

### v2.9 修订记录 (2026-06-14)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 新增 robustness_manager.rs | ai-engine/src/ | 电压异常应急策略管理器 |
| 2 | AnomalyType 枚举 | robustness_manager.rs | VoltageSag/VoltageSurge/BatterySocCritical/BatterySocOverfull/CommunicationTimeout |
| 3 | detect_anomaly 方法 | robustness_manager.rs | 异常类型检测逻辑 |
| 4 | get_robust_action 方法 | robustness_manager.rs | 根据异常类型返回应急动作 |
| 5 | calc_pq_coordination 方法 | reward_calculator.rs | P-Q 协同度奖励逻辑 |
| 6 | calc_smooth_penalty 方法 | reward_calculator.rs | 下垂系数平滑惩罚逻辑 |
| 7 | calc_pv_reward_v2_8 方法 | reward_calculator.rs | 弃光奖励差异化（高电压时放电惩罚） |
| 8 | calc_agri_v2_8 方法 | reward_calculator.rs | SCENE-01 v2.8 奖励函数（7 权重） |
| 9 | AiIntegrator 集成 RobustnessManager | strategy-engine/ai_integration.rs | dispatch_ai_decision 前进行异常检测 |
| 10 | dispatch_robust_action 方法 | strategy-engine/ai_integration.rs | 分发应急动作 |
| 11 | SceneWeights 扩展至 7 权重 | ai-engine/src/config.rs | seasonal_load_management: [f64; 7] |
| 12 | 版本号更新 | 文档头部 | v2.8 → v2.9 |

**修订依据：** v2.9 实现两项核心功能：(1) RobustnessManager 电压异常应急策略，检测 VoltageSag/VoltageSurge/BatterySocCritical/BatterySocOverfull 并返回应急动作；(2) v2.8 奖励函数代码落地，P-Q 协同度奖励 + 下垂平滑惩罚。

### v2.10 修订记录 (2026-06-14)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | FusedSystemState 扩展至 59 维 | data_fusion.rs | D9 新增安全覆盖状态（3 字段） |
| 2 | FrameType::SafetyOverride = 0x0040 | protocol.rs | 新增安全覆盖帧类型 |
| 3 | DataUploadPayload 结构体 | tcp_server.rs | 含 q_realtime_margin 字段 |
| 4 | SafetyOverridePayload 结构体 | tcp_server.rs | 含 trigger_reason, override_p_ref, duration_ms, recovery_condition |
| 5 | IntercoreConnectionState 扩展 | tcp_server.rs | q_margin, safety_override_* 字段 |
| 6 | safety_override_penalty() 方法 | reward_calculator.rs | 安全覆盖惩罚（v2.10 新增） |
| 7 | SCENE-01 扩展 8 权重 | 5.3 奖励函数 | w8=1.0 安全覆盖惩罚 |
| 8 | IntercoreAdapter 数据源适配器 | data_fusion.rs | 从核间通信状态获取 q_realtime_margin 和安全覆盖状态 |
| 9 | 版本号更新 | 文档头部 | v2.9 → v2.10 |
| 10 | **v2.10 短期实现：R1 影子模型验证+渐进式切换** | 7a 节新增 | SafeOnlineUpdater、ShadowModel、GradualSwitcher；安全约束验证 + 性能验证 + 渐进式切换 |
| 11 | **v2.10 短期实现：R2 折扣累积奖励机制** | 5.9 节新增 | DiscountedAccumulator；gamma 折扣因子 [0.9, 0.999]，缓冲区 1000 |
| 12 | **v2.10 短期实现：R3 场景切换平滑过渡** | 4.9 节新增 | SmoothSceneTransition；10 步线性插值过渡 |

**修订依据：** v2.10 实现安全增强功能：(1) q_realtime_margin 数据通道通过核间 DataUpload 帧实时同步；(2) SafetyOverride 帧类型（0x0040）定义实时控制模块临时覆盖 AI 有功指令的接口规范；(3) FusedSystemState 扩展至 59 维，AI 引擎感知安全覆盖事件并在奖励函数中获得惩罚；(4) v2.10 短期实现三项功能：影子模型验证+渐进式切换、折扣累积奖励机制、场景切换平滑过渡。

### v2.11 修订记录 (2026-06-14)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **v2.11 中期实现：自适应权重优化器** | 7b 节新增 | AdaptiveWeightOptimizer（MetaRL）+ ParetoWeightOptimizer（NSGA-II）+ PerformanceCollector trait |
| 2 | **v2.11 中期实现：冲击负荷概率预测** | 5.10 节新增 | LSTM 分位数预测（predict_quantiles）+ 冲击概率计算 + LoadCovariates + WeatherService trait |
| 3 | AdaptiveOptimizerConfig | config.rs | enabled, update_interval_hours, meta_learning_rate, weight_bounds, constraints |
| 4 | ParetoOptimizerConfig | config.rs | enabled, population_size, generations, crossover_rate, mutation_rate |
| 5 | validate_reward_drift 方法 | adaptive_weight_optimizer.rs | AWO-06：验证奖励偏移 < 5% |
| 6 | FusedSystemState 扩展 | data_fusion.rs | load_forecast_quantiles, shock_load_probability, base_load（v2.11 新增） |
| 7 | 版本号更新 | 文档头部 | v2.10 → v2.11 |

**修订依据：** v2.11 实现两项核心功能：(1) 自适应权重优化器（AdaptiveWeightOptimizer + ParetoWeightOptimizer），基于历史性能数据自动调优奖励函数权重，减少人工调参依赖；(2) 冲击负荷概率预测（LSTM 分位数预测），输出 P10/P50/P90 分位数并计算冲击负荷概率，增强需量控制能力。

### v2.12 修订记录 (2026-06-14)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **R-01 奖励子项标准化** | 5.3 SCENE-01 | 各子项标准化到 `[-1, 1]` 区间：r_pv_norm = r_pv/100、p_batt_deg_norm = p_batt_deg×10、p_trafo_norm = (load-0.75)/0.25、r_pq_norm = r_pq/50、r_ramp_norm = r_ramp×10、r_voltage_slope_norm = r_voltage_slope×10、r_smooth_norm = r_smooth/130、r_safety_override_norm = r_safety_override/100 |
| 2 | **R-02 引入塑造奖励** | 5.3 SCENE-01 | 新增 `overload_warning(load)` 和 `soc_warning(soc)` 方法，提前预警稀疏事件 |
| 3 | **R-03 SOC 均衡奖励** | 5.3 SCENE-01 | 新增 `soc_balance_reward(soc, λ)` 方法，鼓励 SOC 保持在 50% 附近 |
| 4 | 标准化公式说明 | 5.3 SCENE-01 | 新增 v2.12 改进说明注释，解释标准化系数选择依据 |
| 5 | 新增测试用例 | 单元测试 | test_v2_12_overload_warning_*、test_v2_12_soc_warning_*、test_v2_12_soc_balance_reward_*、test_v2_12_normalized_* 等 15+ 测试用例 |
| 6 | 版本号更新 | 文档头部 | v2.11 → v2.12 |

**修订依据：** v2.12 基于 SCENE-01 台区季节性负荷模式的专家建议实现三项改进：(1) 奖励子项标准化解决量纲不一致问题，加速 RL 收敛；(2) 塑造奖励提前预警稀疏事件（变压器过载、SOC 边界），帮助 RL 学习避免危险状态；(3) SOC 均衡奖励延长电池寿命，避免 SOC 长期偏向极值。

### v2.13 修订记录 (2026-06-14)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **P-Q协同Sigmoid平滑化** | 5.15.2 | 硬阈值 `if q_margin > 0.10` 替换为 Sigmoid 平滑过渡，消除策略震荡 |
| 2 | **Welford动态归一化** | 5.15.3 / reward_normalizer.rs（新增） | RunningStats 在线算法替代硬编码归一化系数，提升跨台区泛化能力 |
| 3 | **状态改善率奖励** | 5.15.4 | R_improve = w × (V_dev_prev - V_dev_curr) × sign(P_action)，建立"动作-效果"因果链 |
| 4 | **冲击负荷预备度奖励重构** | 5.15.5 | 从 R_shock_response 重构为 R_readiness 预备度奖励，鼓励预留缓冲空间 |
| 5 | **PER+KL正则化** | 5.15.6 / online_updater.rs | 优先经验回放 + KL 散度约束，防止在线微调分布偏移 |
| 6 | **策略混合替代权重混合** | 5.15.7 / mode_selector.rs | a_blended = (1-α)×a_old + α×a_new，过渡期同时运行新旧策略 |
| 7 | ActionOutput 新增 confidence 字段 | 4.4/4.5 | 5 维动作空间扩展为含决策置信度 |
| 8 | 版本号更新 | 文档头部 | v2.12 → v2.13 |

**修订依据：** 基于专家建议（2026-06-14）中有参考意义的 P0/P1 项，解决 P-Q 协同硬阈值震荡、归一化系数泛化差、缺乏因果链、冲击负荷响应不精确、在线微调偏移风险、场景切换策略真空六项问题。

### v2.14 修订记录 (2026-06-15)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **SafetyOverride 惩罚函数重构** | 5.3 SCENE-01 | 分层计算策略：样本不足时原因固定惩罚，样本充足时比例+连续次数惩罚，归一化至 [-1, 0] |
| 2 | FusedSystemState 扩展至 78 维 | 3.5/4.3 | D9 新增 safety_override_consecutive（连续触发次数）+ safety_override_ratio（滑动窗口覆盖比例），D9 从 2 维扩展至 4 维 |
| 3 | **P-Q 协同度与 SafetyOverride 互斥逻辑** | 5.3 SCENE-01 | safety_override_active=true 时跳过 P-Q 协同度惩罚，避免双重惩罚 |
| 4 | to_input_vector 更新至 78 维 | 3.6 | 序列化布局更新：D9 扩展 2 维（consecutive + ratio），debug_assert 更新至 78 |
| 5 | 版本号更新 | 文档头部 | v2.13 → v2.14 |

**修订依据：** v2.14 增强 SafetyOverride 惩罚函数的精细度：(1) 引入连续触发次数和滑动窗口覆盖比例两个新特征；(2) 分层计算避免小样本偏差；(3) 新增互斥逻辑避免 SafetyOverride 与 P-Q 协同度的双重惩罚，使奖励信号更准确。

### v2.15 修订记录 (2026-06-17)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **动作空间精简为 2 维** | §4.1/4.2/4.5/4.6 | ActionOutput 从 5 字段(p_ref, k_droop, load_shedding, pv_limit, confidence)精简为 2 字段(p_ref, k_droop)；load_shedding/pv_limit 下沉至 strategy-engine，confidence 保留在 ModelOutput |
| 2 | 动作空间对比表更新 | §4.2 | 新增 v2.15(2维) 列，标记 load_shedding/pv_limit/confidence 下沉状态 |
| 3 | parse_action_output 精简 | §4.6.2/4.6.3 | 解析格式从 `[p_ref, k_droop, load_shedding, pv_limit, confidence]` 改为 `[p_ref, k_droop]` |
| 4 | ActionValidator 约束规则精简 | §4.8 | ACT-05(load_shedding)/ACT-06(pv_limit)下沉至 strategy-engine，ACT-07(dispatch)合并至 ACT-DUAL-04；v2.15 保留 4 条双参数校验规则 |
| 5 | blend_actions 精简 | §5.15.7 | 5 维插值 → 2 维插值（仅 p_ref + k_droop） |
| 6 | full_decision_cycle DataPoint | §7.3 | 输出向量从 5 元素改为 2 元素 |
| 7 | ADR-005 更新 | §13.5 | 从 7 条约束规则更新为 4 条双参数校验规则 |
| 8 | 版本号更新 | 文档头部 | v2.14 → v2.15 |

**修订依据：** PRD v2.15 (`[REVIEWED: PASS]`) 将动作空间从 5 维精简为 2 维：(1) p_ref/k_droop 通过核间通信下发实时控制模块；(2) load_shedding 下沉至 strategy-engine 需量控制策略独立执行；(3) pv_limit 下沉至 strategy-engine 防逆流策略独立执行；(4) confidence 保留在 ModelOutput 中供 action_validator 内部校验。精简后 AI 引擎专注于核心 P-Q 协同控制，策略引擎承担本地设备控制职责。

### v3.1 精简修订记录 (2026-06-21)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **精简第 2 章 LSTM 模型设计** | §2.1~2.6 | 移除 TCN 备选方案引用、移除 `LstmOutput.confidence` 过时字段、移除 `LstmInput` 中 `history: Vec<f32>` 过时定义、移除硬编码 `/60` 步长计算为 `step_seconds` 统一计算、移除 v2.16 版本标记、精度要求更新为 v3.0 目标 |
| 2 | 新增第 2 章交叉引用 | §2 | 新增引导注释："完整的预测管线设计（VMD 分解、Attention、误差修正、降级层级）见第 14 章" |
| 3 | 更新版本头部 | 文档头部 | v3.0 → v3.1，标注精简与交叉引用变更；PRD 引用 v2.15 → v3.1 |
| 4 | 更新附录A | 附录A | 新增 v3.1 本修订记录 |

### v3.0 合并修订记录 (2026-06-21)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | **合并预测增强分层混合架构设计** | 全文 | 将 `docs/superpowers/plans/2026-06-21-预测增强分层混合架构-DESIGN.md` (v3.0, `[DESIGN_APPROVED]`, ~1876 行) 完整合并入本文档 |
| 2 | 新增第14章 LSTM预测增强管线设计 | §14 | 技术选型（VMD纯Rust/Attention ONNX内嵌/MIC离线/BiLSTM双模型+Go/No-Go/误差修正BiLSTM）、模块划分（5新增+5修改+4不修改）、数据流设计（VMD+Attention/误差修正/降级路径4条）、接口定义（VmdDecomposer/PredictionPipeline/配置/模型文件管理）、性能预算（4路径组合+模型大小+内存）、配置设计（完整YAML段+热加载）、降级设计（8级层级+自动升降级）、测试策略、风险与缓解 |
| 3 | 新增第15章 MSSA超参优化工具设计 | §15 | MSSA算法设计（佳点集初始化/三群体更新/反向学习/Corsi变异/13维混合编码）、目标函数设计（加权MAPE+缓存+惩罚分数）、搜索空间映射（10维逻辑→13维编码）、收敛与终止（3条件）、JSON输出格式（对齐PRD 7.4.2）、配置文件设计（MSSA+IPSO降级路径）、文件结构（~1095行Python） |
| 4 | 更新第9章 文件结构 | §9 | 新增预测增强管线文件（vmd.rs/prediction_pipeline.rs/pipeline_config.rs/residual_buffer.rs/model_validator.rs）+ MSSA工具目录（tools/mssa_optimizer/）+ safety_wrapper.rs/reward_normalizer.rs |
| 5 | 更新9.1 lib.rs模块导出 | §9.1 | 新增预测增强模块导出（VMD/Pipeline/Config/ResidualBuffer）+ SafetyRLWrapper导出 |
| 6 | 更新第10章 配置结构 | §10.1b/10.1c | 新增 PredictionEnhancementConfig 完整Rust结构体定义（5个子配置）+ YAML prediction_enhancement 段 + 热加载策略表 + 默认值策略表 |
| 7 | 更新第11章 错误类型 | §11.1/11.2 | 新增 6 个预测增强错误变体（VmdFailed/VmdNotConverged/AttentionDegraded/ErrorCorrectionFailed/ModelValidationFailed/ResidualBufferInsufficient）+ 错误分类表新增预测增强行 |
| 8 | 更新第13章 ADR | §13.8~13.18 | 新增 ADR-008~ADR-018（VMD纯Rust/Attention ONNX内嵌/MIC离线/BiLSTM双模型+GoNoGo/误差修正独立Runtime/预测管线挂载/三模型独立OTA/data_fusion不修改/MSSA算法与编码/MSSA目标函数与缓存/IPSO降级）共 11 条架构决策记录 |
| 9 | 更新TOC | 目录 | 新增第14章+第15章全部子章节条目 |
| 10 | 更新版本头部 | 文档头部 | v2.15 → v3.0，标注合并来源与三轮覆盖范围 |
| 11 | 更新AiEngineConfig | §10.1 | 新增 `prediction_enhancement: Option<PredictionEnhancementConfig>` 和 `safety_wrapper: SafetyWrapperConfig` 字段 |

**合并依据：** 预测增强分层混合架构设计文档（v3.0, `[DESIGN_APPROVED]`）经过三轮设计评审通过（v1.1 5项Minor + v2.0 7项Minor + v3.0 5项Minor），覆盖 VMD + Attention + BiLSTM + 误差修正 + MSSA 五大技术模块的完整设计。合并后本文档成为 MUPC AI 引擎的单一权威设计文档，消除跨文档查找的设计碎片化问题。合并过程中保留源文件全部设计决策、数据流图、配置定义、性能预算、降级设计、测试策略和ADR，无遗漏。源文件保留在 `docs/superpowers/plans/` 目录作为设计演进历史记录。

