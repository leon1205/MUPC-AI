---

# MUPC AI 引擎 - 模块设计文档

[DESIGN_APPROVED] — v2.15 动作空间精简（5维→2维）

| 版本 | 日期       | 作者   | 状态 |
| ---- | ---------- | ------ | ---- |
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

**对应 PRD:** `docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md` v2.15 (`[REVIEWED: PASS]`)

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

### 2.1 功能概述

LSTM 时序预测模型负责预测未来 15~30 分钟（默认 15 分钟，可配置）的光伏出力和负荷功率，为 RL 决策模型提供前瞻性输入。模型架构支持 LSTM 作为主模型，TCN 作为备选方案，两者均通过 ONNX 导出并部署为 .rknn。

### 2.2 预测规格

| 项目 | 规格 |
|------|------|
| 预测目标 | 光伏出力 (PV forecast)、负荷功率 (Load forecast) |
| 负荷分类 | 基荷（基础用电）、可调负荷（柔性负荷）、冲击负荷（概率预测） |
| 预测范围 | 15 分钟（默认），可配置扩展至 30 分钟 |
| 采样间隔 | 每分钟 1 个采样点 |
| 输入数据 | 历史光伏出力、历史负荷功率、气象数据（光照、温度） |
| 输入窗口 | 60 分钟，由 `LstmConfig.input_window_secs` 配置 |
| 输出窗口 | 15~30 分钟，由 `LstmConfig.output_horizon_secs` 配置 |
| 模型格式 | ONNX（训练）--> INT8 量化 --> .rknn（部署）|
| 精度要求 | 光伏 MAPE <= 10%, 负荷 MAPE <= 15% |

### 2.3 接口定义

```rust
/// LSTM 模型输入
#[derive(Debug, Clone)]
pub struct LstmInput {
    /// 历史时间序列数据（按时间顺序），长度 = input_window_secs / 60
    pub history: Vec<f32>,
    /// UTC 时间戳（秒）
    pub timestamp: i64,
}

/// LSTM 模型输出
#[derive(Debug, Clone)]
pub struct LstmOutput {
    /// 预测值向量（未来 N 个时间步），长度 = output_horizon_secs / 60
    pub predictions: Vec<f32>,
    /// 置信度 (0.0 ~ 1.0)，基于输出方差的简化估计
    pub confidence: f64,
}
```

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
  1. 定义 LSTM/TCN 模型 (torch.nn.LSTM / TCN)
  2. 训练至收敛 (MAPE <= 10%)
  3. torch.onnx.export() 导出 ONNX 模型
  4. rknn-toolkit2 加载 ONNX 模型
  5. 校准数据集 (calibration dataset) INT8 量化
  6. rknn.build() 生成 .rknn 模型文件 (<= 5MB)

部署阶段 (RK3588):
  1. 加载 .rknn 文件到 RKNN Runtime
  2. NPU 执行 INT8 整数推理
  3. 输出 f32 预测值
```

### 2.6 预测向量长度处理

预测输出向量长度由 `LstmConfig.output_horizon_secs / 60` 计算。当实际输出长度与配置不符时：
- 超出部分：截断（取前 N 个值）
- 不足部分：补零填充到配置长度

```rust
let output_size = self.config.output_horizon_secs as usize / 60;
let predictions: Vec<f32> = output.into_iter()
    .take(output_size)
    .chain(std::iter::repeat(0.0))
    .take(output_size)
    .collect();
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
    // ------- D1: 实时数据 (10 个 RL 字段，v2.5 新增 q_realtime_margin) -------
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
    /// 序列化为 78 维输入向量（v2.14）
    /// 布局：
    ///   [0..10]  D1 实时数据 (10 个标量，含 q_realtime_margin)
    ///   [10..25] D2 pv_forecast (15 维)
    ///   [25..40] D2 load_forecast (15 维)
    ///   [40..43] D3 电价 (3 个 RL 字段)
    ///   [43..46] D4 需量 (3 字段)
    ///   [46..48] D5 气象 (2 字段)
    ///   [48]     D6 dispatch_p_set (1 维，None 时填 0.0)
    ///   [49]     D7 q_realtime_margin (1 维)
    ///   [50..58] D8 season_encoding (6 维) + time_period_encoding (2 维)
    ///   [58..62] D9 safety_override (4 维，v2.14 扩展 consecutive+ratio)
    ///   [62..77] D10 load_forecast_quantiles (15 维，v2.11 新增)
    ///   [77]     D10 shock_load_probability (1 维，v2.11 新增)
    ///   [78]     D10 base_load (1 维，v2.11 新增)
    pub fn to_input_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(78);

        // [0..10] D1: 10 个标量 (不含 timestamp，含 q_realtime_margin)
        v.push(self.battery_soc as f32);
        v.push(self.pv_power as f32);
        v.push(self.load_power as f32);
        v.push(self.grid_power as f32);
        v.push(self.transformer_load as f32);
        v.push(self.battery_power as f32);
        v.push(self.voltage_phase_a as f32);
        v.push(self.voltage_phase_b as f32);
        v.push(self.voltage_phase_c as f32);
        v.push(self.q_realtime_margin as f32);  // v2.5 新增

        // [10..25] D2 pv_forecast: 15 维
        let pv = pad_or_truncate(&self.pv_forecast_15min, 15);
        v.extend(pv.iter().map(|&x| x as f32));

        // [25..40] D2 load_forecast: 15 维
        let load = pad_or_truncate(&self.load_forecast_15min, 15);
        v.extend(load.iter().map(|&x| x as f32));

        // [40..43] D3: 3 个 RL 字段
        v.push(self.current_electricity_price as f32);
        v.push(self.next_period_price as f32);
        v.push(self.price_tariff_id as f32);

        // [43..46] D4: 3 字段
        v.push(self.current_demand as f32);
        v.push(self.contract_demand as f32);
        v.push(self.peak_demand_this_month as f32);

        // [46..48] D5: 2 字段
        v.push(self.solar_irradiance as f32);
        v.push(self.temperature as f32);

        // [48] D6: dispatch_p_set (None 时 0.0)
        v.push(self.dispatch_p_set.unwrap_or(0.0) as f32);

        // [49] D7: q_realtime_margin (v2.5 新增)
        v.push(self.q_realtime_margin as f32);

        // [50..58] D8: season_encoding (6 维) + time_period_encoding (2 维)
        for &s in &self.season_encoding { v.push(s as f32); }
        for &t in &self.time_period_encoding { v.push(t as f32); }

        // [58..62] D9: safety_override (4 维，v2.14 扩展)
        v.push(if self.safety_override_active { 1.0 } else { 0.0 });
        v.push(self.safety_override_p_ref.unwrap_or(0.0) as f32);
        v.push(self.safety_override_consecutive as f32);
        v.push(self.safety_override_ratio as f32);

        // [62..77] D10 load_forecast_quantiles: 15 维 (v2.11 新增)
        let quantiles = pad_or_truncate(&self.load_forecast_quantiles, 15);
        v.extend(quantiles.iter().map(|&x| x as f32));

        // [77] D10 shock_load_probability (v2.11 新增)
        v.push(self.shock_load_probability as f32);

        // [78] D10 base_load (v2.11 新增)
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
- 执行器按下垂公式 `P_output = P_ref + k_droop × ΔV` 执行毫秒级暂态调节

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

**输入向量维度（v2.14）：** 78 维 = 10(D1) + 30(D2) + 3(D3) + 3(D4) + 2(D5) + 1(D6) + 1(D7) + 8(D8) + 4(D9) + 17(D10)。

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
- `K_MAX`：k_droop 上限（默认 50.0 kW/V）
- `λ`：超限惩罚系数（默认 1.0）

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
    const K_MAX: f64 = 50.0; // kW/V
    const LAMBDA: f64 = 1.0;
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

R_price_spread         = p_ref * delta_t * (price_current - price_average) * conversion_factor
P_battery_degradation  = beta * abs(p_ref) * delta_t / E_battery_total * 100
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
    // 电价差：当前电价相对于峰谷均价差
    let avg_price = (state.peak_price + state.valley_price) / 2.0;
    let spread = (state.current_electricity_price - avg_price) * action.p_ref * 0.001;
    let r_spread = spread * 100.0; // 缩放
    let p_deg = 100.0 * action.p_ref.abs() / 500.0 * 0.01; // 每 kW 损耗
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
  ├─ predict_quantiles()  （v2.11 分位数预测）
  └─ erfc()               （正态分布 CDF 近似）
        ↓
ProbabilisticLoadOutput
  ├─ quantiles: Vec<QuantilePrediction>  [P10, P50, P90]
  ├─ base_load (P50)
  ├─ shock_probability
  └─ confidence
        ↓
  ┌────┴────┐
  ↓         ↓
RewardCalculator  FusedSystemState
calc_demand_with_   load_forecast_quantiles
uncertainty()       (v2.11 新增字段)
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
pub struct QuantilePrediction {
    pub quantile: f32,  // 分位数（0.0 ~ 1.0）
    pub value: f32,      // 预测值 (kW)
}

pub struct ProbabilisticLoadOutput {
    pub timestamp: i64,
    pub quantiles: Vec<QuantilePrediction>,
    pub base_load: f32,           // 50% 分位数
    pub shock_probability: f64,   // 冲击负荷概率
    pub confidence: f64,
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
│   ├── model_manager.rs          # 模型管理器（full_decision_cycle 统一调度）
│   ├── mode_selector.rs          # 运行场景选择器（5 种互斥场景 + 远程/本地切换）
│   ├── lstm_model.rs             # LSTM 时序预测模型（LstmInput, LstmOutput, LstmModel）
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
│   ├── error.rs                  # 错误类型枚举（AiEngineError, thiserror）
│   └── config.rs                 # 配置结构（AiEngineConfig 及 8 个子配置）
└── tests/
    ├── ai_engine_tests.rs        # 配置默认值测试
    ├── lstm_model_tests.rs       # LSTM 模型集成测试
    ├── rl_model_tests.rs         # RL 模型集成测试
    ├── rknn_runtime_tests.rs     # RKNN Runtime 集成测试
    └── online_updater_tests.rs   # 在线微调集成测试
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

// 重新导出公共类型
pub use config::{
    ActionConstraintConfig, AiEngineConfig, FusionConfig, LstmConfig, ModeConfig,
    ModelType, NpuConfig, OnlineUpdateConfig, QuantizationType, RlAlgorithm, RlConfig,
    SceneWeights,
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
}
```

配置文件示例 (`mupc/config/ai.toml`)：

```toml
[lstm]
model_path = \"/etc/mupc/models/lstm.rknn\"
input_window_secs = 3600
output_horizon_secs = 900
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
| LstmConfig | model_path, input_window_secs, output_horizon_secs, quantization | /etc/mupc/models/lstm.rknn, 3600, 900, INT8 |
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
}
```

### 11.2 错误分类与处理策略

| 错误类别 | 错误变体 | 恢复策略 |
|----------|----------|----------|
| 模型加载 | `ModelLoadFailed`, `VersionMismatch` | 拒绝启动，记录 ERROR，触发降级 |
| 推理运行时 | `InferenceFailed`, `RknnError`, `InputShapeMismatch`, `OutputShapeMismatch` | 重试 1 次，失败后记录 ERROR，连续 3 次后触发 NPU 降级 |
| 资源状态 | `ModelNotLoaded` | 等待模型加载完成 |
| 数据异常 | `FusionFailed`, `DataSourceStale` | 按缺失数据处理策略填充，连续 10 周期后触发降级 |
| 运维操作 | `ModeSwitchFailed`, `ActionValidationFailed`, `OnlineUpdateFailed` | 记录 WARN，操作回滚 |
| 硬件异常 | `NpuOverheating` | 降频保护，连续 5 周期正常后恢复 |

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

### 13.8 关键实现文件

These are the most critical files that need to be created or significantly modified to implement this design:

- `e:\MUPC2\mupc\crates\ai-engine\src\data_fusion.rs` (new: DataFusionEngine, DataSourceAdapter trait, 5 adapter implementations, FusedSystemState with to_input_vector())
- `e:\MUPC2\mupc\crates\ai-engine\src\rl_model.rs` (refactor: replace SystemState with FusedSystemState, replace old 8-field ActionOutput with new 5-field ActionOutput, add parse_action_output, add 48-dim input support)
- `e:\MUPC2\mupc\crates\ai-engine\src\reward_calculator.rs` (new: RewardCalculator with 5 scene formulas, SceneWeights lookup)
- `e:\MUPC2\mupc\crates\ai-engine\src\action_validator.rs` (new: ActionValidator with 5 constraint rules ACT-01~05, clamp logic, ViolationRecord)
- `e:\MUPC2\mupc\crates\ai-engine\src\model_manager.rs` (refactor: add full_decision_cycle(), wire in DataFusionEngine, RewardCalculator, ActionValidator)


---

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

