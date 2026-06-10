---

# MUPC AI 引擎 - 模块设计文档

[DESIGN_APPROVED] — v2.6 农网台区参数更新

| 版本 | 日期       | 作者   | 状态 |
| ---- | ---------- | ------ | ---- |
| v2.6 | 2026-06-10 | 架构师 | 当前版本 |
| v2.5 | 2026-06-08 | 架构师 | 当前版本 |
| v2.4 | 2026-06-07 | 架构师 | 当前版本 |

**对应 PRD:** `docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md` v2.6 (`[REVIEWED: PASS]`)

---

## 目录

1. [模块架构](#1-模块架构)
2. [LSTM 模型设计](#2-lstm-模型设计)
3. [多源数据融合设计](#3-多源数据融合设计)
4. [强化学习模型设计](#4-强化学习模型设计)
5. [奖励函数计算模块](#5-奖励函数计算模块)
6. [RKNN Runtime 设计](#6-rknn-runtime-设计)
7. [ModelManager 统一调度设计](#7-modelmanager-统一调度设计)
8. [与策略引擎集成设计](#8-与策略引擎集成设计)
9. [文件结构](#9-文件结构)
10. [配置结构](#10-配置结构)
11. [错误类型](#11-错误类型)
12. [消息总线集成](#12-消息总线集成)
13. [技术决策记录](#13-技术决策记录)

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
|  | 物联平台   |---订阅------>| (48维向量)     |             |               |
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
| 强化学习模型 | `rl_model.rs` | MADDPG/PPO 多目标决策，4 维动作空间输出 |
| 奖励计算器 | `reward_calculator.rs` | 5 种场景奖励函数计算，驱动在线微调 |
| 动作约束校验 | `action_validator.rs` | 5 条约束规则校验，防止异常值危害设备 |
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
ActionOutput --> ActionValidator.validate() --> 通过--> 下发 strategy-engine
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

数据字段：dispatch_p_set (Option<f64>), dispatch_q_set (Option<f64>)。通过 gateway 事件驱动接收。缺失时两个字段均为 None，RL 决策跳过调度相关约束 (ACT-05)。

### 3.5 FusedSystemState 结构体（v2.5：29 字段）

```rust
/// 融合系统状态（7 大类，26 个 RL 字段 + 3 个辅助字段 = 29 字段，v2.5）
#[derive(Debug, Clone)]
pub struct FusedSystemState {
    // ------- D1: 实时数据 (10 RL + 1 aux = 11 字段，v2.5 新增 q_realtime_margin) -------
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
}
```

### 3.6 to_input_vector() -- 56 维序列化（v2.5）

将 FusedSystemState 转换为 RL 模型输入时，各维度按定义顺序拼接为 **56 维向量**（v2.5 从 48 维扩展）。Option 字段为 None 时填充 0.0。预测向量长度超过配置时裁剪，不足时补零。

```rust
impl FusedSystemState {
    /// 序列化为 56 维输入向量（v2.5）
    /// 布局：
    ///   [0..10]  D1 实时数据 (10 个标量，不含 timestamp，含 q_realtime_margin)
    ///   [10..25] D2 pv_forecast (15 维)
    ///   [25..40] D2 load_forecast (15 维)
    ///   [40..43] D3 电价 (3 个 RL 字段)
    ///   [43..46] D4 需量 (3 字段)
    ///   [46..48] D5 气象 (2 字段)
    ///   [48]     D6 dispatch_p_set (1 维，None 时填 0.0)
    ///   [49]     D7 q_realtime_margin (1 维)
    ///   [50..56] D7 season_encoding (6 维) + time_period_encoding (2 维)
    pub fn to_input_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(56);

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

        // [9..24] D2 pv_forecast: 15 维
        let pv = pad_or_truncate(&self.pv_forecast_15min, 15);
        v.extend(pv.iter().map(|&x| x as f32));

        // [24..39] D2 load_forecast: 15 维
        let load = pad_or_truncate(&self.load_forecast_15min, 15);
        v.extend(load.iter().map(|&x| x as f32));

        // [39..42] D3: 3 个 RL 字段
        v.push(self.current_electricity_price as f32);
        v.push(self.next_period_price as f32);
        v.push(self.price_tariff_id as f32);

        // [42..45] D4: 3 字段
        v.push(self.current_demand as f32);
        v.push(self.contract_demand as f32);
        v.push(self.peak_demand_this_month as f32);

        // [45..47] D5: 2 字段
        v.push(self.solar_irradiance as f32);
        v.push(self.temperature as f32);

        // [47] D6: dispatch_p_set (None 时 0.0)
        v.push(self.dispatch_p_set.unwrap_or(0.0) as f32);

        // [49] D7: q_realtime_margin (v2.5 新增)
        v.push(self.q_realtime_margin as f32);

        // [50..56] D7: season_encoding (6 维) + time_period_encoding (2 维)
        for &s in &self.season_encoding { v.push(s as f32); }
        for &t in &self.time_period_encoding { v.push(t as f32); }

        debug_assert_eq!(v.len(), 56, "输入向量必须为 56 维");
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
| 调度指令 | 对应字段置 None，RL 决策跳过 ACT-05 约束 | INFO |

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

RLModel 使用 MADDPG（多智能体深度确定性策略梯度）或 PPO（近端策略优化）算法，基于融合状态向量、LSTM 预测值和运行场景权重，输出 4 维动作空间的最优控制指令。

### 4.1a 分层控制架构（v2.4）

为适应台区季节性负荷"高频随机脉冲叠加"工况，采用分层控制架构：

**底层（实时控制模块）**
- 无功补偿（Q_batt）：根据电压实时闭环调节，响应时间 ms 级
- 三相不平衡：不涉及电池充放电，由实时控制核心模块独立处理
- 调节方式：查表法或 PID，不经过 AI

**上层（RL决策）**
- 有功设定值 P_batt：由 RL 策略网络输出
- 可中断负荷 Load_shedding：由 RL 策略网络输出
- 光伏限功率比例 pv_limit：由 RL 策略网络输出（v2.6 恢复，用于主动弃光）
- 动作空间：3 维 ∈ [-1,1] × [0,1] × [0,1]

**分层优点：**
- P 是 s/min 级慢变量，Q 是 ms 级快变量，单一网络同时学习两个时间尺度任务收敛困难且易振荡
- RL 专注于能量管理（光伏消纳、SOC平衡、过载预防），电压质量由底层保障
- 部署时 Q 失控风险与 RL 解耦

**动作空间对比：**

| 维度 | v2.3（4维） | v2.4~v2.5（2维） | v2.6（3维） | 说明 |
|------|------------|-----------------|------------|------|
| A1 | p_batt_set [-500,500]kW | p_batt_set [-500,500]kW | p_batt_set [-500,500]kW | 电池有功（RL控制） |
| A2 | q_batt_set [-300,300]kVar | 由实时模块闭环 | 由实时模块闭环 | 无功（实时控制，不经AI） |
| A3 | load_shedding [0,500]kW | load_shedding [0,500]kW | load_shedding [0,500]kW | 可中断负荷（RL控制） |
| A4 | pv_limit [0,1] | 由A2/Q替代 | pv_limit [0,1] | 光伏限功率（v2.6 恢复主动弃光） |

### 4.2 算法选择

| 算法 | 适用场景 | 特点 |
|------|----------|------|
| MADDPG | 多目标优化（默认） | 支持连续动作空间，经验回放，目标网络 |
| PPO | 需快速收敛时 | 信任区域限制，稳定性好，on-policy |

算法类型由 `RlConfig.algorithm` 指定，训练阶段在 x86 服务器完成，部署阶段仅执行推理。

### 4.3 完整状态空间表（7 大类，29 个字段，v2.5）

| 维度 | 字段名 | 类型 | 取值范围 | 单位 | 说明 |
|------|--------|------|----------|------|------|
| **D1-实时（v2.5新增 q_realtime_margin）** | battery_soc | f64 | [0.0, 1.0] | - | 电池荷电状态 |
| | pv_power | f64 | [-1000.0, 1000.0] | kW | 光伏出力 |
| | load_power | f64 | [-1000.0, 1000.0] | kW | 负荷功率 |
| | grid_power | f64 | [-1000.0, 1000.0] | kW | 电网交换功率 |
| | transformer_load | f64 | [0.0, 2.0] | - | 变压器负载率 |
| | battery_power | f64 | [-500.0, 500.0] | kW | 电池充放电功率 |
| | voltage_phase_a | f64 | [0.8, 1.2] | p.u. | A 相电压标幺值 |
| | voltage_phase_b | f64 | [0.8, 1.2] | p.u. | B 相电压标幺值 |
| | voltage_phase_c | f64 | [0.8, 1.2] | p.u. | C 相电压标幺值 |
| | **q_realtime_margin** | f64 | [0.0, 1.0] | - | 实时模块剩余无功容量比例（v2.5新增，0=打满，1=空闲）|
| **D2-预测** | pv_forecast_15min | Vec\<f64\> | 15~30 维 | kW | 光伏预测 |
| | load_forecast_15min | Vec\<f64\> | 15~30 维 | kW | 负荷预测 |
| **D3-电价** | current_electricity_price | f64 | [0.0, 2.0] | 元/kWh | 当前电价 |
| | next_period_price | f64 | [0.0, 2.0] | 元/kWh | 下时段电价 |
| | price_tariff_id | u8 | {0~3} | 枚举 | 谷/平/峰/尖峰 |
| **D4-需量** | current_demand | f64 | [0.0, 10000.0] | kW | 实时需量 |
| | contract_demand | f64 | [0.0, 10000.0] | kW | 合同需量 |
| | peak_demand_this_month | f64 | [0.0, 10000.0] | kW | 月最大需量 |
| **D5-气象** | solar_irradiance | f64 | [0.0, 1500.0] | W/m^2 | 光照强度 |
| | temperature | f64 | [-20.0, 60.0] | deg C | 环境温度 |
| **D6-调度** | dispatch_p_set | Option\<f64\> | [-1000.0, 1000.0] | kW | 调度有功设定 |
| | dispatch_q_set | Option\<f64\> | [-1000.0, 1000.0] | kVar | 调度无功设定 |
| **D7-季节时段（v2.5新增，8 字段）** | season_encoding | [f64; 6] | one-hot | - | 季节编码：[灌溉季, 炒茶季, 空调季, 常规季, 保留, 保留] |
| | time_period_encoding | [f64; 2] | one-hot | - | 时段编码：[白天, 夜间] |

**输入向量维度（v2.5）：** 56 维 = 19 个标量 + 2 个 Option + 2 个向量（各 15 维）+ 1 个定长数组（8 维）。

**电压感知 P/Q 协同控制策略：**

| 场景 | 电压特征 | P 控制 (p_batt_set) | Q 控制 (q_batt_set) |
|------|----------|---------------------|---------------------|
| 光伏超发 | 电压 > 1.05 p.u. | 充电 (负值) -- 吸收有功 | 感性 (负值) -- 吸收无功，抑制电压 |
| 台区季节性负荷 | 电压 < 0.95 p.u. | 放电 (正值) -- 释放有功 | 容性 (正值) -- 释放无功，补偿励磁 |
| 末端低电压 | 电压 < 0.95 p.u. | 放电 (正值) -- 仅当无功不足时 | 容性 (正值) -- 优先手段，不消耗 SOC |

### 4.4 完整动作空间表（3 维 + confidence，v2.6）

| 维度 | 字段名 | 类型 | 取值范围 | 单位 | 说明 |
|------|--------|------|----------|------|------|
| A1 | p_batt_set | f64 | [-50.0, 50.0] | kW | 电池有功设定（负=充电，正=放电） |
| A2 | load_shedding | f64 | [0.0, 60.0] | kW | 可中断负荷切除 |
| A3 | pv_limit | f64 | [0.0, 1.0] | - | 光伏限功率比例（v2.6 恢复，0=全限，1=不限） |
| - | confidence | f64 | [0.0, 1.0] | - | 决策置信度 |

> 注：q_batt_set 由实时电压调节器闭环控制，不经过 RL 动作空间。p_batt_set 范围匹配电池最大充放电功率 50kW，load_shedding 范围匹配负荷峰值 60kW。

### 4.5 ActionOutput 结构体

```rust
/// 强化学习决策输出（3 维动作 + 置信度，v2.6）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutput {
    /// A1: 电池有功功率设定值 (kW), [-50.0, 50.0], 负=充电, 正=放电
    pub p_batt_set: f64,
    /// A2: 可中断负荷切除量 (kW), [0.0, 60.0]
    pub load_shedding: f64,
    /// A3: 光伏限功率比例, [0.0, 1.0], 0=完全限功率, 1=不限功率 (v2.6 恢复)
    pub pv_limit: f64,
    /// 决策置信度 [0.0, 1.0]
    pub confidence: f64,
}
```

### 4.6 RLModel 结构体

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
    /// 输入：56 维融合状态向量（v2.5）
    /// 输出：4 维动作 [(p_batt_set, load_shedding, pv_limit, confidence)]（v2.6）
    pub async fn decide(&self, input_vector: &[f32]) -> Result<ActionOutput, AiEngineError>;

    /// 获取模型类型
    pub fn model_type(&self) -> ModelType;

    /// 获取算法类型
    pub fn algorithm(&self) -> RlAlgorithm;
}
```

### 4.7 parse_action_output()

从 RKNN Runtime 推理输出的 f32 向量解析为 ActionOutput 结构体，并在解析阶段执行 clamp 限幅。

```rust
/// 解析 RL 模型输出向量为 ActionOutput（v2.6，3 维动作）
///
/// 输出格式: [p_batt_set, load_shedding, pv_limit, confidence]
pub fn parse_action_output(raw: &[f32]) -> Option<ActionOutput> {
    if raw.len() < 4 {
        return None;
    }
    Some(ActionOutput {
        p_batt_set:   (raw[0] as f64).clamp(-50.0, 50.0),
        load_shedding:(raw[1] as f64).clamp(0.0, 60.0),
        pv_limit:     (raw[2] as f64).clamp(0.0, 1.0),
        confidence:   (raw[3] as f64).clamp(0.0, 1.0),
    })
}
```

### 4.8 ActionValidator -- 5 条约束规则

```rust
/// 动作约束校验器
pub struct ActionValidator {
    config: ActionConstraintConfig,
    /// 上一周期的动作输出（用于变化率检测）
    previous_action: Arc<RwLock<Option<ActionOutput>>>,
}
```

**5 条约束规则（ACT-01 ~ ACT-05）：**

| 规则 ID | 约束条件 | 校验逻辑 |
|---------|----------|----------|
| ACT-01 | p_batt_set 变化率 <= 50kW/s | `abs(p_batt_new - p_batt_prev) <= config.p_batt_ramp_limit_kw` |
| ACT-02 | q_batt_set 变化率 <= 30kVar/s | `abs(q_batt_new - q_batt_prev) <= config.q_batt_ramp_limit_kvar` |
| ACT-03 | sqrt(p^2 + q^2) <= S_max=500kVA | `hypot(p_batt, q_batt) <= config.max_apparent_power_kva` |
| ACT-04 | pv_limit >= 0.1 (防逆流除外) | `pv_limit >= config.pv_limit_min` (防逆流场景允许 0.0) |
| ACT-05 | 调度约束 | `abs(p_batt) <= abs(dispatch_p_set)` (仅 dispatch_p_set 不为 None 时) |

```rust
impl ActionValidator {
    /// 校验动作输出，返回 clap 后的安全动作
    pub async fn validate(
        &self,
        action: &ActionOutput,
        dispatch_p_set: Option<f64>,
        is_anti_reverse_scenario: bool,
    ) -> (ActionOutput, Vec<ViolationRecord>) {
        let mut validated = action.clone();
        let mut violations = Vec::new();

        // ACT-01: 有功变化率约束
        if let Some(ref prev) = *self.previous_action.read().await {
            let delta_p = (action.p_batt_set - prev.p_batt_set).abs();
            if delta_p > self.config.p_batt_ramp_limit_kw {
                let sign = if action.p_batt_set > prev.p_batt_set { 1.0 } else { -1.0 };
                validated.p_batt_set = prev.p_batt_set + sign * self.config.p_batt_ramp_limit_kw;
                violations.push(ViolationRecord {
                    rule: \"ACT-01\",
                    field: \"p_batt_set\",
                    original: action.p_batt_set,
                    clamped: validated.p_batt_set,
                });
            }
        }

        // ACT-02: 无功变化率约束
        if let Some(ref prev) = *self.previous_action.read().await {
            let delta_q = (action.q_batt_set - prev.q_batt_set).abs();
            if delta_q > self.config.q_batt_ramp_limit_kvar {
                let sign = if action.q_batt_set > prev.q_batt_set { 1.0 } else { -1.0 };
                validated.q_batt_set = prev.q_batt_set + sign * self.config.q_batt_ramp_limit_kvar;
                violations.push(ViolationRecord {
                    rule: \"ACT-02\",
                    field: \"q_batt_set\",
                    original: action.q_batt_set,
                    clamped: validated.q_batt_set,
                });
            }
        }

        // ACT-03: 视在功率约束
        let apparent_power = (validated.p_batt_set.powi(2) + validated.q_batt_set.powi(2)).sqrt();
        if apparent_power > self.config.max_apparent_power_kva {
            let scale = self.config.max_apparent_power_kva / apparent_power;
            validated.p_batt_set *= scale;
            validated.q_batt_set *= scale;
            violations.push(ViolationRecord {
                rule: \"ACT-03\",
                field: \"p_batt_set+q_batt_set\",
                original: apparent_power,
                clamped: self.config.max_apparent_power_kva,
            });
        }

        // ACT-04: 光伏限功率下限
        if !is_anti_reverse_scenario && validated.pv_limit < self.config.pv_limit_min {
            validated.pv_limit = self.config.pv_limit_min;
            violations.push(ViolationRecord {
                rule: \"ACT-04\",
                field: \"pv_limit\",
                original: action.pv_limit,
                clamped: validated.pv_limit,
            });
        }

        // ACT-05: 调度指令权限约束
        if let Some(dp) = dispatch_p_set {
            if validated.p_batt_set.abs() > dp.abs() {
                let sign = validated.p_batt_set.signum();
                validated.p_batt_set = sign * dp.abs();
                violations.push(ViolationRecord {
                    rule: \"ACT-05\",
                    field: \"p_batt_set\",
                    original: action.p_batt_set,
                    clamped: validated.p_batt_set,
                });
            }
        }

        *self.previous_action.write().await = Some(validated.clone());
        (validated, violations)
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

**优化目标：** 最大化光伏消纳 + 防止变压器过载 + 电池寿命保护 + 电压安全

> **v2.6 说明（分层架构原则）：**
> - AI 仅在实时模块无功耗尽时才对电压偏差负责（q_realtime_margin <= 10% + 越限连续 2 步）
> - 实时模块有裕度时，电压问题由实时模块自行处理，AI 不因"旁观"被惩罚
> - 新增自适应损耗系数 α(s) ∈ {1.0, 0.2, 3.0} 区分"常规调度"与"应急处置"的电池损耗价值差异
> - 弃光奖励增加电压前置条件：v_avg >= 1.05 p.u. 时置零
> - **v2.6 新增**：电压变化斜率惩罚 R_voltage_slope = |ΔV|，迫使 AI 平滑调节，避免电压快速变化对电网造成冲击

**v2.6 奖励公式：**

```
R_agri = w1 * R_pv_consumption          // 弃光奖励（含电压安全前置条件）
         - α(s) * w2 * P_battery_degradation   // 自适应损耗系数
         - w3 * P_transformer_overload
         - w4 * P_voltage_deviation             // 条件触发式
         - w5 * R_ramp
         - w6 * R_voltage_slope                 // 电压变化斜率惩罚（v2.6 新增）
```

**子项定义：**

```
R_pv_consumption       = 0.0                                          if v_avg >= 1.05 (电压偏高)
                        = min(P_pv_self_consume / P_pv_total, 1.0) * 100   otherwise
α(s)                   = 3.0   // SOC 极低保护：battery_soc < SOC_CRITICAL (10%)
                        = 0.2   // 电压支撑模式：q_realtime_margin <= 10% 且越限 >= 2 步
                        = 1.0   // 常规调度
P_battery_degradation  = α(s) * (|P_batt| / BATTERY_CAPACITY_KWH)²   # C-rate² × α(s)
P_transformer_overload = Quadratic(L_trafo, start=75%)                 # 见 4.5 节
P_voltage_deviation    = 0.0                                          if |V_avg - 1.0| <= 5%
                                                                                OR q_realtime_margin > 10%
                        = k_v * dev²                                   if 越限连续 >= 2 步 且 q_margin <= 10%
                          k_v = 2.0 (低电压侧), 1.0 (高电压侧)
                          dev = |V_avg - 1.0| - 5%
R_ramp                 = λ * |P_batt_t - P_batt_{t-1}| / BATTERY_CAPACITY_KWH
                          归一化到 C-rate 变化率
R_voltage_slope        = |V_avg_t - V_avg_{t-1}|                       # v2.6 新增，迫使平滑调节

其中 v_avg = (voltage_phase_a + voltage_phase_b + voltage_phase_c) / 3.0
```

**权重表（v2.6）：**

| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 光伏消纳奖励（含电压前置条件） | [0.0, 3.0] |
| w2 | 0.5 | 电池损耗惩罚（C-rate² × α(s)） | [0.0, 2.0] |
| w3 | 2.0 | 变压器过载惩罚 | [0.0, 5.0] |
| w4 | 1.0 | 电压质量惩罚（条件触发式） | [0.0, 3.0] |
| w5 | 0.5 | 功率变化率惩罚 | [0.0, 2.0] |
| w6 | 0.5 | 电压变化斜率惩罚（v2.6 新增） | [0.0, 2.0] |

**Rust 代码实现（v2.5）：**

```rust
/// SCENE-01: 台区季节性负荷模式 v2.5
fn calc_agri_v2_5(&self, state: &FusedSystemState, p_batt_set: f64, prev_p_batt: f64) -> f64 {
    let w = &self.weights.seasonal_load_management;

    // 1. 弃光奖励（含电压安全前置条件）
    let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
    let r_pv = if v_avg >= self.voltage_high_limit {
        0.0
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

    // 5. 条件触发电压惩罚
    let p_voltage = self.conditional_voltage_penalty(state);

    // 6. 变化率惩罚
    let r_ramp = w[4] * (p_batt_set - prev_p_batt).abs() / self.battery_capacity_kwh;

    w[0] * r_pv - w[1] * p_batt_deg - w[2] * p_trafo - w[3] * p_voltage - r_ramp
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

/// 条件触发电压惩罚
fn conditional_voltage_penalty(&self, state: &FusedSystemState) -> f64 {
    let v_avg = (state.voltage_phase_a + state.voltage_phase_b + state.voltage_phase_c) / 3.0;
    let dev = (v_avg - 1.0).abs();
    if dev <= 0.05 {
        self.voltage_violation_count.store(0, Ordering::Relaxed);
        return 0.0;
    }
    let count = self.voltage_violation_count.fetch_add(1, Ordering::Relaxed) + 1;
    if count < 2 || state.q_realtime_margin > self.q_margin_threshold {
        return 0.0;
    }
    let dev_excess = dev - 0.05;
    if v_avg < 1.0 {
        self.voltage_penalty_low * dev_excess * dev_excess
    } else {
        self.voltage_penalty_high * dev_excess * dev_excess
    }
}
```

### 5.4 SCENE-B1：工商业模式-自主套利 (MODE-02)

**优化目标：** 最大化峰谷电价差收益，最小化电池损耗。

**奖励公式：**

```
R_arbitrage = w1 * R_price_spread - w2 * P_battery_degradation

R_price_spread         = p_batt_set * delta_t * (price_current - price_average) * conversion_factor
P_battery_degradation  = beta * abs(p_batt_set) * delta_t / E_battery_total * 100
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
    let spread = (state.current_electricity_price - avg_price) * action.p_batt_set * 0.001;
    let r_spread = spread * 100.0; // 缩放
    let p_deg = 100.0 * action.p_batt_set.abs() / 500.0 * 0.01; // 每 kW 损耗
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
            let p_actual = action.p_batt_set;
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
            RunningMode::AgriculturalIrrigation => &self.agricultural_irrigation[..3],
            RunningMode::CommercialArbitrage => &self.commercial_arbitrage[..2],
            RunningMode::DemandControl => &self.demand_control[..2],
            RunningMode::VirtualPowerPlant => &self.virtual_power_plant[..3],
            RunningMode::UltraGreen => &self.ultra_green[..2],
        }
    }
}
```

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
| 动作输出校验 | 0.5ms | 5 条约束规则 clamp |
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

        // Step 4: 状态向量序列化（48 维）
        let input_vector = fused_state.to_input_vector();

        // Step 5: 获取场景权重
        let scene_weights = SceneWeights::lookup(&self.config.reward_weights, running_mode);

        // Step 6: RL 决策推理
        let rl_action = self.decide_rl(&input_vector).await?;

        // Step 7: 动作约束校验（5 条规则）
        let (validated, violations) = self.action_validator.validate(
            &rl_action,
            fused_state.dispatch_p_set,
            false, // is_anti_reverse_scenario
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
                    validated.p_batt_set as f32,
                    validated.q_batt_set as f32,
                    validated.load_shedding as f32,
                    validated.pv_limit as f32,
                    validated.confidence as f32,
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
│   ├── data_fusion.rs            # 多源数据融合引擎（DataSourceAdapter trait + 5 个实现）
│   ├── action_validator.rs       # 动作约束校验器（5 条约束规则 ACT-01~05）
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
default_mode = \"AgriculturalIrrigation\"
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
| ModeConfig | default_mode, persist_path | \"AgriculturalIrrigation\", \"/var/lib/mupc/current_mode\" |
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

### 13.4 ADR-004: 6 大类状态空间 + 48 维输入向量

**决策：** 状态空间分为 6 大类（D1 实时数据含三相电压, D2 预测数据, D3 电价, D4 需量, D5 气象, D6 调度指令），序列化为 48 维固定长度输入向量。

**理由：**
- 6 大类对应 6 个数据源，按类别组织便于缺失数据处理和配置管理
- 48 维固定长度简化 RKNN Runtime 输入形状校验
- 预测向量超出时截断、不足时补零，保证维度一致性
- D1 中三相电压标幺值 (voltage_phase_a/b/c) 使 AI 引擎能感知台区电压水平，执行 P/Q 协同控制

### 13.5 ADR-005: 5 条动作约束规则 + clamp 限幅

**决策：** AI 模型输出 4 维动作后，经 5 条约束规则（ACT-01~05）校验，违反约束时自动 clamp 到安全边界，并记录 WARN 日志。

**理由：**
- AI 模型输出不能直接下发给物理设备，必须经过安全校验
- 变化率约束 (ACT-01/02) 保护电池设备免受功率突变损害
- 视在功率约束 (ACT-03) 保证 P/Q 组合在逆变器功率圆内
- clamp（截断）比拒绝动作更鲁棒：拒绝动作会导致控制中断，clamp 保留有效部分

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

### Critical Files for Implementation

These are the most critical files that need to be created or significantly modified to implement this design:

- `e:\MUPC2\mupc\crates\ai-engine\src\data_fusion.rs` (new: DataFusionEngine, DataSourceAdapter trait, 5 adapter implementations, FusedSystemState with to_input_vector())
- `e:\MUPC2\mupc\crates\ai-engine\src\rl_model.rs` (refactor: replace SystemState with FusedSystemState, replace old 8-field ActionOutput with new 5-field ActionOutput, add parse_action_output, add 48-dim input support)
- `e:\MUPC2\mupc\crates\ai-engine\src\reward_calculator.rs` (new: RewardCalculator with 5 scene formulas, SceneWeights lookup)
- `e:\MUPC2\mupc\crates\ai-engine\src\action_validator.rs` (new: ActionValidator with 5 constraint rules ACT-01~05, clamp logic, ViolationRecord)
- `e:\MUPC2\mupc\crates\ai-engine\src\model_manager.rs` (refactor: add full_decision_cycle(), wire in DataFusionEngine, RewardCalculator, ActionValidator)


---

## v2.3 修订记录 (2026-06-07)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | SCENE-01 恢复电压质量惩罚 | 5.3 奖励函数 | 增加 w4 * P_voltage_deviation 项；三相电压幅值通过 P/Q 协同控制可主动调节，三相不平衡维持实时控制核心独立处理 |
| 2 | SceneWeights 表更新 | 5.8 | MODE-01 新增 w4=1.0（电压质量）列 |
| 3 | 版本号更新 | 文档头部 | v2.2 → v2.3 |

**修订依据：** MODE-01 农网灌溉场景中，高电压（光伏超发）和低电压（灌溉抽水导致）均需 AI 引擎主动干预。
原 v2.1 以"三相不平衡不涉及电池充放"为由完全移除电压奖励属误判——电压幅值偏差与电池充放电、无功输出密切相关，
AI 引擎可通过 p_batt/q_batt 协同控制主动调节。v2.3 仅恢复电压幅值惩罚，三相不平衡治理维持实时控制核心独立处理。

## v2.4 修订记录 (2026-06-07)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 分层控制架构 | 1.1/4.1a/5.3 | 动作空间4维→2维；Q控制交实时控制核心，RL仅输出P_batt+Load_shedding |
| 2 | 电压死区机制 | 5.3 | ±5%死区，越限连续2步触发惩罚 |
| 3 | 功率变化率惩罚R_ramp | 5.3 | R_ramp = λ·|ΔP_batt| |
| 4 | SceneWeights表更新 | 5.8 | MODE-01新增w5列 |
| 5 | 版本号更新 | 文档头部 | v2.3 → v2.4 |

**修订依据：** 300kW错峰启停产生高频随机脉冲，Q属ms级快变量应由实时控制闭环；AI专注能量管理；电压死区防高频脉冲过度响应；变化率惩罚延长电池寿命。

## v2.5 修订记录 (2026-06-08)

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

## v2.6 修订记录 (2026-06-10)

| 序号 | 修订项 | 修订位置 | 说明 |
|------|--------|----------|------|
| 1 | 农网台区参数更新 | config.rs / reward_calculator.rs / data_fusion.rs / strategy-engine | 变压器容量 500kVA→200kVA，电池容量 200kWh→100kWh，最大充放电功率 100kW→50kW，合同需量 500kW→200kW |
| 2 | 版本号更新 | 文档头部 | v2.5 → v2.6 |

**修订依据：** 农网台区新规格落地：变压器 200kVA、光伏 150kW、储能 50kW/100kWh、居民负荷 60kW、农业冲击负荷最高 120kW。代码默认值已同步更新。

