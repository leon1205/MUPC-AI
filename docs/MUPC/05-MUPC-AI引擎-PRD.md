# MUPC AI 优化引擎 - 模块产品需求文档（统一版 v2.15）

> **版本：** v2.15 | **状态：** [REVIEWED: PASS] | **更新日期：** 2026-06-17

### 变更记录

| 版本 | 日期 | 作者 | 变更说明 | 评审状态 |
|------|------|------|----------|----------|
| v2.15 | 2026-06-17 | 需求分析师 | 动作空间精简：5维→2维（移除load_shedding/pv_limit/confidence），下沉至策略引擎 | [REVIEWED: PASS] |
| v2.14 | 2026-06-15 | - | SafetyOverride 奖励函数重构、FusedSystemState 78维统一 | [REVIEWED: PASS] |
| v2.13 | 2026-06-14 | - | Sigmoid P-Q平滑化、Welford奖励归一化、confidence字段 | [REVIEWED: PASS] |

---

## 1. 产品概述

### 1.1 产品定位

AI 优化引擎是 MUPC 通信管理模块（"大脑"）的核心智能决策组件，负责根据实时运行数据和外部信息进行时序预测、场景识别与强化学习决策，生成最优控制指令下发给实时控制模块（"小脑"）执行。

AI 引擎遵循 **"AI 优先，本地兜底"** 策略：正常时 AI 引擎主导决策，AI 失效时自动、无缝地降级至本地策略引擎（strategy-engine）接管控制。

### 1.2 核心职责

| 职责 | 说明 |
|------|------|
| 时序预测 | LSTM 模型预测光伏出力 / 负荷变化趋势（含分位数预测） |
| 预设运行场景选择 | 5 种预设运行场景，支持远程控制/本地选择，同一时刻仅 1 种互斥运行 |
| 多目标决策 | MADDPG/PPO 强化学习模型，2 维动作空间，5 种奖励函数 |
| 数据融合 | 融合电气量、电池数据、电价、气象、调度指令等 10 大类 **78 维**状态 |
| NPU 推理 | 在 RK3588 NPU 上执行 INT8 量化模型推理 |
| 在线微调 | 基于新数据持续更新模型权重（影子模型验证 + 渐进式切换） |
| 安全校验 | 输出指令经 ActionValidator 约束校验后再下发 |

### 1.3 目标平台

| 项目 | 要求 |
|------|------|
| 硬件 | RK3588 (NPU: 6 TOPS) |
| 操作系统 | openEuler 22.03+ |
| 编程语言 | Rust >= 1.75 |
| 异步运行时 | Tokio |
| 推理框架 | RKNN Runtime (INT8 量化) |
| 模型格式 | ONNX（训练）→ .rknn（部署） |
| 训练平台 | x86 服务器（PyTorch + rknn-toolkit2） |

### 1.4 核心价值

| 价值 | 说明 | 量化目标 |
|------|------|----------|
| 预设场景互斥 | 5 种预设运行场景，调度主站远程/本地选择，互斥运行保证确定性 | 场景切换延迟 < 2s（远程）、< 1s（本地） |
| 多目标优化 | 根据不同场景动态平衡经济收益、设备寿命、电网安全 | 综合目标函数值提升 >= 20%（相比固定策略） |
| 全维度决策 | 覆盖有功、无功两个控制维度 | 无功调节覆盖率 100% |
| 可量化训练 | 每个场景有明确奖励公式，模型训练目标可量化 | 奖励函数可计算误差 < 1% |

### 1.5 用户角色

| 角色 | 描述 | 权限范围 |
|------|------|----------|
| **调度主站** | 上级调度中心，通过 IEC 104/IEC 61850 远程指挥台区运行 | 远程切换运行场景（最高优先级）、下发调度指令 |
| **AI 运维人员** | 负责 AI 模型训练、部署、监控和维护的技术人员 | 模型版本管理、推理参数配置、在线微调启停、Web UI 决策可视化（预测曲线、决策逻辑、实时奖励，详见模块08-PRD 第6章） |
| **策略管理员** | 负责本地选择运行场景和调整权重的电力系统运维人员 | 本地场景切换、场景权重配置、奖励函数参数调整、Web UI 决策可视化（预测曲线、决策逻辑、实时奖励，详见模块08-PRD 第6章） |
| **本地运维人员** | 负责 MUPC 装置日常运维的操作人员 | 查看运行场景/AI 决策日志、接收告警、强制降级至本地策略 |

**权限优先级规则：**
- 调度主站远程切换运行场景优先级最高（全局电网调度视角）
- 策略管理员本地切换运行场景，冲突时远程指令优先
- AI 运维人员可配置所有模型参数和训练参数，但不能切换运行模式
- 本地运维人员可强制降级至本地策略模式，覆盖 AI 输出

### 1.6 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI 优化引擎 (ai-engine)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ LSTM 预测    │    │MADDPG/PPO   │    │ 在线微调    │        │
│  │ (时序预测)   │    │ (决策优化)   │    │ (持续学习)   │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                       ┌────▼────┐                              │
│                       │ 模型    │                              │
│                       │ 管理器  │                              │
│                       └────┬────┘                              │
│                   ┌────────┼────────┐                         │
│                   ▼        ▼        ▼                          │
│  ┌──────────┐ ┌────────┐ ┌──────────────┐                    │
│  │ 数据融合 │ │ 模式   │ │ 奖励函数     │                    │
│  │ 引擎     │ │ 选择器 │ │ 计算器       │                    │
│  └──────────┘ └────────┘ └──────────────┘                    │
│                             │                                  │
│                    ┌────────▼────────┐                        │
│                    │ 动作约束校验器   │                        │
│                    └────────┬────────┘                        │
│                             │                                  │
│                    ┌────────▼────────┐                        │
│                    │  RKNN Runtime   │ ←→ librknnrt.so (FFI) │
│                    │  (NPU 推理)     │                        │
│                    └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│  strategy-engine│    │  data-processing     │
│ (指令校验 + 兜底)│    │ (数据管道 + MQTT)    │
└─────────────────┘    └──────────────────────┘
```

### 1.7 数据流

```
历史数据 → LSTMModel.predict() → 光伏/负荷预测值 → 供 RL 模型使用
                                                         ↓
远程指令/本地选择 → ModeSelector → 运行模式 → 权重映射 + 奖励函数选择
                                                         ↓
融合数据 + 预测值 + 运行模式 → RLModel.decide() → ActionOutput
                                                         ↓
ActionOutput → ActionValidator (5条约束规则校验) → strategy-engine
                                                         ↓
新数据积累 → OnlineUpdater.update() → 模型权重增量更新 → 保存
```

---

## 2. 核心业务流程图

```mermaid
flowchart TD
    subgraph 数据源层
        A[intercore 实时数据] --> F[DataFusionEngine]
        B[LSTM 预测数据] --> F
        C[气象 API] --> F
        D[物联平台 电价] --> F
        E[gateway 调度指令] --> F
    end

    F --> G[FusedSystemState 78维]
    G --> H[RLModel.decide]
    B --> H
    
    H --> I[ActionOutput 2维]
    I --> J[ActionValidator 约束校验]
    J --> K[strategy-engine]
    
    K --> L[实时控制模块 执行]
    L --> M[反馈]
    M --> F
    
    N[ModeSelector] --> O[RunningMode 1-5]
    O --> P[权重映射 + 奖励函数]
    P --> H
```

---

## 3. LSTM 时序预测

### 3.1 功能概述

LSTM（Long Short-Term Memory）时序预测模型负责预测未来一段时间内的光伏出力和负荷功率，为强化学习决策模型提供前瞻性输入。模型架构支持 LSTM 作为主模型，TCN（Temporal Convolutional Network）作为备选方案，两者均通过 ONNX 格式导出并部署为 .rknn。

负荷预测需区分基荷、可调负荷、冲击负荷（如灌溉水泵启动），对冲击负荷进行概率预测（输出概率分布而非点估计）。

### 3.2 核心需求

| 需求 | 说明 |
|------|------|
| 预测范围 | 未来 15-30 分钟（默认 15 分钟，可配置扩展至 30 分钟），每分钟一个采样点 |
| 输入数据 | 历史光伏出力、历史负荷功率、气象数据（光照、温度） |
| 模型格式 | ONNX（训练）→ INT8 量化后部署为 .rknn |
| 部署方式 | RKNN Runtime 在 NPU 上执行推理 |
| 分位数预测 | 输出 P10/P50/P90 分位数（v2.11） |
| 冲击负荷概率 | 基于 P90-P50 差值法计算（v2.11） |

### 3.3 接口定义

```rust
/// 模型输入
pub struct ModelInput {
    pub battery_soc: f64,
    pub pv_power: f64,
    pub load_power: f64,
    pub grid_power: f64,
    pub timestamp: i64,
}

/// 模型输出（预测）
pub struct LstmOutput {
    pub pv_forecast: Vec<f32>,              // 光伏预测（15维）
    pub load_forecast: Vec<f32>,             // 负荷预测（15维）
    pub load_forecast_quantiles: Vec<f32>,   // 分位数预测（15维，v2.11）
    pub shock_load_probability: f32,         // 冲击负荷概率（v2.11）
    pub base_load: f32,                      // 基础负荷（v2.11）
    pub confidence: f64,
}
```

### 3.4 预测精度要求

| 指标 | 要求 | 测量方法 |
|------|------|----------|
| 光伏预测 MAPE | <= 10%（15 分钟预测范围） | 回测验证 |
| 负荷预测 MAPE | <= 15%（15 分钟预测范围） | 回测验证 |

### 3.5 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| LSTM-01 | LSTM 模型加载成功 | 单元测试 |
| LSTM-02 | LSTM 预测延迟 < 1s | 性能测试 |
| LSTM-03 | ONNX 模型格式正确 | 模型验证 |
| LSTM-04 | INT8 量化后模型大小 <= 5MB（.rknn 文件）| 模型文件验证 |
| LSTM-05 | 预测输出向量长度可配置（默认 15，可配置扩展至 30）| 单元测试 |
| PLF-01 | LSTM 可输出多分位数预测（10%、50%、90%）| P0 | 单元测试 |
| PLF-02 | 冲击负荷概率计算正确（P90 - P50 差值法）| P0 | 单元测试 |
| PLF-03 | 分位数预测延迟 <= 1s | P0 | 集成测试 |
| PLF-04 | 需量控制奖励函数考虑冲击负荷概率 | P1 | 单元测试 |
| PLF-05 | 协变量（温度、日期类型）正确传入 | P1 | 集成测试 |
| PLF-06 | 概率预测在测试集上 P90 分位数误差 < 15% | P1 | 离线评估 |
| PLF-07 | FusedSystemState 正确存储分位数预测结果 | P0 | 集成测试 |

---

## 4. 多源数据融合

### 4.1 功能概述

DataFusionEngine 负责周期性（默认 1Hz，可配置 1s~60s）从多个数据源采集数据，融合为统一的 `FusedSystemState`，供场景分类器和 RL 决策器使用。

### 4.2 数据源映射

| 融合字段 | 数据来源 | 获取方式 | 更新频率 |
|----------|----------|----------|----------|
| 实时电气量 | intercore 模块 | 核间 TCP 通信 | 1 Hz |
| 电池数据 | intercore 模块（BMS） | 核间 TCP 通信 | 1 Hz |
| 电价 | data-processing / MQTT 北向 | 定时拉取 + 订阅推送 | 15 分钟 |
| 气象 | data-processing / MQTT 北向 | 定时拉取 | 15 分钟 |
| 调度指令 | gateway (IEC 104 / IEC 61850) | 事件驱动 | 事件触发 |

### 4.3 消息总线集成（数据融合内部）

数据融合引擎通过消息总线订阅外部数据源，发布融合结果：

| Topic | 发布者 | 订阅者 | 数据格式 | 频率 |
|-------|--------|--------|----------|------|
| `ai/fused_state` | DataFusionEngine | RLModel, SceneClassifier | FusedSystemState JSON | 1Hz |
| `price/real_time` | data-processing | DataFusionEngine | ElectricityPrice JSON | 15min / 事件 |
| `weather/forecast` | data-processing | DataFusionEngine | WeatherData JSON | 15min |
| `demand/current` | data-processing | DataFusionEngine | DemandData JSON | 1Hz |

### 4.4 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| FUSION-01 | 融合周期固定 1 秒，支持运行时配置（范围 1s~60s） | 配置验证 + API 测试 |
| FUSION-02 | 融合输出数据包含全部 10 个大类、78 个状态字段 | 单元测试 |
| FUSION-03 | 每个字段的值类型和取值范围符合定义 | 单元测试 |
| FUSION-04 | 数据源异常不阻塞融合流程，缺失字段取上一周期值填充 | 集成测试 |
| FUSION-05 | 数据源连续 3 周期无更新时 WARN，连续 10 周期 ERROR 并通知 | 集成测试 |
| FUSION-06 | 融合数据写入共享内存缓冲区，读写锁冲突等待时间 < 1ms | 性能测试 |
| FUSION-07 | 融合输出带 UTC 时间戳，精度到毫秒 | 单元测试 |
| FUSION-08 | 每个数据源记录最后一次成功获取的时间戳和状态码 | 单元测试 |
| FUSION-09 | 数据源连续 3 周期无更新时，产生 WARN 级别告警 | 集成测试 |
| FUSION-10 | 数据源健康状态通过 Web UI 实时展示（绿色=正常，黄色=延迟，红色=断连）| UI 集成测试 |

### 4.5 SafetyOverride 帧类型接口规范（v2.10）

核间 TCP 通信定义 `FrameType::SafetyOverride = 0x0040` 帧类型（hex 值 0x0040），供实时控制模块在极端工况下临时覆盖 AI 有功指令。

**SafetyOverridePayload 结构体：**

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| trigger_reason | String | - | 触发原因：voltage_violation / q_exhausted / emergency / other |
| override_p_ref | f64 | kW | 强制放电功率（实时模块计算的代际参考值） |
| duration_ms | u32 | ms | 覆盖持续时间 |
| recovery_condition | String | - | 恢复条件：voltage_recovered / q_margin_available / manual_reset |

**数据流：**
1. 实时控制模块检测到极端工况（电压越限 + q_realtime_margin <= 10%）
2. 发送 SafetyOverride 帧（0x0040）至 AI 引擎
3. AI 引擎解析帧，设置 FusedSystemState.safety_override_* 字段
4. 奖励函数计算 R_safety_override 惩罚
5. 恢复条件满足后，实时模块发送正常 DataUpload 帧清除覆盖状态

---

## 5. 预设运行场景与互斥模式选择

### 5.1 功能概述

系统内置 5 种预设运行场景（MODE-01 ~ MODE-05），由**调度主站远程控制**或**策略管理员本地选择**来切换。同一时刻仅 1 种场景运行（互斥）。场景确定后，对应的奖励函数与权重映射立即生效。

### 5.2 5 种预设运行场景

| 场景 ID | 场景名称 | 优化目标 | 适用工况 |
|---------|----------|----------|----------|
| MODE-01 | 台区季节性负荷 | 最大化光伏消纳 + 防止变压器过载 + 电池寿命保护 | 台区季节性负荷（夏季灌溉/炒茶/冬季空调等） |
| MODE-02 | 自主套利 | 最大化峰谷电价差收益 + 最小化电池损耗 | 工商业台区有分时电价 |
| MODE-03 | 需量控制 | 减免需量罚金 + 最小化舒适度损失 | 工商业台区需量接近合同值 |
| MODE-04 | 虚拟电厂 | 最大化辅助服务收益 + 响应精度 | 已注册 VPP 服务且有调度指令 |
| MODE-05 | 极致绿色 | 最大化绿电消纳比例 + 最小化碳排放 | 绿色电力消纳目标 |

> **注意：** MODE-01~05 与第 7 章 5 种奖励函数一一对应。奖励函数公式和权重映射表完全保持不变。
>
> **电能质量要求：** 无功(Q)由实时控制核心模块根据电压自行调节，无需AI控制；AI仅输出有功设定值。

### 5.3 场景选择方式

| 方式 | 通道/界面 | 说明 | 性能要求 |
|------|----------|------|----------|
| 远程 IEC 104 | 信息体地址 0x4001，命令值 1~5 | 通过选择-执行模式切换场景 | < 2s |
| 远程 IEC 61850 | 逻辑节点 GGIO1.SPCSO1，ctlVal 1~5 | 通过 SetDataValues 映射场景 ID | < 2s |
| 本地 Web UI | 仪表盘 → AI 控制面板 → 运行场景选择器 | 单选下拉框，确认对话框后即时生效 | < 1s |
| 配置文件 | `mupc/config/ai.toml` → `[mode] default_scene` | 系统启动时加载的默认场景 | 启动时 |

**优先级：** 调度主站远程指令 > 策略管理员本地选择（冲突时远程优先）

### 5.4 互斥运行机制

| 规则 | 说明 |
|------|------|
| 状态互斥 | 由 `ModeSelector`（tokio::sync::Mutex）保护，任何时刻仅 1 个 `RunningMode` 生效 |
| 切换原子性 | 场景切换为原子操作，锁定窗口 < 100ms，期间 AI 决策暂停 1 周期 |
| 并发控制 | 多来源同时切换时，以时间戳最晚的指令为准（Last-Write-Wins） |
| 切换通知 | 通过消息总线 topic `ai/mode_switch` 广播切换事件 |

### 5.5 场景切换平滑过渡（v2.10）

场景切换时奖励函数权重线性插值过渡，避免策略震荡：

```rust
pub struct SmoothSceneTransition {
    transition_steps: usize,              // 过渡步数，默认 10
    current_weights: Option<Vec<f32>>,    // 旧场景权重
    target_weights: Option<Vec<f32>>,     // 目标场景权重
    step_counter: usize,
}
```

每步权重线性插值：`α = step_counter / transition_steps`，`weights = (1-α)*old + α*target`

**验收标准：**

| ID | 标准 | 验证方法 |
|----|------|----------|
| CC1 | 场景切换时触发平滑过渡，过渡步数可配置（默认10步）| 单元测试 |
| CC2 | 每步权重线性插值，确保最终权重与目标一致 | 数学验证 |
| CC3 | 过渡期间控制指令无突变（梯度 < 5%）| 集成测试 |
| CC4 | 过渡完成后自动停止插值，返回目标权重 | 单元测试 |

### 5.6 接口定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RunningMode {
    SeasonalLoadManagement = 1,   // MODE-01
    CommercialArbitrage = 2,      // MODE-02
    DemandControl = 3,            // MODE-03
    VirtualPowerPlant = 4,        // MODE-04
    UltraGreen = 5,               // MODE-05
}

pub enum SwitchSource {
    RemoteDispatch { protocol: String, address: String },
    LocalWeb { username: String },
    LocalConfig,
}

/// 模式切换事件（通过消息总线广播）
pub struct ModeSwitchEvent {
    pub previous: RunningMode,
    pub current: RunningMode,
    pub source: SwitchSource,
    pub timestamp: i64,
}
```

### 5.7 边界条件与异常处理

| 异常场景 | 检测条件 | 处理措施 |
|----------|----------|----------|
| 系统启动时无模式配置 | 配置文件缺少 `[mode]` 段 | 默认 MODE-01，记录 INFO |
| 远程与本地并发冲突 | 两个来源同时发送切换指令 | 远程优先，本地操作被拒绝 |
| AI 未就绪时收到切换 | ai_ready = false | 暂存指令，就绪后执行，暂存超时 30s 丢弃 |
| 切换到当前已激活模式 | new_mode == current_mode | 幂等处理，不触发切换 |
| 无效模式 ID | 指令值超出 1~5 范围 | 拒绝执行，返回 400 错误 |
| 模式切换过程中断电 | 持久化文件写入中断 | 重启时从持久化文件恢复，文件损坏则回退 MODE-01 |

### 5.8 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| MODE-01 | 系统启动后加载配置的默认场景，缺失时默认 MODE-01 | 启动测试 |
| MODE-02 | 远程切换 < 2s，本地切换 < 1s | 集成测试 |
| MODE-03 | 同一时刻仅 1 个场景生效，并发时最后到达的指令生效 | 并发测试 |
| MODE-04 | 场景切换时自动更新权重和奖励函数 | 集成测试 |
| MODE-05 | 切换锁定窗口 < 100ms | 性能测试 |
| MODE-06 | 所有切换事件写入审计日志 | 日志验证 |
| MODE-07 | PUT /api/v1/mode 需要 Operator+ 角色鉴权 | API 测试 |
| MODE-08 | 远程指令冲突时调度主站优先，本地操作返回"远程指令优先"提示 | 集成测试 |

---

## 6. 强化学习多目标决策

### 6.1 功能概述

RLModel 使用 MADDPG 或 PPO 算法，基于融合状态、LSTM 预测值和场景标签，输出 2 维动作空间的最优控制指令。

**电压感知 P/Q 协同控制（v2.2）：** AI 引擎感知台区电压水平，在以下场景执行有功/无功协调控制：

| 场景 | 电压特征 | P 控制 | Q 控制 | 物理机理 |
|------|----------|--------|--------|----------|
| 光伏超发（中午） | 电压升高（>1.05 p.u.） | 充电消纳多余光伏 | 吸收感性无功抑制抬升 | 吸收无功 = 降低电压 |
| 农网灌溉（抽水） | 电压降低（<0.95 p.u.） | 放电补充能量缺口 | 释放容性无功补偿励磁 | 减少线路无功流 → 降低压降 |
| 末端低电压（夜间） | 电压降低（<0.95 p.u.） | 放电（仅当 Q 不足时） | 释放容性无功（优先） | 就地补偿减少传输损耗 |

**v2.2 架构说明：** 三相电压幅值（D1 实时数据）与 v2.1 移除的电能质量 D5 是不同用途：
- v2.1 移除的 D5：三相不平衡度 + 频率 → 电能质量监测，由实时控制核心独立处理
- v2.2 新增到 D1：三相电压幅值 → 过/低电压检测 → P/Q 控制策略，是 AI 引擎决策的必要输入

### 6.2 状态空间定义（10 大类，78 维，v2.14）

⚠️ **[⚠️待确认冲突]** PRD 历史版本标题写"59 维"为错误，实际为 **78 维**

| 类别 | 字段名 | 数据类型 | 取值范围 | 单位 | 说明 | 来源 |
|------|--------|----------|----------|------|------|------|
| **D1-实时数据** | battery_soc | f64 | [0.0, 1.0] | - | 电池荷电状态 | intercore |
| | pv_power | f64 | [-1000.0, 1000.0] | kW | 光伏出力（正值=发电）| intercore |
| | load_power | f64 | [-1000.0, 1000.0] | kW | 负荷功率（正值=用电）| intercore |
| | grid_power | f64 | [-1000.0, 1000.0] | kW | 电网交换功率（正值=购电）| intercore |
| | transformer_load | f64 | [0.0, 2.0] | - | 变压器负载率 | intercore |
| | battery_power | f64 | [-500.0, 500.0] | kW | 电池当前充放电功率 | intercore |
| | voltage_phase_a | f64 | [0.8, 1.2] | p.u. | A 相电压标幺值（用于过/低电压检测，指导 P/Q 控制）| intercore |
| | voltage_phase_b | f64 | [0.8, 1.2] | p.u. | B 相电压标幺值 | intercore |
| | voltage_phase_c | f64 | [0.8, 1.2] | p.u. | C 相电压标幺值 | intercore |
| | q_realtime_margin | f64 | [0.0, 1.0] | - | 实时模块剩余无功容量比例（0=打满，1=空闲，v2.5 新增）| intercore |
| **D2-预测数据** | pv_forecast_15min | Vec\<f64\>(15) | [-1000.0, 1000.0] | kW | 未来 15 分钟光伏预测 | LSTM |
| | load_forecast_15min | Vec\<f64\>(15) | [-1000.0, 1000.0] | kW | 未来 15 分钟负荷预测 | LSTM |
| **D3-电价** | current_electricity_price | f64 | [0.0, 2.0] | 元/kWh | 当前实时电价 | 物联平台 |
| | next_period_price | f64 | [0.0, 2.0] | 元/kWh | 下一时段电价 | 物联平台 |
| | price_tariff_id | u8 | {0=谷,1=平,2=峰,3=尖峰} | 枚举 | 分时电价时段标识 | 物联平台 |
| | peak_price | f64 | [0.0, 2.0] | 元/kWh | 峰值电价（辅助）| 配置 |
| | valley_price | f64 | [0.0, 2.0] | 元/kWh | 谷值电价（辅助）| 配置 |
| **D4-需量** | current_demand | f64 | [0.0, 10000.0] | kW | 当前实际需量 | intercore |
| | contract_demand | f64 | [0.0, 10000.0] | kW | 需量合同值 | 配置 |
| | peak_demand_this_month | f64 | [0.0, 10000.0] | kW | 本月最大需量 | data-processing |
| **D5-气象** | solar_irradiance | f64 | [0.0, 1500.0] | W/m² | 当前光照强度 | 气象 API |
| | temperature | f64 | [-20.0, 60.0] | °C | 环境温度 | 气象 API |
| **D6-调度** | dispatch_p_set | Option\<f64\> | [-1000.0, 1000.0] | kW | 调度主站下发的有功设定值 | gateway |
| | dispatch_q_set | Option\<f64\> | [-1000.0, 1000.0] | kVar | 调度主站下发的无功设定值 | gateway |
| **D7-实时模块** | q_realtime_margin | f64 | [0.0, 1.0] | - | 实时模块剩余无功容量比例（0=打满，1=空闲）| intercore |
| **D8-季节时段** | season_encoding | [f64; 6] | one-hot | - | 季节编码：[灌溉季, 炒茶季, 空调季, 常规季, 保留, 保留] | data-processing |
| | time_period_encoding | [f64; 2] | one-hot | - | 时段编码：[白天, 夜间] | data-processing |
| **D9-安全覆盖** | safety_override_active | bool | {true, false} | - | 安全覆盖激活标志，true=实时模块正在覆盖 AI 有功指令（v2.10 新增）| intercore |
| | safety_override_reason | Option\<String\> | - | - | 触发原因（voltage_violation/q_exhausted/emergency，仅 active=true 时有效）| intercore |
| | safety_override_p_ref | Option\<f64\> | [-50.0, 50.0] | kW | 安全覆盖强制放电功率（仅 active=true 时有效）| intercore |
| | safety_override_consecutive | u32 | [0, ∞) | - | 连续触发次数（v2.14 新增）| intercore |
| | safety_override_ratio | f64 | [0.0, 1.0] | - | 滑动窗口内覆盖比例（v2.14 新增）| intercore |
| **D10-概率负荷** | load_forecast_quantiles | Vec\<f64\>(15) | [0.0, 10000.0] | kW | 分位数负荷预测（P10/P50/P90...，v2.11 新增）| LSTM |
| | shock_load_probability | f64 | [0.0, 1.0] | - | 冲击负荷发生概率（v2.11 新增）| LSTM |
| | base_load | f64 | [0.0, 10000.0] | kW | 基础负荷，50% 分位数（v2.11 新增）| LSTM |

**总维度：** D1(10) + D2(30) + D3(5) + D4(3) + D5(2) + D6(2) + D7(1) + D8(8) + D9(4) + D10(17) = **78 维**。

> **v2.14 说明：** D9 新增 `safety_override_consecutive` 和 `safety_override_ratio`，用于精细化 SafetyOverride 惩罚计算。D9 从 2 维扩展至 4 维，输入向量从 76 维扩展至 78 维。
>
> **v2.11 说明：** D10 新增分位数负荷预测，支撑冲击负荷预备度奖励计算。输入向量从 61 维扩展至 76 维。
>
> **v2.10 说明：** D9 新增安全覆盖状态（3 维），AI 引擎感知实时控制模块临时覆盖事件。输入向量从 56 维扩展至 59 维，RL 模型文件需重新训练或填充默认值向后兼容。
>
> **v2.5 说明：** D1 新增 `q_realtime_margin`，使 AI 引擎感知实时模块无功裕度边界；D8 新增季节/时段编码，用于季节性负荷模式识别。输入向量从 48 维扩展至 56 维。
>
> **历史说明：** PRD v2.10/v2.11 中 59 维的描述不准确，实际应为 61 维（v2.10）和 76 维（v2.11）。

序列化为推理输入向量时，各维度按定义顺序拼接。

### 6.3 动作空间定义（2 维，v2.15）

| 维度 | 字段名 | 数据类型 | 取值范围 | 单位 | 说明 | 分发路径 |
|------|--------|----------|----------|------|------|----------|
| A1 | p_ref | f64 | [-50.0, 50.0] | kW | 有功基准点（负值=充电，正值=放电）| 核间→实时控制模块 |
| A2 | k_droop | f64 | [0.0, 30.0] | kW/V | 电压-有功下垂系数，范围由实时控制模块提供 | 核间→实时控制模块 |

> **v2.15 精简说明：** 原 A3(load_shedding)、A4(pv_limit)、A5(confidence) 已从动作空间移除：
> - **load_shedding（可中断负荷切除）**：属于南向设备直控，由策略引擎的需量控制策略独立执行，不作为 AI 引擎输出维度
> - **pv_limit（光伏限功率）**：属于南向设备直控，由策略引擎的防逆流策略独立执行，不作为 AI 引擎输出维度
> - **confidence（决策置信度）**：属于 ActionOutput 元数据而非动作维度，已移至 `ModelOutput`，由 ActionValidator 校验后注入，不参与 AI 决策

**下垂控制公式：** `P_output = P_ref + k_droop × ΔV`

其中：
- `P_output`：执行器最终输出的有功功率设定值
- `P_ref`：AI 输出的有功基准点
- `k_droop`：AI 输出的下垂系数（电压-有功下垂系数）
- `ΔV`：电压偏差 = V_actual - V_target，单位 V

**k_droop 物理含义：** 电压每升高 1V，输出功率增加 k_droop kW（充电方向）；电压每降低 1V，输出功率减少 k_droop kW（放电方向）。

### 6.4 双参数动作空间（v2.7）

动作空间从单参数升级为双参数（P_ref + k_droop），实现时间尺度解耦：
- **AI 负责稳态全局优化**（P_ref）
- **执行器负责毫秒级暂态调节**（k_droop × ΔV）

**通信中断降级规则：**
- 执行器保持最后收到的 P_ref 和 k_droop，继续下垂控制，保障本质安全不停机
- P_ref 和 k_droop 同时为 NaN 时，触发 AI 降级

**降级原则：** 通信中断时保持最后有效的 P_ref 和 k_droop，不主动归零。继续按下垂公式 `P_output = P_ref + k_droop × ΔV` 计算，保障基础安全不停机。

### 6.5 动作约束规则

| 规则 ID | 约束条件 | 说明 |
|---------|----------|------|
| ACT-01 | p_ref 变化率 <= 50 kW/步 | 防止电池功率突变 |
| ACT-02 | k_droop 变化率 <= 10 kW/V/步 | 防止下垂系数突变 |
| ACT-03 | p_ref ∈ [p_ref_min, p_ref_max] | 有功基准点范围约束 |
| ACT-04 | k_droop ∈ [k_droop_min, k_droop_max] | 下垂系数范围约束 |
| ACT-05 | dispatch_p_set 有效时，\|p_ref\| <= \|dispatch_p_set\| | 调度指令权限约束 |

> **v2.15 说明：** load_shedding、pv_limit 的约束校验（原 ACT-05、ACT-06）已下沉至策略引擎独立执行，不再纳入 AI 动作约束规则。confidence 校验移至 ActionValidator → ModelOutput 流程。

### 6.6 接口定义

```rust
#[derive(Debug, Clone)]
pub struct ActionOutput {
    pub p_ref: f64,           // 有功功率基准点 (kW), 负=充电, 正=放电
    pub k_droop: f64,         // 电压-有功下垂系数 (kW/V)
}
```

> **v2.15 说明：** v2.15 从 ActionOutput 中移除 `load_shedding`（下沉至策略引擎需量控制）、`pv_limit`（下沉至策略引擎防逆流）、`confidence`（移至 `ModelOutput` 作为校验结果元数据）。
>
> **legacy 版本：** v2.6 及之前的 `ActionOutput` 结构体（使用 `p_batt_set` 字段）已废弃，仅用于兼容旧模式。

### 6.7 消息总线集成

| Topic | 发布者 | 订阅者 | 数据格式 | 频率 |
|-------|--------|--------|----------|------|
| `ai/action_output` | ModelManager | strategy-engine | ActionOutput JSON | 1Hz |
| `ai/reward_value` | RewardCalculator | OnlineUpdater, Web UI | RewardValue JSON | 1Hz |
| `ai/model_status` | ModelManager | Web UI, 告警模块 | ModelStatus JSON | 1Hz |
| `ai/mode_switch` | ModeSelector | RewardCalculator, strategy-engine | ModeSwitchEvent JSON | 事件驱动 |
| `ai/droop_range` | intercore | ActionValidator | {k_min, k_max} JSON | 按需更新 |
| `ai/current_mode` | ModeSelector | Web UI（心跳查询）| RunningMode JSON | 按需查询 |

**指令分发说明（v2.15）：** ActionOutput 的 2 个控制维度按以下路径分发：
- `p_ref` + `k_droop` → 通过 intercore（TCP/RJ45）发送到**实时控制模块**，用于下垂控制公式

**下沉至策略引擎执行的功能（v2.15）：**
- `load_shedding`（可中断负荷切除）→ 策略引擎的需量控制策略通过南向 RS485/HPLC 发送到负荷控制装置，AI 引擎不再直接输出
- `pv_limit`（光伏限功率）→ 策略引擎的防逆流策略通过南向 RS485/HPLC 发送到光伏逆变器，AI 引擎不再直接输出
- `confidence`（决策置信度）→ 从 ActionOutput 移至 ModelOutput，由 ActionValidator 校验后注入，作为决策质量评估元数据在 Web UI 展示

### 6.8 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| STATE-01 | 状态空间包含全部 10 个大类、78 维 | 单元测试 |
| STATE-02 | 每个字段的数据类型和取值范围与定义严格一致 | 单元测试 |
| STATE-03 | 状态输入到推理开始的总延迟 < 5ms | 性能测试 |
| STATE-04 | Option 字段为 None 时，RL 决策器自动取其维度值 = 0.0 并跳过相关约束 | 集成测试 |
| STATE-05 | 预测数据向量长度固定 15 维，超出/不足时自动裁剪/补零 | 单元测试 |
| STATE-v2.10-01 | FusedSystemState 新增 safety_override_active/reason/p_ref 字段 | P0 | v2.10 PRD |
| STATE-v2.10-02 | to_input_vector() 返回 59 维向量（向后兼容）| P0 | v2.10 PRD |
| STATE-v2.10-03 | q_realtime_margin 数据来源为核间 DataUpload 帧 | P0 | v2.10 PRD |
| OVERRIDE-01 | SafetyOverride 帧（0x0040）可正确解析 | P0 | v2.10 PRD |
| OVERRIDE-02 | FusedSystemState.safety_override_active 在收到帧后正确设置 | P0 | v2.10 PRD |
| OVERRIDE-03 | AI 感知 override_active=true 时获得 R_safety_override 惩罚 | P0 | v2.10 PRD |
| ACT-01 | 动作空间包含全部 2 个动作维度 | 单元测试 |
| ACT-02 | 每个动作维度的取值范围严格执行定义边界 | 单元测试 + clamp 验证 |
| ACT-03 | 约束校验违反时自动 clamp 并记录 WARN 日志 | 集成测试 |
| ACT-04 | 约束校验总延迟 < 0.5ms | 性能测试 |
| DUAL-01 | RL 模型输出包含 P_ref 和 k_droop 两个参数 | 单元测试 |
| DUAL-02 | P_ref 取值范围符合 ActionSpaceConfig 约束 | 单元测试（边界值验证）|
| DUAL-03 | k_droop 取值范围符合实时控制模块提供的 [k_min, k_max] | 集成测试 |
| DUAL-04 | 双参数通过 intercore 同时下发（TCP 帧 v2.0）| 集成测试（抓包验证）|
| DUAL-05 | 下垂公式 P = P_ref + k_droop × ΔV 在执行器端正确执行 | 集成测试（模拟 ΔV）|
| DUAL-06 | k_droop 管理权归 AI 引擎，实时控制模块不得修改 | 代码审查 |
| DUAL-07 | 通信中断时执行器保持最后有效的 P_ref 和 k_droop | 集成测试 |
| DUAL-08 | P_ref 和 k_droop 任一超范围时自动 clamp 并记录 WARN 日志 | 集成测试 |
| FALLBACK-01 | AI 推理失败时自动降级至本地策略 | 集成测试 |
| FALLBACK-02 | 通信中断时执行器保持最后有效的 P_ref 和 k_droop | 集成测试（断联测试）|
| FALLBACK-03 | 通信恢复后自动切回 AI 双参数控制 | 集成测试 |
| RL-01 | RL 模型加载成功 | 单元测试 |
| RL-02 | RL 决策延迟 < 1s | 性能测试 |
| RL-03 | RL 决策综合回报相比固定策略提升 >= 20% | 对比实验 |

### 6.9 电压异常应急策略（v2.9）

RobustnessManager 负责检测电压异常和电池 SOC 异常，在 RL 决策前返回应急动作，保障本质安全。

#### 6.9.1 异常类型

```rust
pub enum AnomalyType {
    VoltageSag,               // 电压骤降（< 0.90 p.u.）
    VoltageSurge,             // 电压骤升（> 1.10 p.u.）
    BatterySocCritical,        // 电池 SOC 极低（< 10%）
    BatterySocOverfull,       // 电池 SOC 极满（> 95%）
    CommunicationTimeout,     // 核间通信超时（> 5s）
}
```

#### 6.9.2 应急动作策略

| 异常类型 | 检测条件 | 应急动作 | 说明 |
|----------|----------|----------|------|
| VoltageSag | v_avg < 0.90 p.u. | p_ref = max_discharge（放电补充）| 立即响应，不等待 RL |
| VoltageSurge | v_avg > 1.10 p.u. | p_ref = max_charge（充电消纳）| 防止逆流 |
| BatterySocCritical | soc < 0.10 | p_ref = 0（停止放电）| SOC 极低保护 |
| BatterySocOverfull | soc > 0.95 | p_ref = 0（停止充电）| 防止过充 |
| CommunicationTimeout | intercore 断开 > 5s | 执行器保持最后有效参数 | 通信中断不停机 |

#### 6.9.3 与 strategy-engine 集成

```rust
// ai_integration.rs
async fn dispatch_ai_decision(&self, action: ActionOutput) -> Result<(), AiError> {
    // 1. 先进行异常检测
    if let Some(robust_action) = self.robustness_manager.detect_and_respond(&self.fused_state).await? {
        return self.dispatch_robust_action(robust_action).await;
    }
    // 2. 正常 RL 决策下发
    self.rl_model.decide(...).await
}
```

#### 6.9.4 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| RB-01 | VoltageSag 检测延迟 < 100ms | 集成测试 |
| RB-02 | 应急动作下发延迟 < 200ms | 性能测试 |
| RB-03 | 应急动作期间 AI 推理暂停，不产生冲突指令 | 集成测试 |
| RB-04 | 异常恢复后自动切回 AI 模式 | 集成测试 |

---

## 7. 5 种场景奖励函数

### 7.1 功能概述

RewardCalculator 根据当前场景标签，选择对应的奖励函数公式计算奖励值。奖励值用于在线微调阶段的模型权重更新，以及 Web UI 上展示决策质量。

### 7.2 SCENE-01：台区季节性负荷模式

**优化目标：** 最大化光伏消纳 + 防止变压器过载 + 电池寿命保护 + P-Q 协同优化

**v2.5 分层架构原则：**
- AI 仅在实时模块无功耗尽时才对电压偏差负责（q_realtime_margin <= 10% + 越限连续 2 步）
- 实时模块有裕度时，电压问题由实时模块自行处理，AI 不因"旁观"被惩罚
- 自适应损耗系数 α(s) ∈ {1.0, 0.2, 3.0} 区分"常规调度"与"应急处置"的电池损耗价值差异

**完整公式：**
```
R_agri = w1 * R_pv_consumption
       - α(s) * w2 * P_battery_degradation
       - w3 * P_transformer_overload
       + w4 * R_PQ_coordination
       - w5 * R_ramp
       - w6 * R_voltage_slope
       - w7 * R_smooth
       - R_safety_override
```

**自适应损耗系数 α(s)：**
| 条件 | α(s) 值 | 说明 |
|------|---------|------|
| battery_soc < 10% | 3.0 | SOC 极低保护，优先级最高 |
| q_realtime_margin <= 10% 且 \|ΔV\| > 5% 连续 >= 2 步 | 0.2 | 电压支撑模式 |
| 其他 | 1.0 | 常规调度 |

**P-Q 协同度奖励 R_PQ_coordination（v2.8 + v2.13 Sigmoid 平滑化）：**
```rust
w_save = 1 / (1 + exp(-k * (q_margin - q_threshold)));
w_support = 1 - w_save;
r_pq = w_save * r_lazy + w_support * r_correct;
```
- Q 有裕度（q_margin > 10%）：AI"偷懒"省电池 → +50
- Q 饱和（q_margin <= 10%）：低电压 + 放电 或 高电压 + 充电 → +50

**弃光奖励差异化（v2.8）：**
- v_avg >= 1.05 时：充电消纳 → 正常奖励；放电 → -20 惩罚

**SafetyOverride 惩罚 R_safety_override（v2.14 重构）：**
```rust
if consecutive < 10 {
    // 样本不足：原因固定惩罚 / 15
    match reason {
        "voltage_violation" => -50.0 / 15,
        "q_exhausted" => -30.0 / 15,
        "emergency" => -100.0 / 15,
        _ => -20.0 / 15,
    }
} else {
    // 样本充足：比例 + 连续次数惩罚，归一化至 [-1, 0]
    (-5.0 * ratio - 10.0 * (consecutive / 10).clamp(0, 1)) / 15.0
}
```

**互斥逻辑（v2.14）：** `safety_override_active = true` 时，跳过该步的 P-Q 协同度惩罚

**权重配置：**
| 权重 | 默认值 | 说明 | 可配置范围 |
|------|--------|------|------------|
| w1 | 1.0 | 光伏消纳奖励 | [0.0, 3.0] |
| w2 | 0.5 | 电池损耗惩罚 | [0.0, 2.0] |
| w3 | 2.0 | 变压器过载惩罚 | [0.0, 5.0] |
| w4 | 1.0 | P-Q 协同度奖励 | [0.0, 5.0] |
| w5 | 0.5 | 功率变化率惩罚 | [0.0, 2.0] |
| w6 | 0.5 | 电压变化斜率惩罚 | [0.0, 2.0] |
| w7 | 0.5 | 下垂系数平滑惩罚 | [0.0, 2.0] |
| w8 | 1.0 | 安全覆盖惩罚 | [0.0, 5.0] |

### 7.3 SCENE-B1：自主套利模式

**优化目标：** 最大化峰谷电价差收益 + 最小化电池损耗

> **v2.4 说明：** Q 由实时控制模块调节，本奖励函数中 Q 的影响体现在 P_batt 物理方程中。

```
R_arbitrage = w1 * R_price_spread - w2 * P_battery_degradation

R_price_spread = sum(P_batt_set * delta_t * (price_sell - price_buy)) * conversion_factor
P_battery_degradation = beta * sum(|P_batt_set| * delta_t) / E_battery_total * 100
```

### 7.4 SCENE-B2：需量控制模式

**优化目标：** 减免需量罚金 + 最小化舒适度损失

```
R_demand = w1 * R_demand_penalty_avoidance - w2 * P_comfort_loss

R_demand_penalty_avoidance = max(0, D_peak_baseline - D_peak_actual) * penalty_rate
P_comfort_loss = gamma * P_load_shed * delta_t * price_loss
```

### 7.5 SCENE-B3：虚拟电厂模式

**优化目标：** 最大化辅助服务收益 + 响应精度

```
R_vpp = w1 * R_ancillary_service + w2 * R_response_accuracy - w3 * P_deadline_deviation

R_ancillary_service = P_regulation_capacity * capacity_price + P_regulation_mileage * mileage_price
R_response_accuracy = 100 * max(0, 1 - |P_actual - P_target| / P_target_range)
```

### 7.6 SCENE-B5：极致绿色模式

**优化目标：** 最大化绿电消纳比例 + 最小化碳排放

```
R_green = w1 * R_green_consumption + w2 * R_carbon_reduction

R_green_consumption = 100 * E_green_self_consume / E_total_consume
R_carbon_reduction = 100 * (C_baseline - C_actual) / C_baseline
```

### 7.7 场景-权重映射表

| 场景 | w1 | w2 | w3 | w4 | 说明 |
|------|----|----|----|----|------|
| 台区季节性负荷 | 1.0 | 0.5 | 2.0 | - | 变压器过载惩罚最重 |
| 自主套利 | 1.0 | 1.0 | - | - | 经济性主导 |
| 需量控制 | 1.0 | 0.5 | - | - | 需量费减免为主 |
| VPP | 1.0 | 2.0 | 1.0 | - | 响应精度权重最高 |
| 极致绿色 | 1.0 | 1.0 | - | - | 环境效益导向 |

### 7.8 动态权重调整

| ID | 标准 | 验证方法 |
|----|------|----------|
| WEIGHT-01 | 场景切换时权重映射自动更新，延迟 < 1s | 集成测试 |
| WEIGHT-02 | 权重映射表可通过配置文件修改，支持热加载 | 配置测试 |
| WEIGHT-03 | 策略管理员可通过 Web UI 手动调整当前场景权重 | UI 集成测试 |
| WEIGHT-04 | 手动调整的权重在下一次场景切换时复位为默认值 | 集成测试 |
| WEIGHT-05 | 权重修改记录操作日志，包含修改人、修改时间、修改前后值 | 日志验证 |

### 7.9 奖励函数验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| REWARD-A1 | 光伏完全消纳时 R_pv_consumption = 100 | 单元测试 |
| REWARD-A2 | 变压器负载率 1.0 时 P_transformer_overload = 0，1.1 时 = 20 | 单元测试 |
| REWARD-A3 | delta_SOC = 0 时 P_battery_degradation = 0 | 单元测试 |
| REWARD-B1 | 峰时放电、谷时充电时 R_price_spread > 0 | 单元测试 |
| REWARD-B2 | 峰时充电时 R_price_spread < 0（策略错误惩罚）| 单元测试 |
| REWARD-B3 | 电池无动作时 R_arbitrage = 0 | 单元测试 |
| REWARD-C1 | 成功削峰 100kW 时 R_demand_penalty_avoidance = 100 * penalty_rate | 单元测试 |
| REWARD-C2 | D_peak_actual >= D_peak_baseline 时 R_demand_penalty_avoidance = 0 | 单元测试 |
| REWARD-C3 | 无切负荷时 P_comfort_loss = 0 | 单元测试 |
| REWARD-D1 | P_actual = P_target 时 R_response_accuracy = 100 | 单元测试 |
| REWARD-D2 | P_actual 偏离超过 P_target_range 时 R_response_accuracy = 0 | 单元测试 |
| REWARD-D3 | 响应延迟 delta_t <= T_allowed 时 P_deadline_deviation <= 100 | 单元测试 |
| REWARD-D4 | VPP 指令无效时 R_vpp 强制置 0 | 集成测试 |
| REWARD-E1 | 全部用电来自绿电时 R_green_consumption = 100 | 单元测试 |
| REWARD-E2 | C_actual = 0 时 R_carbon_reduction = 100 | 单元测试 |
| REWARD-E3 | C_actual >= C_baseline 时 R_carbon_reduction = 0 | 单元测试 |
| REWARD-E4 | 电网排放因子从配置文件读取，默认 0.581 kg CO2/kWh | 配置验证 |
| REWARD-ALL | 奖励函数完整计算时间 < 1ms | 性能测试 |
| REWARD-v2.5-01 | q_realtime_margin > 0.10 时 R_voltage = 0（条件不触发）| P0 | v2.5 PRD |
| REWARD-v2.5-02 | q_realtime_margin <= 0.10 且电压越限连续2步时触发电压惩罚 | P0 | v2.5 PRD |
| REWARD-v2.5-03 | SOC < 10% 时 α = 3.0，电池损耗惩罚加重 | P0 | v2.5 PRD |
| REWARD-v2.5-04 | v_avg >= 1.05 p.u. 时弃光奖励 = 0 | P0 | v2.5 PRD |
| REWARD-v2.5-05 | α(s) 三状态（常规/电压支撑/SOC极低）互斥，取最高优先级 | P0 | v2.5 PRD |
| REWARD-v2.8-01 | Q 有裕度时 AI 不动作（|p_ref| < 5kW）→ R_PQ = +50.0 | P0 | v2.8 PRD |
| REWARD-v2.8-02 | Q 饱和 + 低电压时 AI 放电（p_ref < 0）→ R_PQ = +50.0；不放电 → R_PQ = -30.0 | P0 | v2.8 PRD |
| REWARD-v2.8-03 | Q 饱和 + 高电压时 AI 充电（p_ref > 0）→ R_PQ = +50.0；不充电 → R_PQ = -30.0 | P0 | v2.8 PRD |
| REWARD-v2.8-04 | v_avg >= 1.05 时 AI 充电消纳 → R_pv 正常；放电 → R_pv = -20.0 | P0 | v2.8 PRD |
| REWARD-v2.8-05 | R_smooth 惩罚项存在（|Δk_droop| + λ·超限惩罚）| P0 | v2.8 PRD |
| REWARD-v2.10-01 | safety_override_active=true 时 R_safety_override 根据触发原因惩罚 | P0 | v2.10 PRD |
| CONFIG-v2.5-01 | reward_thresholds 配置项可通过 ai.toml 加载，缺失时自动回退默认值 | P1 | v2.5 PRD |
| TO-01 | L=0.70 时变压器过载惩罚为 0 | 单元测试 |
| TO-02 | L=0.90 时变压器过载惩罚为 10 | 单元测试 |
| TO-03 | L=1.00 时变压器过载惩罚为 50 | 单元测试 |
| TO-04 | L=1.05 时变压器过载惩罚为 100（硬惩罚）| 单元测试 |
| TO-05 | 变压器过载持续时间降低 >= 40% | 对比实验 |
| DV-01 | ΔV=0 时 w6 = base_w6 | 单元测试 |
| DV-02 | ΔV=0.05 时 w6 = base_w6 × (1.0 + k × 0.05) | 数学验证 |
| DV-03 | k 值可配置，范围 [0.0, 5.0] | 配置测试 |
| DV-04 | 电压波动幅度降低 >= 25% | 对比实验 |
| SH-01 | 无冲击负荷时 R_shock = 0 | 单元测试 |
| SH-02 | 冲击负荷发生时，策略引擎执行 load_shedding 越大奖励越高（v2.15：load_shedding 值从策略引擎观测获取，非 AI 直接输出）| 单元测试 |
| SH-03 | 响应时间越长惩罚越大 | 单元测试 |
| SH-04 | 需量超标次数降低 >= 30% | 对比实验 |
| TH-01 | Q_THRESHOLD 可通过配置修改 | 配置测试 |
| TH-02 | P_THRESHOLD 可通过配置修改 | 配置测试 |
| TH-03 | 配置缺失时使用默认值（Q=0.10, P=5.0）| 单元测试 |
| TH-04 | 阈值修改后即时生效（热重载）| 配置热重载测试 |

---

## 8. RKNN Runtime NPU 推理

### 8.1 功能概述

RKNN Runtime 是 Rockchip 提供的 NPU 推理引擎，通过 FFI 调用 `librknnrt.so` C 库，在 RK3588 NPU 上执行 INT8/FP16 量化模型推理。所有 FFI 调用使用 `tokio::task::spawn_blocking` 在后台线程执行。

**模型生命周期状态：**

```rust
pub enum ModelStatus {
    Unloaded,   // 未加载
    Loading,    // 加载中
    Ready,      // 就绪，可推理
    Error,      // 错误状态（触发降级）
}
```

### 8.2 推理流程

```
训练阶段 (x86 服务器):
PyTorch → ONNX → rknn-toolkit2 量化 → INT8 模型 (.rknn)

部署阶段 (RK3588):
INT8 模型 → RKNN Runtime → NPU 推理 (< 100ms)
```

### 8.3 FFI 核心函数

| C API | 功能 | Rust FFI 签名 |
|-------|------|---------------|
| `rknn_init` | 模型加载与初始化 | `fn rknn_init(ctx: *mut u64, model_path: *const c_char, type: c_int, flag: c_int) -> c_int` |
| `rknn_inputs_set` | 输入 tensor 设置 | `fn rknn_inputs_set(ctx: u64, n: u32, inputs: *mut RknnInput) -> c_int` |
| `rknn_run` | 推理执行 | `fn rknn_run(ctx: u64, reserved: *mut u64) -> c_int` |
| `rknn_outputs_get` | 输出 tensor 获取 | `fn rknn_outputs_get(ctx: u64, n: u32, outputs: *mut RknnOutput) -> c_int` |
| `rknn_destroy` | 资源释放 | `fn rknn_destroy(ctx: u64) -> c_int` |

**RKNN 错误码映射：**

| 错误码 | 含义 | 映射错误类型 |
|--------|------|-------------|
| 0 | 成功 | - |
| -1 | 初始化失败 | `ModelLoadFailed` |
| -2 | 模型格式错误 | `ModelLoadFailed` |
| -3 | 模型不符合框架要求 | `ModelLoadFailed` |
| -4 | SDK 版本不匹配 | `ModelLoadFailed` |
| -5 | 输入数量不匹配 | `InferenceFailed` |
| -6 | 输出数量不匹配 | `InferenceFailed` |
| -7 | 输入格式错误 | `InferenceFailed` |
| -8 | 输出格式错误 | `InferenceFailed` |
| -9 | 推理超时 | `InferenceFailed` |
| -10 | 上下文无效 | `InferenceFailed` |

### 8.4 推理延迟预算

| 阶段 | 最大延迟 | 说明 |
|------|----------|------|
| 状态输入准备 | 5ms | 融合数据读取 + 特征向量序列化 |
| NPU 推理 | 100ms | RKNN Runtime run() 调用 |
| 动作输出校验 | 0.5ms | 约束规则校验 + clamp |
| **总端到端延迟** | **120ms** | 从状态输入就绪到动作输出可用 |

### 8.5 技术约束

| 约束项 | 说明 |
|--------|------|
| 链接方式 | 静态链接 `#[link(name = "rknnrt")]` 优先，兜底采用 `libloading` 动态加载 |
| 异步安全 | 所有 FFI 调用必须使用 `spawn_blocking`，不得在 async 上下文中直接调用阻塞 C API |
| 线程安全 | `RwLock<Option<RknnContext>>` 提供内部可变性；`unsafe impl Send + Sync` |
| 内存安全 | 输入/输出缓冲区使用 `Box::into_raw` / `Box::from_raw`，确保无 use-after-free |
| NPU 独占 | AI 推理任务独占 RK3588 NPU 核心，不与非 AI 任务共享 |

### 8.6 推理性能验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| NPU-01 | AI 推理任务独占 RK3588 NPU 核心 | 内核调度配置验证 |
| NPU-02 | NPU 推理延迟 < 100ms (P99) | 性能测试（1000 次采样） |
| NPU-03 | AI 完整决策总延迟 < 120ms | 性能测试 |
| NPU-04 | NPU 推理失败时自动降级至 CPU 推理，降级延迟 < 5s | 集成测试 |
| NPU-05 | CPU 推理模式下推理延迟 < 500ms | 性能测试 |
| RK-01 | rknn_init 成功加载 .rknn 模型 | 单元测试 |
| RK-02 | rknn_run 正确执行推理 | 单元测试 |
| RK-03 | 输入形状验证正确 | 单元测试 |
| RK-04 | 输出形状正确 | 单元测试 |
| RK-05 | 资源正确释放 (rknn_destroy) | 单元测试 |
| RK-06 | 错误处理正确 | 单元测试 |
| RK-07 | 异步封装不阻塞 runtime | 集成测试 |
| RK-08 | Send + Sync 实现正确 | 编译测试 |
| NPU-06 | NPU 温度监控，超过 85°C 时触发降频保护，推理频率降低不超过初始频率的 50% | 压力测试 |

---

## 9. 在线微调

### 9.1 功能概述

OnlineUpdater 模块负责基于新产生的运行数据，在设备运行期间对模型权重进行增量更新，使模型持续适应工况变化。

### 9.2 核心需求

| 需求 | 说明 |
|------|------|
| 触发方式 | 数据积累达到 batch_size=32 后自动触发 |
| 触发限制 | 仅在系统闲时（负荷率 < 30%）触发 |
| 更新目标 | LSTM 预测模型权重 + RL 决策模型权重 |
| 安全保护 | loss 连续 10 周期不下降或上升时，停止微调并回滚至上一检查点 |
| 数据存储 | 保留最近 30 天训练数据，本地存储 <= 1GB |

### 9.3 影子模型验证 + 渐进式切换（v2.10）

```rust
pub enum UpdateError {
    SafetyViolation { score: f32, threshold: f32 },
    PerformanceDegradation { current: f32, shadow: f32 },
    SwitchInProgress,
    ModelNotReady,
}

async fn safe_update(&self, new_weights: &[f32]) -> Result<bool, UpdateError> {
    // 1. 影子模型验证
    self.shadow_model.load_weights(new_weights);
    let safety_score = self.safety_constraints.validate(&self.shadow_model).await?;
    if safety_score < self.safety_threshold {
        return Err(UpdateError::SafetyViolation);
    }

    // 2. 性能验证
    let current_perf = self.performance_monitor.evaluate(&self.current_model).await?;
    let shadow_perf = self.performance_monitor.evaluate(&self.shadow_model).await?;
    if shadow_perf < current_perf * 0.95 {
        return Err(UpdateError::PerformanceDegradation);
    }

    // 3. 渐进式切换
    self.gradual_switch(new_weights).await
}
```

**UpdateError 错误处理策略：**

| 错误类型 | 触发条件 | 处理策略 | 日志级别 |
|----------|----------|----------|----------|
| SafetyViolation | 影子模型安全评分 < 阈值 | 拒绝更新，保留当前权重 | WARN |
| PerformanceDegradation | 影子模型性能 < 当前 × 0.95 | 拒绝更新，保留当前权重 | WARN |
| SwitchInProgress | 切换进行中再次调用 safe_update | 拒绝，返回 Ok(false) | INFO |
| ModelNotReady | 影子模型未初始化 | 返回错误 | ERROR |

### 9.4 验收标准

| ID | 标准 | 验证方法 |
|----|------|----------|
| UPDATE-01 | 在线微调功能正常 | 单元测试 |
| UPDATE-02 | 在线微调延迟 <= 10s（单次微调，batch_size=32）| 性能测试 |
| UPDATE-03 | 微调期间不影响并发推理请求 | 集成测试 |
| UPDATE-04 | Loss 发散时自动停止微调并回滚 | 集成测试 |
| AC1 | 新权重违反安全约束时，拒绝更新并返回错误 | 单元测试 |
| AC2 | 影子模型性能下降超过 5% 时，拒绝更新 | 单元测试 |
| AC3 | 渐进式切换权重，每步间隔可配置（默认 1 秒）| 集成测试 |
| AC4 | 切换过程记录日志，包含每步权重混合比例 | 日志审查 |

### 9.5 自适应权重优化器（v2.11）

AdaptiveWeightOptimizer 基于元学习（MetaRL）和 NSGA-II 多目标优化，自动调优 SCENE-01 奖励函数的 w1~w7 权重，减少人工调参依赖。

#### 9.5.1 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `AdaptiveWeightOptimizer` | `adaptive_weight_optimizer.rs` | MetaRL 元学习器，基于历史性能数据预测最优权重调整方向 |
| `ParetoWeightOptimizer` | `pareto_optimizer.rs` | NSGA-II 多目标优化器，搜索 Pareto 前沿权重候选 |
| `PerformanceCollector` | `performance_collector.rs` | 性能指标收集器，采集光伏消纳率、电池循环次数、变压器负载等 |

#### 9.5.2 数据流

```
离线训练管线 → MetaLearner 训练 → AdaptiveWeightOptimizer → SceneWeights 更新
                                                                        ↓
                                               RL 模型决策 → PerformanceCollector 收集
```

#### 9.5.3 约束保护

| 约束 | 说明 |
|------|------|
| 权重非负 | w_i >= 0 |
| 权重比例 | w_i / sum(w) <= max_adjustment_per_update |
| 物理一致性 | w3（变压器过载）始终为最高权重优先级 |

#### 9.5.4 验收标准

| ID | 标准 | 优先级 | 验证方法 |
|----|------|--------|----------|
| AWO-01 | 自适应权重优化器可正确加载配置 | P0 | 单元测试 |
| AWO-02 | 元学习器可基于历史性能数据输出权重调整 | P1 | 离线仿真验证 |
| AWO-03 | NSGA-II 可搜索 Pareto 前沿并输出多组权重候选 | P1 | 离线仿真验证 |
| AWO-04 | 权重调整受物理约束约束（正数、比例）| P0 | 单元测试 |
| AWO-05 | 单次更新周期内权重变化不超过 max_adjustment_per_update | P0 | 单元测试 |
| AWO-06 | 优化后的奖励函数与原始策略无显著偏离（偏移 < 5%）| P1 | 集成测试 |
| AWO-07 | 权重更新后 RL 模型可正常推理（推理延迟 < 1s）| P0 | 集成测试 |

---

## 10. 非功能性需求

### 10.1 推理性能

| 指标 | 要求 | 测量方法 |
|------|------|----------|
| NPU 推理延迟 | < 100ms (P99) | 1000 次连续推理，计算 P99 |
| 场景切换延迟 | < 2s（远程）、< 1s（本地） | 模拟 100 次切换 |
| 状态空间构建延迟 | < 5ms | 1000 次构建 |
| 动作约束校验延迟 | < 0.5ms | 1000 次校验 |
| 奖励函数计算延迟 | < 1ms（单场景）| 1000 次计算 |
| LSTM 预测延迟 | < 1s | 性能测试 |
| RL 决策延迟 | < 1s | 性能测试 |
| AI 完整决策周期 | 1Hz（默认，与融合周期一致）| 运行时观测 |
| 权重优化推理延迟（v2.11）| < 100ms | 性能测试 |
| 权重优化更新周期（v2.11）| >= 1 小时 | 运行时观测 |
| 分位数预测延迟（v2.11）| <= 1s | 性能测试 |
| 冲击负荷概率计算延迟（v2.11）| <= 10ms | 性能测试 |

### 10.2 模型精度

| 指标 | 要求 | 测量方法 |
|------|------|----------|
| 光伏预测 MAPE | <= 10% | 回测验证 |
| 负荷预测 MAPE | <= 15% | 回测验证 |
| RL 决策综合回报 | 相比固定策略提升 >= 20% | 对比实验 |

### 10.3 模型大小与资源占用

| 指标 | 要求 |
|------|------|
| 单模型 INT8 量化后大小 | <= 5MB |
| 推理运行时内存占用 | <= 200MB |
| 训练数据本地存储 | <= 1GB（30 天） |
| 日志存储 | 按系统滚动策略（单文件 10MB，保留 10 个） |

### 10.4 可靠性

| 指标 | 要求 |
|------|------|
| AI 引擎 MTBF | >= 1,000 小时 |
| AI 失效时自动降级至本地策略 | < 2s |
| 模型热加载 | 支持（双缓冲模式） |
| 模型版本回滚 | 支持，回滚时间 < 30s |
| A/B 测试框架 | 支持模型版本对比评估（新版本 vs 当前版本），详见模块08-PRD 第8章 |

### 10.5 安全性

| 需求 | 说明 |
|------|------|
| 模型文件完整性 | 模型文件加载前进行 SHA256 校验，校验失败拒绝加载 |
| 推理输入验证 | 对输入张量进行 NaN/Inf 检查，异常输入拒绝推理 |
| 动作输出限幅 | 所有动作输出经 ActionValidator 校验后再下发 |
| 在线微调防护 | 在线微调仅在系统闲时（负荷率 < 30%）触发，微调不得影响推理性能 |
| 配置加密 | 奖励函数权重参数存储在加密配置文件中 |

### 10.6 Phase 3C 验收标准汇总

| ID | 标准 | 优先级 | 来源 |
|----|------|--------|------|
| AI-01 | LSTM 模型加载成功 | P0 | Phase3C 设计 |
| AI-02 | RL 模型加载成功 | P0 | Phase3C 设计 |
| AI-03 | LSTM 预测延迟 < 1s | P0 | Phase3C 设计 |
| AI-04 | RL 决策延迟 < 1s | P0 | Phase3C 设计 |
| AI-05 | ONNX 模型格式正确 | P0 | Phase3C 设计 |
| AI-06 | RK3588 NPU INT8 量化支持 | P0 | Phase3C 设计 |
| AI-07 | 在线微调功能正常 | P1 | Phase3C 设计 |
| AI-08 | 与 strategy-engine 集成正确 | P0 | Phase3C 设计 |

---

## 11. 异常流程与边界条件

### 11.0 错误类型定义

```rust
#[derive(Error, Debug)]
pub enum AiEngineError {
    #[error("模型加载失败: {0}")]
    ModelLoadFailed(String),
    #[error("模型文件校验失败: 期望 {expected}, 实际 {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("推理失败: {0}")]
    InferenceFailed(String),
    #[error("模型未加载")]
    ModelNotLoaded,
    #[error("输入形状不匹配: 期望 {expected:?}, 实际 {actual:?}")]
    InputShapeMismatch { expected: Vec<i32>, actual: Vec<i32> },
    #[error("输出形状不匹配")]
    OutputShapeMismatch,
    #[error("RKNN Runtime 错误: {0}")]
    RknnError(String),
    #[error("模型版本不兼容: {0}")]
    VersionMismatch(String),
    #[error("在线微调失败: {0}")]
    OnlineUpdateFailed(String),
    #[error("数据融合失败: {0}")]
    FusionFailed(String),
    #[error("模式切换失败: {0}")]
    ModeSwitchFailed(String),
    #[error("动作校验失败: {0}")]
    ActionValidationFailed(String),
    #[error("数据源过期: {0}")]
    DataSourceStale(String),
    #[error("NPU 温度过高: current={current}°C, limit={limit}°C")]
    NpuOverheating { current: f32, limit: f32 },
    #[error("奖励计算错误: {0}")]
    RewardCalculationError(String),
}
```

**错误分类与恢复策略：**

| 错误类别 | 错误变体 | 恢复策略 |
|----------|----------|----------|
| 模型加载 | `ModelLoadFailed`, `VersionMismatch` | 拒绝启动，记录 ERROR，触发降级 |
| 推理运行时 | `InferenceFailed`, `RknnError`, `InputShapeMismatch`, `OutputShapeMismatch` | 重试 1 次，失败后记录 ERROR，连续 3 次后触发 NPU 降级 |
| 资源状态 | `ModelNotLoaded` | 等待模型加载完成 |
| 数据异常 | `FusionFailed`, `DataSourceStale` | 按缺失数据处理策略填充，连续 10 周期后触发降级 |
| 运维操作 | `ModeSwitchFailed`, `ActionValidationFailed`, `OnlineUpdateFailed` | 记录 WARN，操作回滚 |
| 硬件异常 | `NpuOverheating` | 降频保护，连续 5 周期正常后恢复 |

### 11.1 核心异常处理

| 异常场景 | 检测条件 | 处理措施 |
|----------|----------|----------|
| 通信中断 | intercore TCP 断开 | 执行器保持最后 P_ref 和 k_droop，继续下垂控制 |
| AI 推理失败 | RKNN 返回错误 | 自动降级至本地策略，< 2s 切换完成 |
| NPU 推理延迟超标 | 连续 100 次推理 > 10% 超出 150ms | 降级至 CPU 推理模式 |
| NPU 温度过高 | 温度 > 85°C | 触发降频保护，推理频率降低不超过初始频率的 50%；温度连续 5 周期恢复正常后自动恢复全速 |
| 推理精度持续下降 | loss 连续 10 周期不下降或上升 | 停止在线微调，回滚至上一检查点 |
| 模型文件损坏 | SHA256 校验失败 | 拒绝加载，尝试 OTA 备份恢复 |
| 数据源异常 | 连续 3 周期无更新 | WARN 告警；连续 10 周期 → AI 降级 |
| 无效模式 ID | 指令值超出 1~5 | 拒绝执行，返回 400 错误 |
| 奖励计算异常 | 奖励值超出 [0, 200] | 截断至边界值，记录 ERROR 日志 |
| 双参数缺失 | P_ref 和 k_droop 同时为 NaN | 触发 AI 降级至兜底策略 |
| k_droop 超范围 | k_droop < k_min 或 k_droop > k_max | clamp 至 [k_min, k_max]，记录 WARN 日志 |
| p_ref 超范围 | p_ref < -P_safe 或 p_ref > P_safe | clamp 至 [-P_safe, P_safe]，记录 WARN 日志 |

### 11.2 数据缺失处理

| 缺失数据 | 处理方式 | 告警级别 |
|----------|----------|----------|
| 电价数据 | 上一有效值；连续 3 周期 → 默认分时电价表 | WARN |
| 气象数据 | 上一有效值；连续 10 周期 → R_green 置 0 | WARN |
| 调度指令 | 置 None，跳过相关约束 | INFO |
| LSTM 预测 | 全零向量 | WARN |
| 实时数据 | 上一有效值；连续 3 周期 → WARN；连续 10 周期 → AI 降级 | ERROR → 降级 |

### 11.3 降级流程

**AI 引擎异常检测条件：**
- ModelStatus == Error
- 连续 3 次推理失败
- 数据融合任一数据源连续 10 周期无更新

```
AI引擎异常 → 检测异常（心跳/状态码/连续失败计数）→ 切换至本地策略 (< 2s) → 告警通知 → strategy-engine接管
    ↓
数据恢复 5 连续周期后 → 自动切回 AI 模式
```

---

## 12. 配置参数

### 12.1 奖励函数阈值配置

```toml
[reward_thresholds]
# === 核心阈值（v2.5 初版，v2.12 R-07 可配置化）===
voltage_deadband = 0.05       # ±5% 死区
q_margin_threshold = 0.10     # Q 裕度阈值，低于此值视为"无功耗尽"
p_threshold_kw = 5.0          # P 阈值（kW），省电策略判定阈值
pv_high_limit = 1.05          # 弃光前置电压阈值
soc_critical = 0.10           # SOC 极低保护阈值，<此值触发 α=3.0

# === 电压惩罚系数（v2.5）===
voltage_penalty_high = 2.0   # 高电压侧（光伏超发）
voltage_penalty_low = 1.0     # 低电压侧（灌溉/炒茶/空调负荷）

# === 默认值行为（v2.5）===
# 配置文件缺失 `reward_thresholds` 时，所有参数自动回退至默认值，保证向后兼容
```

### 12.2 折扣累积奖励机制（v2.10）

```toml
[discounted_reward]
gamma = 0.99                 # 折扣因子，范围 [0.9, 0.999]
buffer_size = 1000            # 奖励历史缓冲区大小
```

**验收标准：**

| ID | 标准 | 验证方法 |
|----|------|----------|
| BC1 | 折扣因子 gamma 可配置，范围 [0.9, 0.999] | 配置测试 |
| BC2 | 缓冲区大小可配置，默认 1000 | 配置测试 |
| BC3 | gamma=0.99 时 100 步前奖励权重约 0.366 | 数学验证 |
| BC4 | 与现有奖励函数正交，不影响即时奖励计算 | 回归测试 |

### 12.3 v2.12 奖励函数改进配置

#### 12.3.1 R-01 奖励子项标准化（已实现）

各奖励子项统一归一化到 `[-1, 1]` 区间，解决量纲不一致问题，加速 RL 收敛：

| 子项 | 原始范围 | 归一化方式 | 说明 |
|------|----------|------------|------|
| r_pv_norm | [0, 100] | r / 100 | 光伏消纳 |
| p_batt_deg_norm | [0, ∞) | 线性映射 | 电池损耗 |
| p_trafo_norm | [0, 100] | r / 100 | 变压器过载 |
| r_pq_norm | [-30, 50] | (r + 30) / 80 | P-Q 协同度 |
| r_ramp_norm | [0, ∞) | tanh(r) | 功率变化率 |
| r_voltage_slope_norm | [0, ∞) | tanh(r) | 电压斜率 |
| r_smooth_norm | [0, ∞) | tanh(r) | 下垂平滑 |
| r_safety_override_norm | [-100, 0] | r / 100 | 安全覆盖 |

#### 12.3.2 R-02 塑造奖励（已实现）

引入提前预警机制，帮助 RL 学习避免危险状态：

```rust
// 过载预警：负载率 > 85% 开始预警，提前引导策略调整
fn overload_warning(load: f64) -> f64 {
    if load > 0.85 { (load - 0.85) / 0.15 } else { 0.0 }
}

// SOC 边界预警：接近 15% 或 85% 时预警
fn soc_warning(soc: f64, lambda: f64) -> f64 {
    if soc < 0.15 { lambda * (0.15 - soc) / 0.15 }
    else if soc > 0.85 { lambda * (soc - 0.85) / 0.15 }
    else { 0.0 }
}
```

#### 12.3.3 R-03 SOC 均衡奖励（已实现）

鼓励 SOC 保持在 50% 附近，延长电池寿命：

```rust
fn soc_balance_reward(soc: f64, lambda: f64) -> f64 {
    -lambda * (soc - 0.5).abs()
}
```

#### 12.3.4 R-04~R-07 规划项配置

```toml
[reward_planned]
# === R-04 变压器过载分段惩罚（规划中）===
transformer_overload_piecewise = true
# 分段函数：
#   L < 0.75:          0.0                    # 安全区
#   0.75 <= L < 0.90:  (L - 0.75) / 0.15 * 10  # 线性增长，0~10
#   0.90 <= L < 1.00:  10 + (L - 0.90) / 0.10 * 40  # 指数增长，10~50
#   L >= 1.00:          100.0                  # 硬惩罚

# === R-05 电压斜率动态权重（规划中）===
voltage_slope_dynamic_weight = true
# w6(v) = base_w6 × (1.0 + k × |ΔV|)
w6_base = 0.5                 # 基础权重
w6_k = 2.0                   # 放大系数，范围 [0.0, 5.0]

# === R-06 冲击负荷响应奖励（规划中）===
shock_response_enabled = true
w_shock = 20.0               # 冲击负荷响应权重
lambda_shock = 5.0            # 响应时间惩罚系数
shock_threshold_kw = 10.0     # 冲击负荷判定阈值（P90 - P50 > threshold）
```

### 12.4 v2.13 精细化改进配置

> **v2.13 核心变更：** 基于专家建议（2026-06-14）中有参考意义的 P0/P1 项，解决以下问题：

| 来源 | 问题 | 解决方案 |
|------|------|----------|
| 专家建议 §2.2 | P-Q 协同硬阈值导致策略震荡 | Sigmoid 平滑过渡 |
| 专家建议 §2.1 | 归一化系数硬编码，跨台区泛化差 | Welford 动态归一化 |
| 专家建议 §2.3 | 缺乏"动作-效果"因果链 | 状态改善率奖励 |
| 专家建议 §2.5 | 1Hz 无法响应 ms 级冲击 | 冲击负荷预备度奖励 |
| 专家建议 §3.2 | 在线微调分布偏移风险 | PER + KL 正则化 |
| 专家建议 §3.3 | 场景切换策略真空 | 策略混合（动作插值） |

**配置：**

```toml
[reward_v2_13]
# === Welford 动态归一化 ===
welford_epsilon = 1e-6       # 防止除零

# === Sigmoid 平滑化（P-Q 协同度阈值过渡）===
sigmoid_k = 50.0             # 控制过渡陡峭程度

# === 状态改善率奖励 ===
state_improve_weight = 1.0    # R_improve = w * (V_dev_prev - V_dev_curr) * sign(P_action)

# === 冲击负荷预备度奖励 ===
readiness_w1 = 1.0           # SOC 储备权重
readiness_w2 = 1.0           # P_ref 储备权重
soc_reserve_target = 0.3      # 目标 SOC 储备
p_ref_reserve_target = 10.0   # 目标放电基准点储备（kW）

# === PER+KL 正则化（在线微调）===
per_enabled = true           # 优先采样高 TD-error 样本
kl_beta = 0.01               # KL 散度约束系数
# L_online = L_task + β * D_KL(π_new || π_offline)

# === 策略混合（场景切换过渡）===
policy_mix_enabled = true     # a_blended = (1-α)*a_old + α*a_new
mix_steps = 10                # 过渡步数
```

### 12.5 下垂控制参数（v2.7/v2.8）

```toml
[droop_control]
K_MAX = 50.0                # k_droop 上限，防止系统震荡
lambda_smooth = 1.0          # 超限惩罚系数
# R_smooth = -|Δk_droop| - λ * max(0, k_droop - K_MAX)
```

### 12.6 在线微调参数（v2.10）

```toml
[online_updater]
batch_size = 32              # 触发微调的样本数
idle_load_threshold = 0.30   # 系统闲时阈值（负荷率 < 30%）
max_retention_days = 30     # 训练数据保留天数
max_storage_mb = 1024        # 本地存储上限（MB）
rollback_checkpoints = 1      # 回滚检查点数量
```

---

## 13. 历史遗留问题与技术债

| 编号 | 问题 | 说明 | 优先级 | 状态 |
|------|------|------|--------|------|
| T-01 | 状态空间维度描述错误 | PRD 历史版本标题写 59 维，实际应为 78 维 | **高** | ✅ 已修复（v2.14 统一版全部修正） |
| T-02 | R-04 变压器过载分段惩罚 | 分段函数：安全区(0~75%) / 线性(75~90%) / 指数(90~100%) / 硬惩罚(>100%) | 中 | ✅ 已实现（`overload_penalty_piecewise`） |
| T-03 | R-05 电压斜率动态权重 | w6(v) = base_w6 × (1.0 + k × \|ΔV\|) | 中 | ✅ 已实现（`dynamic_voltage_slope_weight`） |
| T-04 | R-06 冲击负荷响应奖励 | R_shock = w × load_shedding / max - λ × response_time / max_response（v2.15：load_shedding 由策略引擎执行，AI 引擎从观测获取该值计算奖励）| 中 | ⚠️ 未实现（仅有注释掉的测试桩） |
| T-05 | R-07 P-Q 阈值可配置化 | Q_THRESHOLD / P_THRESHOLD 从硬编码改为配置文件读取 | 中 | ✅ 已实现（`PqCoordinationThresholds` + Default） |
| T-06 | 在线微调 Phase 3C.2 | 原为占位框架，现已完整实现影子模型验证+渐进式切换 | 中 | ✅ 已实现（`SafeOnlineUpdater::safe_update`） |
| T-07 | v2.0 前 SceneClassifier | 已废弃，替换为 ModeSelector | 低 | ✅ 已关闭 |
| T-08 | v2.1 前 D5 电能质量 | 已从 AI 引擎移除，实时控制模块独立处理 | 低 | ✅ 已关闭 |
| T-09 | AI 引擎与 strategy-engine 集成测试 | RobustnessManager 异常检测→应急动作→正常恢复的端到端测试覆盖不足 | 中 | ⚠️ 待补充 |
| T-10 | Phase 2+ 协议支持 | IEC 61850-7-420、MQTT over TLS、SM2/SM4 国密（影响 AI 引擎数据管道安全） | 低 | 📋 规划中 |
| T-11 | P90 分位数误差离线评估 | PLF-06 要求 P90 误差 < 15%，需离线回测数据验证 | 低 | ⚠️ 待验证 |

---

## 14. 附录

### 14.1 ai-engine crate 模块结构

```
mupc/crates/ai-engine/
├── src/
│   ├── lib.rs                    # 模块导出
│   ├── model_manager.rs          # 模型管理器（统一调度）
│   ├── mode_selector.rs          # 运行场景选择器
│   ├── lstm_model.rs             # LSTM 预测模型
│   ├── rl_model.rs               # MADDPG/PPO 决策模型
│   ├── reward_calculator.rs      # 奖励函数计算器
│   ├── data_fusion.rs            # 多源数据融合引擎
│   ├── action_validator.rs       # 动作约束校验器
│   ├── online_updater.rs         # 在线微调
│   ├── rknn_runtime.rs           # RKNN Runtime 推理
│   ├── robustness_manager.rs    # 电压异常应急策略（v2.9）
│   ├── reward_normalizer.rs      # Welford 动态归一化（v2.13）
│   ├── adaptive_weight_optimizer.rs  # 自适应权重优化器（v2.11）
│   ├── pareto_optimizer.rs       # NSGA-II 多目标优化（v2.11）
│   ├── performance_collector.rs  # 性能指标收集器（v2.11）
│   ├── load_covariates.rs        # 负荷协变量（v2.11）
│   ├── error.rs                  # 错误类型
│   └── config.rs                 # 配置结构
```

### 14.2 待澄清问题

| 序号 | 问题 | 优先级 | 影响评估 |
|------|------|--------|----------|
| 1 | 气象数据的外部来源是何种 API？是否需要额外商务授权？ | 高 | 影响 DataFusionEngine 气象数据获取 |
| 2 | 电价数据是直接来自物联平台下发，还是需要通过 MUPC 本地配置？ | 高 | 影响 DataFusionEngine 电价数据管道 |
| 4 | VPP 辅助服务的容量价格和里程价格是否有标准合同模板？ | 中 | 影响 R_ancillary_service 参数来源 |
| 5 | 在线微调是否需要经过审批流程（安全考虑）？还是自动触发？ | 中 | 影响 OnlineUpdater 触发策略 |
| 6 | 气象数据连续缺失时长"10 个周期"是以融合周期（10 秒）还是气象更新周期（150 分钟）计？ | 中 | 影响 FUSION 告警阈值配置 |

---

**文档状态：** 统一版 v2.15（整合了 v1.0~v2.15 所有历史版本）

**来源文档：**
- `docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md`（v1.0~v2.15 历史版本合并）
- `docs/superpowers/plans/modules/05-MUPC-AI引擎-设计文档.md`（设计文档）