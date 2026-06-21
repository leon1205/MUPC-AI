# 上游训练管线（MUPC-AI2）改造要求

> **来源**：MUPC v3.0 预测增强分层混合架构（VMD + Attention + BiLSTM + 误差修正 + MSSA）
> **合并参考**：`docs/superpowers/specs/modules/05-MUPC-AI引擎-PRD.md` v3.0 §3.8、§14.3.3
> **设计参考**：`docs/superpowers/plans/modules/05-MUPC-AI引擎-设计文档.md` v3.0 第 14、15 章
> **日期**：2026-06-21
> **状态**：[REVIEWED: PASS] (2026-06-21 需求评审通过)

---

本文档汇总 MUPC 推理端（Rust/RK3588）已完成的三轮预测增强改造对上游训练管线（MUPC-AI2，Python/PyTorch）提出的全部修改要求。

---

## 1. ONNX 模型导出要求

### 1.1 模型文件清单

MUPC 推理端期望从训练管线获得以下模型文件（按必须程度排序）：

| 文件名 | 必须 | 轮次 | 说明 |
|--------|------|------|------|
| `lstm_attn.rknn` | **是** | R1 | 单向 LSTM + Attention，INT8 量化 |
| `bilstm_attn.rknn` | 否（Go/No-Go） | R2 | 双向 LSTM + Attention，INT8 量化。需硬件延迟摸底验证后启用 |
| `error_correction.rknn` | 否（可选） | R2 | 独立轻量 BiLSTM，用于残差修正，INT8 量化 |

### 1.2 Attention 层嵌入 ONNX 计算图

Attention 机制**必须由训练管线在 ONNX 导出时嵌入计算图**，MUPC 推理端（Rust）不做后处理。

**计算图节点**：LSTM hidden states → MatMul(score) → Softmax → ReduceSum(context) → FC(output)

**约束**：
- 所有算子（MatMul、Softmax、ReduceSum、FC）必须是 ONNX 标准算子，确保 RKNN Toolkit 2 可转换
- 可选：增加一个输出节点 `attention_weights`（shape=[input_window]），供推理端导出注意力可视化数据（对应验收标准 ATT-08）

### 1.3 ONNX metadata_props（必须）

每个导出的 ONNX 模型必须在 `metadata_props` 中包含以下键值对，推理端启动时与配置交叉校验：

| 元数据键 | 类型 | 说明 | 示例值 |
|----------|------|------|--------|
| `mupc_model_type` | string | 模型架构类型 | `"lstm"` 或 `"bilstm"` |
| `mupc_with_attention` | string | 是否含 Attention 层 | `"true"` 或 `"false"` |
| `mupc_with_vmd` | string | 是否期望推理端 VMD 预处理 | `"true"` 或 `"false"` |
| `mupc_mic_topk` | int | MIC 筛选的特征数 | `7` |
| `mupc_output_horizon` | int | 预测输出步数（固定 15） | `15` |
| `mupc_input_window` | int | 输入窗口步数 | `12`、`24` 或 `36` |
| `mupc_hidden_size` | int | LSTM 隐状态维度 | `64` 等 |
| `mupc_num_layers` | int | LSTM 层数 | `1`、`2` 或 `3` |
| `mupc_direction` | string | 单向/双向 | `"forward"` 或 `"bidirectional"` |
| `mupc_version` | string | 模型版本号 | `"v3.0.1"` |

**校验行为**：
- `mupc_model_type` 与配置 `bilstm.enabled` 交叉校验。不一致时若为必选模型则拒绝加载，若为可选模型则记录 WARN 跳过
- `mupc_with_vmd` 与配置 `vmd.enabled` 交叉校验。不一致时记录 WARN，以 metadata 为准（VMD 预处理实际是否执行）
- `mupc_input_window` 与推理端配置 `input_window_secs / step_seconds` 对比，不一致时拒绝加载

### 1.4 输入/输出维度约定

#### 1.4.1 无 VMD 模式

| 维度 | Shape | 说明 |
|------|-------|------|
| 输入 | `[batch_size, input_window, num_selected_features]` | input_window 由 MSSA 确定（12/24/36）；num_selected_features 由 MIC 确定（≤ top_k，当前为 7） |
| 输出 | `[batch_size, output_horizon, 3]` | output_horizon=15（15 步分位数预测）；channel 0=P10, 1=P50, 2=P90 |

#### 1.4.2 含 VMD 模式

| 维度 | Shape | 说明 |
|------|-------|------|
| 输入 | `[batch_size, input_window, num_selected_features]` | 同上。**训练阶段**：对每条样本先执行 VMD 分解得到 K 个子模态，各子模态分别输入 LSTM。**推理阶段**：推理端在 CPU 上执行 VMD 分解，各 IMF 分别送入 NPU 推理 |
| 输出 | `[batch_size, K, output_horizon, 3]` | K 为 VMD 模态数（光伏 4~6，负荷 5~8）；推理端在 K-dim 上 `sum(K)` 重构为 `[batch_size, output_horizon, 3]` |

**注意**：训练管线需输出 K 通道，推理端负责聚合。语义对齐由 `mupc_with_vmd = "true"` 元数据标记保证。

### 1.5 模型参数量约束

| 模型 | 约束 | 验收方法 |
|------|------|----------|
| LSTM + Attention (INT8) | ≤ 8MB | 文件大小验证 |
| BiLSTM + Attention (INT8) | ≤ 12MB | 文件大小验证 |
| BiLSTM 参数量 vs 单向 LSTM | ≤ 2.2x | metadata `mupc_hidden_size` × `mupc_num_layers` × 2（双向）校验 |
| 误差修正 BiLSTM (INT8) | ≤ 3MB（≤ 主模型参数量的 50%） | 文件大小验证 |

---

## 2. 训练脚本接口要求（供 MSSA 调用）

### 2.1 命令行接口

MSSA 超参优化工具（`tools/mssa_optimizer/`）通过 subprocess 调用训练脚本：

```bash
python -m mupc_ai2.train --config <临时训练配置文件路径>
```

**要求**：
- 训练脚本接受 `--config` 参数，指向一个 JSON/YAML 临时配置文件
- 配置文件包含单组超参（由 MSSA 每次迭代生成），格式见 §2.2

### 2.2 临时训练配置文件格式

```json
{
  "hidden_size": 64,
  "num_layers": 2,
  "attn_score": "additive",
  "vmd_k": 5,
  "vmd_alpha": 2000.0,
  "lr": 0.001,
  "batch_size": 32,
  "dropout": 0.2,
  "optimizer": "Adam",
  "input_window": 24
}
```

### 2.3 stdout 输出格式（MSSA 解析）

训练完成后，训练脚本**必须**向 stdout 输出以下格式的行（供 `ResultParser` 解析）：

```
PV_MAPE=0.073 LOAD_MAPE=0.11
```

**约束**：
- `PV_MAPE=` 后跟浮点数（光伏 MAPE）
- `LOAD_MAPE=` 后跟浮点数（负荷 MAPE）
- 两个值以空格分隔
- 正则解析模式：`PV_MAPE=([0-9.]+).*LOAD_MAPE=([0-9.]+)`

### 2.4 出错处理

- 训练脚本以非零退出码退出时，MSSA 将该组超参标记为无效（penalty_score = 1e6）
- 训练超时（默认 10 分钟）时同样标记为无效
- stdout 无法解析出 MAPE 值时标记为无效

---

## 3. 数据交换接口

### 3.1 MIC 分析 → 训练管线

MIC 离线分析脚本（Python，`minepy` 库）输出 JSON，训练管线读取以确定模型输入特征集。

**JSON Schema**：见 PRD §14.3.3.1（完整定义）。

**关键字段**：
- `features[]` — 每个特征的 MIC 值、排名、是否选中
- `top_k` — 筛选的特征数（默认 7）
- `analysis_metadata.target` — 预测对象（`"pv_power"` 或 `"load_power"`）

**训练管线消费方式**：
- 按 `features[].selected == true` 筛选特征
- 按 `rank` 升序排列特征维度
- 若 `selected` 特征数 ≠ `top_k`，以实际选中数量为准

### 3.2 MSSA 搜索结果 → 训练管线

MSSA 超参搜索完成后输出 JSON，训练管线读取以设置最优超参做最终训练。

**JSON Schema**：见 PRD §14.3.3.2（完整定义）。

**关键字段**：
- `best_hyperparameters` — 最优超参组合（10 维）
- `best_objective` — 最优加权 MAPE 及其分项
- `quality_flag` — `"usable"`（MAPE ≤ 人工基线 × 1.1）或 `"unusable"`（应使用人工基线超参）
- `convergence_curve` — 迭代收敛曲线

### 3.3 训练数据指纹

MSSA 缓存使用 `training_data_fingerprint` 来检测训练数据变更并自动使缓存失效。训练管线需提供此指纹（例如训练集 CSV 文件的 SHA256 前 16 位）。

---

## 4. 训练阶段数据处理要求

### 4.1 VMD 预处理

训练阶段须对每条训练样本的输入窗口执行 VMD 分解后送入 LSTM：

```
原始序列 → VMD(K) → [IMF_1, ..., IMF_K]
各 IMF 分别输入 LSTM/Attention → K 个输出通道
```

K 值在训练超参中确定（光伏 4~6，负荷 5~8）。

### 4.2 MIC 特征筛选

训练前，使用 MIC 离线分析结果确定模型输入特征集。未被选中的特征不参与训练。

### 4.3 误差修正模型训练

误差修正 BiLSTM（`error_correction.rknn`）的训练流程：

1. 主预测模型训练完成
2. 用主模型对训练集做预测，计算残差序列 `e = y_true - y_pred`
3. 以历史残差为输入、未来残差为输出，训练独立轻量 BiLSTM
4. 输入窗口 = `residual_window_steps`（默认 24），输出 = 15 步残差预测
5. 导出 ONNX → rknn-toolkit2 → `error_correction.rknn`
6. 偏差检查：主模型 Bias 绝对值 > 3% MAPE 时启用误差修正；否则可跳过训练

### 4.4 真分位数回归（建议，非本轮必须）

当前 LSTM 使用 MSE Loss + 后处理分位数（基于协变量调整和正态假设）。训练管线可在后续迭代中替换为 Quantile Loss（Pinball Loss）直接输出 P10/P50/P90，提升分位数预测精度。

---

## 5. 复现与验证要求

### 5.1 随机种子

- 训练须支持固定随机种子以复现结果
- MSSA 搜索亦通过 `random_seed` 保证确定性

### 5.2 精度验证

| 指标 | 基线 | R1 目标 | R2 目标 | 验证方法 |
|------|------|---------|---------|----------|
| 光伏 MAPE（第 1 步） | ≤ 10% | ≤ 8.5% | ≤ 7.5% | 测试集（≥ 30 天）回测 |
| 负荷 MAPE（第 1 步） | ≤ 15% | ≤ 13% | ≤ 12% | 测试集（≥ 30 天）回测 |
| 光伏 MAPE（第 15 步） | 无约束 | ≤ 22% | ≤ 18% | 测试集回测 |
| 负荷 MAPE（第 15 步） | 无约束 | ≤ 28% | ≤ 24% | 测试集回测 |
| Attention 相对改善 | — | MAPE 降低 ≥ 5% | — | 消融实验 |
| 误差修正绝对改善 | — | — | MAPE 降低 ≥ 3% | 消融实验 |
| 峰谷时段改善 | — | ≤ 日均 MAPE × 1.3 | ≤ 日均 MAPE × 1.15 | 分时段回测 |

### 5.3 BiLSTM 延迟摸底（准入条件）

在 R1 完成后、R2 BiLSTM 启用前，需在 RK3588 硬件上用原型 BiLSTM ONNX 模型执行延迟摸底：

- 若 P99 推理延迟 ≥ 900ms → **No-Go**（BiLSTM 不启用，仅保留误差修正）
- 若 P99 推理延迟 < 900ms → **Go**（BiLSTM 可启用）
- 结果写入配置 `bilstm.gate_passed`（`true`/`false`）

---

## 6. 与 MUPC 推理端的协作清单

| # | 训练管线职责 | 推理端职责 |
|---|-------------|-----------|
| 1 | 导出含 Attention 的 ONNX 模型 | 消费 .rknn，不实现 Attention 计算 |
| 2 | 嵌入 ONNX metadata_props（10 个键） | 启动时读取 metadata 与配置交叉校验 |
| 3 | 按 MIC 筛选结果训练模型（固定特征集） | 按 ONNX 输入 shape 构造 LstmInput |
| 4 | 训练并导出 VMD 兼容模型（K 通道输出） | CPU 执行 VMD 分解 + NPU 逐 IMF 推理 + 求和重构 |
| 5 | 训练并导出 `bilstm_attn.rknn` 和 `error_correction.rknn` | 根据配置+准入条件加载模型，独立管理 2-3 个 RknnRuntime |
| 6 | 训练脚本接受 `--config` + stdout 输出 MAPE | MSSA 通过 subprocess 调用 + 解析 stdout |
| 7 | 消费 MSSA JSON 做最终训练 | MSSA 工具在 `tools/mssa_optimizer/`，对最终 ONNX 透明 |
| 8 | 提供 `training_data_fingerprint` | MSSA 缓存命中判断 |

---

## 7. 非目标（本轮训练管线不做）

| 项 | 状态 | 说明 |
|----|------|------|
| QAT（量化感知训练） | 推迟 | 当前使用 PTQ（后训练量化），QAT 待后续精度需求推动 |
| 数据增强与领域随机化 | 推迟 | 属训练侧优化，不影响推理管线架构 |
| 多头自注意力（Multi-Head Self-Attention） | 不做 | 参数量与延迟超标，不适用 RK3588 |
| Transformer 架构替换 LSTM | 不做 | 24 步短序列场景下 LSTM 已足够 |
| 在线 VMD K 值自适应调整 | 不做 | K 值由训练阶段确定后固定 |
| CEEMDAN 替换 VMD | 备选 | VMD 优先，若光伏预测提升不足则切换 |

---

**文档状态**：[REVIEWED: PASS] — 2026-06-21 需求评审通过，7 章 43 条全部确认
