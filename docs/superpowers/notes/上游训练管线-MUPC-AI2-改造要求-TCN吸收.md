# 上游训练管线（MUPC-AI2）改造要求 — TCN 特征提取层吸收

> **来源**：孙炎《光伏高渗透率配电台区储能优化配置与调度》(2025) — TCN-IECA-BiLSTM 光伏预测
> **吸收级别**：R2 可选增强
> **MUPC 推理侧影响**：零改动（TCN 嵌入 ONNX 计算图，RKNN Toolkit 2 原生支持 Conv1D/BN/ReLU/残差）
> **日期**：2026-06-22

---

## 1. 吸收方法

TCN（Temporal Convolutional Network，时域卷积网络）作为 LSTM 之前的前置特征提取层：

```
原架构: VMD分解 → LSTM → Attention → 6头Linear
新架构: VMD分解 → TCN → LSTM → Attention → 6头Linear
                            ↑
                       本次插入位置
```

### 1.1 TCN 层规格

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 层数 (num_levels) | 4 | 膨胀率 [1, 2, 4, 8]，覆盖 15 步感受野 |
| 卷积核 (kernel_size) | 3 | 1D 因果卷积 |
| 滤波器数 (filters) | 64 | 与 LSTM hidden_size 一致 |
| 激活函数 | ReLU | ONNX 标准算子 |
| 归一化 | BatchNorm | ONNX 标准算子 |
| 残差连接 | 是 | 每层 Conv → BN → ReLU → Dropout → Add(x) |

### 1.2 ONNX 计算图结构

```
Input (B, T, K)
  ↓
TCN Block 0: Conv1D(k=3, d=1) → BN → ReLU → Dropout → +Input
  ↓
TCN Block 1: Conv1D(k=3, d=2) → BN → ReLU → Dropout → +Block0
  ↓
TCN Block 2: Conv1D(k=3, d=4) → BN → ReLU → Dropout → +Block1
  ↓
TCN Block 3: Conv1D(k=3, d=8) → BN → ReLU → Dropout → +Block2
  ↓
Output (B, T, 64) → 送入 LSTM
```

**关键约束：**
- 因果卷积（左侧 padding=kernel_size-1，右侧无 padding）—— 保证不依赖未来信息
- 所有算子（Conv1D、BatchNorm、ReLU、Add）必须是 ONNX opset 13 标准算子
- 膨胀卷积的 dilation 参数需在 ONNX Conv 节点中显式设置

---

## 2. 训练脚本改造

### 2.1 模型定义

在 `lstm_model.py` 中新增 `TCNBlock` 类和 `TCNFeatureExtractor` 类：

```python
class TCNBlock(nn.Module):
    """单层 TCN 残差块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation,
                              padding=(kernel_size - 1) * dilation)  # 因果padding
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        out = self.relu(self.bn(self.conv(x.permute(0, 2, 1))))
        out = out.permute(0, 2, 1)  # (B, C, T) → (B, T, C)
        out = out[:, :x.size(1), :]  # 截断因果padding
        return self.dropout(out) + self.residual(x)


class TCNFeatureExtractor(nn.Module):
    """4 层 TCN 特征提取器"""
    def __init__(self, input_dim, hidden_dim=64, num_levels=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            self.blocks.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

    def forward(self, x):
        # x: (B, T, K) → (B, T, hidden_dim)
        for block in self.blocks:
            x = block(x)
        return x
```

### 2.2 LSTMForecast 集成

在 `LSTMForecast.__init__` 中新增 `with_tcn: bool = False` 参数：

```python
class LSTMForecast:
    def __init__(self, ..., with_tcn: bool = False):
        ...
        self.tcn = TCNFeatureExtractor(input_dim, hidden_dim) if with_tcn else None
        # LSTM 输入维度：若使用 TCN 则为 hidden_dim，否则为 input_dim
        lstm_input_dim = hidden_dim if with_tcn else input_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, ...)
```

### 2.3 forward 流程

```python
def forward(self, x):
    if self.tcn is not None:
        x = self.tcn(x)  # TCN 特征提取
    h_seq, (h_n, _) = self.lstm(x)
    # ... 后续 Attention + 6头 Linear 不变
```

---

## 3. ONNX 导出改造

### 3.1 导出参数

在 `export_onnx.py` 中新增 `--with-tcn` 参数：

```bash
python export_onnx.py --lstm lstm_checkpoint.pt --with-attention --with-tcn
```

### 3.2 metadata_props 扩展

在现有 10 键基础上追加 1 键：

| 键 | 值 | 说明 |
|----|-----|------|
| `mupc_with_tcn` | `"true"` / `"false"` | 是否含 TCN 特征提取层 |

MUPC 推理端在加载模型时读取此元数据用于日志/审计，不做校验（TCN 对推理端透明）。

---

## 4. 测试与验证

### 4.1 消融实验

| 配置 | 说明 |
|------|------|
| LSTM+Attention（基线） | 当前 R1 架构 |
| TCN+LSTM+Attention | 本次新增 |
| TCN+LSTM（无 Attention） | 验证 TCN 独立增益 |

### 4.2 精度目标

| 指标 | 目标 | 测量方法 |
|------|------|----------|
| 光伏 MAPE 改善（vs 基线） | 降低 ≥ 2%（VMD 路径）/ ≥ 5%（非 VMD 路径） | 测试集回测 |
| 非 VMD 路径（Level 3/4）MAPE | 与 VMD 路径（Level 2）差距 ≤ 15% | 消融实验 |

### 4.3 性能验证

| 指标 | 要求 |
|------|------|
| TCN 层参数量 | ≤ 100K（INT8 ≤ 0.5MB） |
| 端到端推理延迟增加 | ≤ 20ms（vs 无 TCN 基线） |
| RKNN 转换成功率 | 100%（Conv1D/BN/ReLU/Add 均为标准算子） |

---

## 5. 非目标

| 项 | 状态 | 理由 |
|----|------|------|
| IECA Attention | 不做 | 论文的改进通道注意力，MUPC 已有 Additive Attention |
| MHSA 多头自注意力 | 不做 | PRD 已标记：参数和延迟超标，不适用于 RK3588 |
| BiGRU 替换 BiLSTM | 不做 | BiLSTM 已完成，BiGRU 收益有限（仅节省 25% 参数） |
| FCM 相似日聚类 | 不做 | 数据预处理优化，与 TCN 正交，可独立评估 |
| NRBO 超参优化 | 不做 | MSSA 已覆盖，NRBO 无增量优势 |
| TCN 替换 LSTM | 不做 | TCN 作为 LSTM 的前置提取器，不替代 LSTM 的时序建模 |

---

**文档状态**：待 MUPC-AI2 训练管线团队评审
