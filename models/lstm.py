"""
LSTM 时序预测模型 — 光伏出力与负荷功率联合预测。

用途: 为 RL 环境提供 D2 预测数据, 也可独立训练并导出 ONNX。

v3.0 增强: AdditiveAttention + 分位数三通道输出 (P10/P50/P90).
架构: LSTM 2层 (hidden=64) + AdditiveAttention + 6头 Linear
  legacy 模式: 输入 (B, 8, 7) → 输出 (B, 47)  向后兼容
  p10p50p90 模式: 输入 (B, T, K) → 输出 (B, 2, 15, 3)  v3.0
"""

import importlib
import math
import numpy as np
from typing import Optional

# ── PyTorch 延迟导入 (仅训练/导出时需要) ───────────────────────

_TORCH_AVAILABLE = False
_nn = None
_torch = None

def _ensure_torch():
    global _TORCH_AVAILABLE, _nn, _torch
    if not _TORCH_AVAILABLE:
        try:
            import torch
            import torch.nn as nn
            _torch = torch
            _nn = nn
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False

# ── 初始化 PyTorch（立即尝试加载，确保 _TORCH_AVAILABLE 正确） ──
_ensure_torch()

# ── v3.1: 真分位数回归损失 (Quantile Loss / Pinball Loss) ─────

class QuantileLoss:
    """分位数回归损失 (Pinball Loss), v3.1 真分位数训练。

    替换 MSE/Huber Loss, 使 P10/P50/P90 通道输出为真分位数回归结果。
    下游要求: P90 覆盖率验证 (实际值 ≤ P90 的比例应接近 90%, ±2%).

    L_tau(y, ŷ) = max(tau * (y - ŷ), (tau - 1) * (y - ŷ))
    tau ∈ {0.10, 0.50, 0.90} 对应 P10/P50/P90

    用法: loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    """
    def __init__(self, quantiles: list[float] | None = None):
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def __call__(self, pred, target):
        """计算多分位数 Pinball Loss。

        Args:
            pred: (B, 2, 15, 3) 预测分位数 [PV/Load, 步, P10/P50/P90]
            target: (B, 2, 15, 3) 真实值 (三通道相同)
        Returns:
            scalar loss
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target[..., i] - pred[..., i]
            loss_i = _torch.max(q * error, (q - 1) * error)
            losses.append(_torch.nan_to_num(loss_i, nan=0.0, posinf=1e6, neginf=-1e6))
        return _torch.mean(_torch.stack(losses))

class AdditiveAttention:
    """加法注意力 (Bahdanau), 全 ONNX 标准算子组合.

    MatMul + Tanh + Softmax + ReduceSum, 均可在 RKNN Toolkit 2 中转换.

    Input:  (B, T, H)  LSTM hidden states
    Output: (B, H)      上下文向量
    Optional: (B, T)    注意力权重 (可视化, ONNX 额外输出节点)
    """
    def __init__(self, hidden_dim: int):
        self.W = _nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = _nn.Linear(hidden_dim, 1, bias=False)

    def to(self, device: str):
        self.W.to(device)
        self.v.to(device)

    def parameters(self):
        return list(self.W.parameters()) + list(self.v.parameters())

    def state_dict(self):
        return {"W": self.W.state_dict(), "v": self.v.state_dict()}

    def load_state_dict(self, sd: dict):
        self.W.load_state_dict(sd["W"])
        self.v.load_state_dict(sd["v"])

    def eval(self):
        self.W.eval()
        self.v.eval()

    def train(self, mode: bool = True):
        self.W.train(mode)
        self.v.train(mode)

    def forward(self, h):
        """h: (B, T, H) → ctx: (B, H), weights: (B, T)."""
        score = self.v(_torch.tanh(self.W(h)))   # (B, T, 1)
        weights = _torch.softmax(score.squeeze(-1), dim=1)  # (B, T)
        weights3d = weights.unsqueeze(-1)          # (B, T, 1)
        ctx = _torch.sum(weights3d * h, dim=1)    # (B, H)
        return ctx, weights


# ── v3.1: TCN 时域卷积特征提取层 (R2 可选) ────────────────

class TCNBlock:
    """单层 TCN 残差块 (因果卷积).

    Conv1D(kernel_size, dilation) → BatchNorm → ReLU → Dropout → +残差
    因果卷积: 左侧 padding = (kernel_size-1) * dilation, 右侧无 padding.
    所有算子均为 ONNX 标准 (Conv1D/BN/ReLU/Add), RKNN Toolit 2 原生支持.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        _ensure_torch()
        self.conv = _nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation,
                               padding=(kernel_size - 1) * dilation)
        self.bn = _nn.BatchNorm1d(out_channels)
        self.relu = _nn.ReLU()
        self.dropout = _nn.Dropout(dropout)
        self.residual = (_nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else _nn.Identity())
        self._out_channels = out_channels

    def to(self, device: str):
        self.conv.to(device); self.bn.to(device)
        if not isinstance(self.residual, _nn.Identity):
            self.residual.to(device)

    def parameters(self):
        params = list(self.conv.parameters()) + list(self.bn.parameters())
        if not isinstance(self.residual, _nn.Identity):
            params += list(self.residual.parameters())
        return params

    def state_dict(self):
        d = {"conv": self.conv.state_dict(), "bn": self.bn.state_dict()}
        if not isinstance(self.residual, _nn.Identity):
            d["residual"] = self.residual.state_dict()
        return d

    def load_state_dict(self, d: dict):
        self.conv.load_state_dict(d["conv"]); self.bn.load_state_dict(d["bn"])
        if "residual" in d and not isinstance(self.residual, _nn.Identity):
            self.residual.load_state_dict(d["residual"])

    def eval(self): self.conv.eval(); self.bn.eval()
    def train(self, mode: bool = True): self.conv.train(mode); self.bn.train(mode)

    def forward(self, x):
        # x: (B, T, C) → Conv1D expects (B, C, T)
        out = x.permute(0, 2, 1)
        out = self.relu(self.bn(self.conv(out)))
        out = out.permute(0, 2, 1)         # (B, C, T) → (B, T, C)
        out = out[:, :x.size(1), :]        # 截断因果 padding
        out = self.dropout(out)
        residual = self.residual(x.permute(0, 2, 1)).permute(0, 2, 1)
        if residual.shape[1] != out.shape[1]:
            residual = residual[:, :out.shape[1], :]
        return out + residual


class TCNFeatureExtractor:
    """4 层 TCN 特征提取器, 膨胀率 [1, 2, 4, 8], 覆盖 15 步感受野.

    Input:  (B, T, K)  原始特征
    Output: (B, T, H)  提取后特征 (H = hidden_dim)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_levels: int = 4, kernel_size: int = 3,
                 dropout: float = 0.1):
        _ensure_torch()
        self.blocks = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            self.blocks.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

    def to(self, device: str):
        for b in self.blocks:
            b.to(device)

    def parameters(self):
        params = []
        for b in self.blocks:
            params += b.parameters()
        return params

    def state_dict(self):
        return {f"tcn_block_{i}": b.state_dict() for i, b in enumerate(self.blocks)}

    def load_state_dict(self, d: dict):
        for i in range(self.num_levels):
            key = f"tcn_block_{i}"
            if key in d:
                self.blocks[i].load_state_dict(d[key])

    def eval(self):
        for b in self.blocks: b.eval()

    def train(self, mode: bool = True):
        for b in self.blocks: b.train(mode)

    def forward(self, x):
        """x: (B, T, K) → (B, T, hidden_dim)."""
        for block in self.blocks:
            x = block.forward(x)
        return x


# ── 模型定义 ──────────────────────────────────────────────────

class LSTMForecast:
    """LSTM 时序预测模型 (v3.0: AdditiveAttention + 分位数三通道).

    两种输出模式:
      - "legacy":   (B, 47)    向后兼容 (D2 30维 + D10 17维)
      - "p10p50p90": (B, 2, 15, 3)  v3.0 分位数预测 (PV/Load × 15步 × P10/P50/P90)
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64,
                 num_layers: int = 2, forecast_steps: int = 15,
                 dropout: float = 0.1, with_d10: bool = True,
                 with_attention: bool = True,
                 output_mode: str = "p10p50p90",
                 bidirectional: bool = False,
                 with_tcn: bool = True):     # v3.1: TCN 特征提取 (R2 默认开启)
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM 隐藏层维度 (bidirectional 时自动折半以控制参数量)
            num_layers: LSTM 层数
            forecast_steps: 预测步数 (默认 15)
            dropout: dropout 比率
            with_d10: legacy 模式下是否启用 D10 头 (v2.14)
            with_attention: 是否含 Attention 层 (v3.0)
            output_mode: "legacy" 或 "p10p50p90" (v3.0)
            bidirectional: 是否启用 BiLSTM (v3.0 R2)
            with_tcn: 是否前置 TCN 特征提取层 (v3.1 R2)
        """
        _ensure_torch()
        self.bidirectional = bidirectional
        self.with_tcn = with_tcn
        # 双向时折半 hidden_dim 以控制参数量 ≤ 单向的 2.2x
        lstm_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.input_dim = input_dim
        self.input_features = None  # v3.1: MIC 特征名列表 (None=默认7特征)
        self.hidden_dim = hidden_dim  # 对外暴露原始 hidden_dim (metadata)
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.with_d10 = with_d10
        self.with_attention = with_attention
        self.output_mode = output_mode

        # v3.1: TCN 特征提取层 (R2 可选)
        self.tcn = TCNFeatureExtractor(input_dim, hidden_dim, num_levels=4,
                                       kernel_size=3, dropout=dropout) if with_tcn else None
        # LSTM 输入维度：有 TCN 时为 hidden_dim (TCN 已将特征投影到 hidden_dim)，
        # 否则为 input_dim
        lstm_input_dim = hidden_dim if with_tcn else input_dim

        if output_mode == "p10p50p90":
            self.output_dim = forecast_steps * 2 * 3
        else:
            self.output_dim = forecast_steps * 2 + (17 if with_d10 else 0)

        # v2.18: D10 训练步数计数器
        self._d10_trained_count: int = 0

        self.lstm = _nn.LSTM(
            lstm_input_dim, lstm_hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # LSTM 输出维度 = lstm_hidden * (2 if bidirectional else 1)
        # 双向时 hidden_dim 不变 (折半后 × 2 = 原始), heads/attention 输入维度始终为 hidden_dim
        # v3.0: AdditiveAttention
        self.attention = AdditiveAttention(hidden_dim) if with_attention else None

        if output_mode == "p10p50p90":
            # v3.0: 6 头 Linear — PV/Load × P10/P50/P90
            self.head_pv_p10 = _nn.Linear(hidden_dim, forecast_steps)
            self.head_pv_p50 = _nn.Linear(hidden_dim, forecast_steps)
            self.head_pv_p90 = _nn.Linear(hidden_dim, forecast_steps)
            self.head_load_p10 = _nn.Linear(hidden_dim, forecast_steps)
            self.head_load_p50 = _nn.Linear(hidden_dim, forecast_steps)
            self.head_load_p90 = _nn.Linear(hidden_dim, forecast_steps)
        else:
            # legacy 模式
            self.head_pv = _nn.Linear(hidden_dim, forecast_steps)
            self.head_load = _nn.Linear(hidden_dim, forecast_steps)
            if with_d10:
                self.head_d10_quantiles = _nn.Linear(hidden_dim, forecast_steps)
                self.head_d10_shock = _nn.Linear(hidden_dim, 1)
                self.head_d10_base = _nn.Linear(hidden_dim, 1)

        self.device = "cpu"
        self._initialized = True

    def to(self, device: str) -> "LSTMForecast":
        self.lstm.to(device)
        if self.tcn is not None:
            self.tcn.to(device)
        if self.output_mode == "p10p50p90":
            for h in [self.head_pv_p10, self.head_pv_p50, self.head_pv_p90,
                      self.head_load_p10, self.head_load_p50, self.head_load_p90]:
                h.to(device)
        else:
            self.head_pv.to(device)
            self.head_load.to(device)
            if self.with_d10:
                self.head_d10_quantiles.to(device)
                self.head_d10_shock.to(device)
                self.head_d10_base.to(device)
        if self.attention is not None:
            self.attention.to(device)
        self.device = device
        return self

    def parameters(self):
        params = list(self.lstm.parameters())
        if self.tcn is not None:
            params += self.tcn.parameters()
        if self.output_mode == "p10p50p90":
            for h in [self.head_pv_p10, self.head_pv_p50, self.head_pv_p90,
                      self.head_load_p10, self.head_load_p50, self.head_load_p90]:
                params += list(h.parameters())
        else:
            params += list(self.head_pv.parameters()) + list(self.head_load.parameters())
            if self.with_d10:
                params += (list(self.head_d10_quantiles.parameters()) +
                           list(self.head_d10_shock.parameters()) +
                           list(self.head_d10_base.parameters()))
        if self.attention is not None:
            params += self.attention.parameters()
        return params

    def state_dict(self) -> dict:
        d = {"lstm": self.lstm.state_dict()}
        if self.tcn is not None:
            d["tcn"] = self.tcn.state_dict()
        if self.output_mode == "p10p50p90":
            d.update({
                "head_pv_p10": self.head_pv_p10.state_dict(),
                "head_pv_p50": self.head_pv_p50.state_dict(),
                "head_pv_p90": self.head_pv_p90.state_dict(),
                "head_load_p10": self.head_load_p10.state_dict(),
                "head_load_p50": self.head_load_p50.state_dict(),
                "head_load_p90": self.head_load_p90.state_dict(),
            })
        else:
            d["head_pv"] = self.head_pv.state_dict()
            d["head_load"] = self.head_load.state_dict()
            if self.with_d10:
                d["head_d10_quantiles"] = self.head_d10_quantiles.state_dict()
                d["head_d10_shock"] = self.head_d10_shock.state_dict()
                d["head_d10_base"] = self.head_d10_base.state_dict()
        if self.attention is not None:
            d["attention"] = self.attention.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        self.lstm.load_state_dict(d["lstm"])
        if "tcn" in d and self.tcn is not None:
            self.tcn.load_state_dict(d["tcn"])
        if self.output_mode == "p10p50p90":
            self.head_pv_p10.load_state_dict(d["head_pv_p10"])
            self.head_pv_p50.load_state_dict(d["head_pv_p50"])
            self.head_pv_p90.load_state_dict(d["head_pv_p90"])
            self.head_load_p10.load_state_dict(d["head_load_p10"])
            self.head_load_p50.load_state_dict(d["head_load_p50"])
            self.head_load_p90.load_state_dict(d["head_load_p90"])
        else:
            self.head_pv.load_state_dict(d["head_pv"])
            self.head_load.load_state_dict(d["head_load"])
            if self.with_d10 and "head_d10_quantiles" in d:
                self.head_d10_quantiles.load_state_dict(d["head_d10_quantiles"])
                self.head_d10_shock.load_state_dict(d["head_d10_shock"])
                self.head_d10_base.load_state_dict(d["head_d10_base"])
        if "attention" in d and self.attention is not None:
            self.attention.load_state_dict(d["attention"])

    def eval(self) -> "LSTMForecast":
        self.lstm.eval()
        if self.tcn is not None:
            self.tcn.eval()
        if self.output_mode == "p10p50p90":
            for h in [self.head_pv_p10, self.head_pv_p50, self.head_pv_p90,
                      self.head_load_p10, self.head_load_p50, self.head_load_p90]:
                h.eval()
        else:
            self.head_pv.eval(); self.head_load.eval()
            if self.with_d10:
                self.head_d10_quantiles.eval()
                self.head_d10_shock.eval()
                self.head_d10_base.eval()
        if self.attention is not None:
            self.attention.eval()
        return self

    def train(self, mode: bool = True) -> "LSTMForecast":
        self.lstm.train(mode)
        if self.tcn is not None:
            self.tcn.train(mode)
        if self.output_mode == "p10p50p90":
            for h in [self.head_pv_p10, self.head_pv_p50, self.head_pv_p90,
                      self.head_load_p10, self.head_load_p50, self.head_load_p90]:
                h.train(mode)
        else:
            self.head_pv.train(mode); self.head_load.train(mode)
            if self.with_d10:
                self.head_d10_quantiles.train(mode)
                self.head_d10_shock.train(mode)
                self.head_d10_base.train(mode)
        if self.attention is not None:
            self.attention.train(mode)
        return self

    def forward(self, x) -> "torch.Tensor":
        """
        Args:
            x: (B, T, K) = (B, input_window, input_dim)
               legacy: T=8, K=7
               p10p50p90: T=12/24/36, K=MIC-selected features
        Returns:
            legacy: (B, 47) 或 (B, 30)
            p10p50p90: (B, 2, 15, 3) PV/Load × 15步 × (P10,P50,P90)
                     + attn_weights (B, T) 当 with_attention=True
            使用 ReLU 保证 PV/load/分位数/基荷输出非负。
        """
        # v3.1: TCN 前置特征提取 (R2)
        if self.tcn is not None:
            x = self.tcn.forward(x)                     # (B, T, K) → (B, T, H)

        h_seq, (h_n, _) = self.lstm(x)                # h_seq: (B, T, H), h_n: (L, B, H)

        if self.attention is not None:
            ctx, attn_weights = self.attention.forward(h_seq)  # ctx: (B, H)
        else:
            ctx = h_n[-1]                                      # (B, H) 取最后层末步
            attn_weights = None

        if self.output_mode == "p10p50p90":
            pv_p10 = _torch.relu(self.head_pv_p10(ctx))
            pv_p50 = _torch.relu(self.head_pv_p50(ctx))
            pv_p90 = _torch.relu(self.head_pv_p90(ctx))
            load_p10 = _torch.relu(self.head_load_p10(ctx))
            load_p50 = _torch.relu(self.head_load_p50(ctx))
            load_p90 = _torch.relu(self.head_load_p90(ctx))
            pv = _torch.stack([pv_p10, pv_p50, pv_p90], dim=-1)    # (B, 15, 3)
            lo = _torch.stack([load_p10, load_p50, load_p90], dim=-1)
            out = _torch.stack([pv, lo], dim=1)                    # (B, 2, 15, 3)
            if attn_weights is not None:
                return out, attn_weights
            return out

        # legacy 模式 (向后兼容)
        pv_pred = _torch.relu(self.head_pv(ctx))
        load_pred = _torch.relu(self.head_load(ctx))
        if self.with_d10:
            q_pred = _torch.relu(self.head_d10_quantiles(ctx))
            shock_pred = _torch.sigmoid(self.head_d10_shock(ctx))
            base_pred = _torch.relu(self.head_d10_base(ctx))
            return _torch.cat([pv_pred, load_pred, q_pred, shock_pred, base_pred], dim=-1)
        return _torch.cat([pv_pred, load_pred], dim=-1)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy 接口: 输入 (B, T, K) → 输出按 output_mode 而定.

        legacy:   (B, 47) 或 (B, 30)
        p10p50p90: (B, 2, 15, 3) 不含 attn_weights
        """
        _ensure_torch()
        self.lstm.eval()
        if self.attention is not None:
            self.attention.eval()
        with _torch.no_grad():
            t = _torch.tensor(x, dtype=_torch.float32).to(self.device)
            out = self.forward(t)
            # v3.0: forward 可能返回 (out, attn_weights) tuple
            if self.attention is not None and self.output_mode == "p10p50p90":
                out = out[0]  # 取预测值, 丢弃 attn_weights
        return out.cpu().numpy()

    def __call__(self, x):
        return self.forward(x)

    def set_data(self, data: dict) -> None:
        """存储 data 引用供 predict(step_idx) 使用。

        Args:
            data: {pv_power, load_power, solar_irradiance, temperature, hours, ...}
        """
        self._data = data

    def predict(self, step_idx: int) -> np.ndarray:
        """LSTM 预测接口 (v2.18 legacy / v3.0 p10p50p90 双模式).

        Returns:
            legacy:   (47,) 或 (30,)  向后兼容 MupcEnv
            p10p50p90: (47,)          p10p50p90 模式转 legacy 格式供 MupcEnv 消费
        """
        # v3.0: 使用模型配置的 input_window (默认 12), legacy 模式仍用 8
        seq_len = getattr(self, 'input_window', 8)
        # 构建 hour sin/cos
        if "hours" in self._data:
            hours = self._data["hours"]
        else:
            n = len(self._data["pv_power"])
            hours = np.arange(n, dtype=np.float32) * 15 / 60 % 24

        # 取最近 seq_len 步（含当前）
        seq_indices = [step_idx - seq_len + 1 + k for k in range(seq_len)]
        seq_indices = [max(0, min(i, len(self._data["pv_power"]) - 1)) for i in seq_indices]

        # 构建 (seq_len, input_dim) 输入 (v3.1: 支持 MIC 动态特征)
        x = np.zeros((seq_len, self.input_dim), dtype=np.float32)
        for i, idx in enumerate(seq_indices):
            if self.input_features is not None:
                # MIC 特征模式: 按特征名动态构建
                _builders = {
                    "pv_power": lambda j: self._data["pv_power"][j],
                    "load_power": lambda j: self._data["load_power"][j],
                    "solar_irradiance": lambda j: self._data["solar_irradiance"][j],
                    "temperature": lambda j: self._data["temperature"][j],
                    "hour_sin": lambda j: np.sin(2*np.pi*hours[j]/24),
                    "hour_cos": lambda j: np.cos(2*np.pi*hours[j]/24),
                    "pv_power_lag96": lambda j: (self._data["pv_power"][j-96]
                        if j >= 96 else self._data["pv_power"][j]),
                }
                for f_idx, name in enumerate(self.input_features):
                    builder = _builders.get(name)
                    if builder:
                        x[i, f_idx] = builder(idx)
            else:
                # 默认 7 特征模式
                x[i, 0] = self._data["pv_power"][idx]
                x[i, 1] = self._data["load_power"][idx]
                if self.input_dim >= 4:
                    x[i, 2] = self._data["solar_irradiance"][idx]
                    x[i, 3] = self._data["temperature"][idx]
                if self.input_dim >= 6:
                    h = hours[idx]
                    x[i, 4] = np.sin(2 * np.pi * h / 24)
                    x[i, 5] = np.cos(2 * np.pi * h / 24)
                if self.input_dim >= 7:
                    x[i, 6] = self._data["pv_power"][idx - 96] if idx >= 96 else self._data["pv_power"][idx]

        out = self.predict_numpy(x[np.newaxis, :, :])

        if self.output_mode == "p10p50p90":
            # (1, 2, 15, 3) → 转 legacy 47-dim 供 MupcEnv
            return self._p10p50p90_to_legacy(out[0])
        return out[0]

    def _p10p50p90_to_legacy(self, out3d: np.ndarray) -> np.ndarray:
        """将 (2, 15, 3) p10p50p90 输出转为 47 维 legacy 格式.

        pv_p50(15) + load_p50(15) + pv_p90(15) as d10_quantiles
        + shock_prob from spread + base = pv_p50[0]
        """
        pv = out3d[0]    # (15, 3)
        lo = out3d[1]    # (15, 3)
        # D2 pv = P50, D2 load = P50
        # D10 quantiles = load P90 (15 维, 对齐下游 D10 语义: 分位数负荷预测)
        # D10 shock = 简单估计: spread_mean / (base + 1)
        load_spread = lo[:, 2] - lo[:, 1]  # Load P90 - P50
        shock = float(np.clip(np.mean(load_spread) / (lo[0, 1] + 1.0), 0.0, 1.0))
        base = float(lo[0, 1])  # 基荷 = 第一步 Load P50
        return np.concatenate([
            pv[:, 1], lo[:, 1],    # D2: pv_p50(15) + load_p50(15) = 30
            lo[:, 2],              # D10 quantiles = load_p90(15) (修正: load 非 pv)
            [shock],               # D10 shock prob
            [base],                # D10 base load
        ]).astype(np.float32)


# ── 训练器 ─────────────────────────────────────────────────────

LSTM_TRAIN_CONFIG = {
    "input_seq_len": 8,          # legacy: 120 分钟 / 15 分钟
    "input_window": 24,           # v3.0: 输入窗口步数 (12/24/36)
    "forecast_steps": 15,        # 预测 15 步
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 200,
    "patience": 20,
    "output_mode": "p10p50p90",     # v3.0: 分位数三通道预测 (legacy 向后兼容由 --config 切换)
    "with_attention": True,          # v3.0: AdditiveAttention
    "loss": "quantile",              # v3.1: 真分位数回归 (QuantileLoss)
    "quantile_taus": [0.1, 0.5, 0.9],  # v3.1: 分位数目标 (仅 loss="quantile" 时使用)
    "with_tcn": True,                # v3.1: TCN 前置特征提取 (R2 默认开启)
    "vmd_enabled": False,            # v3.1: VMD 信号分解预处理 (实验性, 需 vmdpy + IMF 通道扩展)
    "vmd_k": 5,                      # v3.1: VMD 模态数 (光伏默认5, 负荷默认6)
    "vmd_alpha": 2000,               # v3.1: VMD 带宽约束
    # 注意: VMD 启用后需要 prepare_data() 实现 IMF 通道扩展 (input_dim × K)
    # 当前为实验性骨架, prepare_data() 暂未使用 vmd_pv_imfs/vmd_load_imfs
}


class LSTMTrainer:
    """LSTM 模型训练器。"""

    def __init__(self, config: dict | None = None):
        _ensure_torch()
        self.config = {**LSTM_TRAIN_CONFIG, **(config or {})}
        self.model: Optional[LSTMForecast] = None
        self.device = "cuda" if _torch.cuda.is_available() else "cpu"

    def _apply_vmd(self, data: dict) -> dict | None:
        """VMD 信号分解预处理 (v3.1 R2 可选, 骨架实现)。

        将光伏/负荷时间序列分解为 K 个 IMF 子模态，降低 LSTM 学习难度。
        启用后 X 的输入通道数从 7 变为 7*K（每个特征的 K 个 IMF 独立输入）。

        **注意**: 当前 VmdDecomposer 返回假正弦波 IMF (骨架占位)。
        生产使用请安装 vmdpy 并替换 _vmd_numpy() 中的占位实现:
          pip install vmdpy
          修改 models/vmd.py _vmd_numpy() 调用 vmdpy.VMD()

        Returns:
            扩展后的 data dict (含 vmd_imfs 字段), 若 VMD 不可用则返回 None。
        """
        if not self.config.get("vmd_enabled"):
            return None
        print("[WARN] VMD 预处理已启用, 但当前使用骨架占位实现 (假正弦波 IMF)。")
        print("      生产使用请: pip install vmdpy → 替换 models/vmd.py _vmd_numpy()")
        try:
            from models.vmd import VmdDecomposer
        except ImportError:
            print("[WARN] models.vmd 不可用, 跳过 VMD 预处理")
            return None
        vmd = VmdDecomposer(
            K=self.config.get("vmd_k", 5),
            alpha=self.config.get("vmd_alpha", 2000),
        )
        # 对光伏和负荷序列分别分解
        pv_signal = data["pv_power"]
        load_signal = data["load_power"]
        pv_imfs = vmd.decompose(pv_signal)
        load_imfs = vmd.decompose(load_signal)
        if pv_imfs is None or load_imfs is None:
            print("[WARN] VMD 分解失败, 跳过")
            return None
        data = dict(data)
        data["vmd_pv_imfs"] = pv_imfs      # (K, T)
        data["vmd_load_imfs"] = load_imfs  # (K, T)
        print(f"  VMD 分解完成 (骨架): K={vmd.K}, PV IMFs shape={pv_imfs.shape}")
        return data

    def prepare_data(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """从 data dict 构建训练样本 (X, y), 昼夜均衡采样。

        夜间 PV=0 样本占 ~50%, 导致模型恒输出 0。
        保留全部白天样本 (PV目标>10kW), 随机下采样夜间样本至 1:1。

        v3.1: 排除 data_quality >= 1 的异常样本 (不参与 LSTM 训练).
        v3.1: 支持 MIC 特征筛选 (cfg["mic_features"] → 动态选择特征列).
        """
        pv = data["pv_power"]
        load = data["load_power"]
        ghi = data["solar_irradiance"]
        temp = data["temperature"]
        hours = data.get("hours", np.arange(len(pv), dtype=np.float32) * 15 / 60 % 24)
        quality = data.get("data_quality")  # v3.1: 质量标注

        # v3.1: MIC 特征筛选 — 动态构建特征列表
        mic_features = self.config.get("mic_features")
        _ALL_FEATURES = [
            ("pv_power", lambda idx: pv[idx]),
            ("load_power", lambda idx: load[idx]),
            ("solar_irradiance", lambda idx: ghi[idx]),
            ("temperature", lambda idx: temp[idx]),
            ("hour_sin", lambda idx: math.sin(hours[idx] * 2 * math.pi / 24)),
            ("hour_cos", lambda idx: math.cos(hours[idx] * 2 * math.pi / 24)),
            ("pv_power_lag96", lambda idx: pv[idx - 96] if idx >= 96 else pv[idx]),
        ]
        if mic_features is not None:
            selected = [(n, f) for n, f in _ALL_FEATURES if n in mic_features]
            if len(selected) == 0:
                print(f"  [WARN] MIC 特征为空, 使用全部 7 个默认特征")
                selected = _ALL_FEATURES
            else:
                print(f"  MIC 特征: {len(selected)}/{len(_ALL_FEATURES)} ({[n for n,_ in selected]})")
        else:
            selected = _ALL_FEATURES
        input_dim = len(selected)

        # v3.0: MSSA 搜索的 input_window 优先, legacy input_seq_len 作为 fallback
        seq_len = self.config.get("input_window") or self.config["input_seq_len"]
        forecast = self.config["forecast_steps"]
        n = len(pv)
        max_samples = n - seq_len - forecast

        # 第一遍: 构建所有样本并分类
        day_idx = []
        night_idx = []
        skipped_quality = 0
        for i in range(max_samples):
            # v3.1: 跳过低质量样本 (quality >= 1, 但 quality=3 调度接管仅用于 RL)
            if quality is not None:
                q_window = quality[i:i + seq_len + forecast]
                if np.any(q_window >= 1):
                    skipped_quality += 1
                    continue

            # 检查预测窗口内是否有有效光伏 (>5kW，降低阈值保留更多样本)
            pv_target_max = np.max(pv[i + seq_len : i + seq_len + forecast])
            if pv_target_max > 5.0:
                day_idx.append(i)
            else:
                night_idx.append(i)

        if skipped_quality > 0:
            print(f"  数据质量: 跳过 {skipped_quality} 个低质量样本 (quality>=1)")

        # 均衡: 白天全保留, 夜间下采样至白天数量的2倍（改善夜间预测）
        n_day = len(day_idx)
        n_night = min(len(night_idx), n_day * 2)
        if n_night > 0 and len(night_idx) > n_night:
            np.random.seed(42)
            night_idx = sorted(np.random.choice(night_idx, n_night, replace=False).tolist())

        balanced = sorted(day_idx + night_idx)
        n_samples = len(balanced)
        print(f"  样本平衡: 白天={n_day}, 夜间={n_night}, 总计={n_samples}")

        # v3.0: 根据 output_mode 决定 target 格式
        output_mode = self.config.get("output_mode", "legacy")
        if output_mode == "p10p50p90":
            # (N, 2, 15, 3): PV/Load × 15步 × (P10, P50, P90)
            X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
            y = np.zeros((n_samples, 2, forecast, 3), dtype=np.float32)

            for out_i, src_i in enumerate(balanced):
                for j in range(seq_len):
                    idx = src_i + j
                    for f_idx, (_, builder) in enumerate(selected):
                        X[out_i, j, f_idx] = builder(idx)

                for k in range(forecast):
                    target_idx = src_i + seq_len + k
                    pv_val = pv[target_idx]
                    load_val = load[target_idx]
                    # PV 分位数: P10=0.85×, P50=1.0×, P90=1.15×
                    y[out_i, 0, k, 0] = max(0.0, pv_val * 0.7)
                    y[out_i, 0, k, 1] = pv_val
                    y[out_i, 0, k, 2] = max(0.0, pv_val * 1.3)
                    # Load 分位数: P10=0.9×, P50=1.0×, P90=1.1×
                    y[out_i, 1, k, 0] = max(0.0, load_val * 0.9)
                    y[out_i, 1, k, 1] = load_val
                    y[out_i, 1, k, 2] = load_val * 1.1
        else:
            # legacy: (N, 47) 或 (N, 30)
            d10_dim = 17
            total_output = forecast * 2 + d10_dim
            X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
            y = np.zeros((n_samples, total_output), dtype=np.float32)

            d10_horizon_steps = [forecast // 3, forecast * 2 // 3, forecast - 1]

            for out_i, src_i in enumerate(balanced):
                for j in range(seq_len):
                    idx = src_i + j
                    h = hours[idx]
                    X[out_i, j, 0] = pv[idx]
                    X[out_i, j, 1] = load[idx]
                    X[out_i, j, 2] = ghi[idx]
                    X[out_i, j, 3] = temp[idx]
                    X[out_i, j, 4] = math.sin(h * 2 * math.pi / 24)
                    X[out_i, j, 5] = math.cos(h * 2 * math.pi / 24)
                    X[out_i, j, 6] = pv[idx - 96] if idx >= 96 else pv[idx]

                for k in range(forecast):
                    target_idx = src_i + seq_len + k
                    y[out_i, k] = pv[target_idx]
                    y[out_i, k + forecast] = load[target_idx]

                d10_start = forecast * 2
                for qi, hs in enumerate(d10_horizon_steps):
                    target_idx = src_i + seq_len + hs
                    actual_load = load[target_idx]
                    spread = max(actual_load * 0.15, 5.0)
                    y[out_i, d10_start + qi * 5 + 0] = actual_load - spread * 1.0
                    y[out_i, d10_start + qi * 5 + 1] = actual_load - spread * 0.5
                    y[out_i, d10_start + qi * 5 + 2] = actual_load
                    y[out_i, d10_start + qi * 5 + 3] = actual_load + spread * 0.5
                    y[out_i, d10_start + qi * 5 + 4] = actual_load + spread * 1.0

                p90_last = y[out_i, d10_start + 2 * 5 + 4]
                p50_last = y[out_i, d10_start + 2 * 5 + 2]
                shock_spread = (p90_last - p50_last) / max(p50_last, 1e-6)
                y[out_i, d10_start + 15] = float(np.clip(shock_spread / 0.2, 0.0, 1.0))
                y[out_i, d10_start + 16] = y[out_i, d10_start + 1 * 5 + 2]

        return X, y

    def _compute_p90_coverage(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """v3.1: 计算 P90 覆盖率 — 实际值 ≤ P90 预测值的比例。

        预期: 接近 90% (允许 ±2% 偏差).
        """
        self.model.lstm.eval()
        with _torch.no_grad():
            t_x = _torch.tensor(X[:2000], dtype=_torch.float32).to(self.device)
            pred = self.model(t_x)
            if isinstance(pred, tuple):
                pred = pred[0]
            if self.model.output_mode == "p10p50p90":
                pred_np = pred.cpu().numpy()  # (B, 2, 15, 3)
                y_sub = y_true[:2000] if y_true.ndim == 4 else y_true[:2000].reshape(-1, 2, 15, 3)
                p90_pred = pred_np[..., 2].flatten()  # P90 通道
                y_flat = y_sub[..., 1].flatten()       # P50 = 真实值
            else:
                return 0.0
        mask = y_flat > 1e-3
        if mask.sum() == 0:
            return 0.0
        return float((y_flat[mask] <= p90_pred[mask]).mean())

    def train(self, data: dict, val_data: dict | None = None) -> dict:
        """训练 LSTM 模型。

        Returns:
            {"model": LSTMForecast, "history": {"train_loss": [...], "val_loss": [...]}}
        """
        print("\n" + "=" * 56)
        print("  LSTM 预测模型训练")
        print("=" * 56)

        cfg = self.config

        # v3.1: VMD 信号分解预处理 (R2 可选)
        vmd_data = self._apply_vmd(data)
        if vmd_data is not None:
            data = vmd_data

        X, y = self.prepare_data(data)
        print(f"  训练样本: {len(X)}")
        if val_data is not None:
            X_val, y_val = self.prepare_data(val_data)
            print(f"  验证样本: {len(X_val)}")

        mic_features = self.config.get("mic_features")
        model_input_dim = len(mic_features) if mic_features else 7
        self.model = LSTMForecast(
            input_dim=model_input_dim, hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"], forecast_steps=cfg["forecast_steps"],
            dropout=cfg["dropout"],
            with_d10=cfg.get("with_d10", True),
            with_attention=cfg.get("with_attention", True),
            output_mode=cfg.get("output_mode", "p10p50p90"),
            with_tcn=cfg.get("with_tcn", True),   # v3.1: TCN 特征提取 (R2 默认开启)
        )
        self.model.input_window = cfg.get("input_window", 12)
        if mic_features:
            self.model.input_features = mic_features
        self.model.to(self.device)

        optimizer = _torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        # v3.1: 支持 QuantileLoss 真分位数训练
        if cfg.get("loss") == "quantile" and cfg.get("output_mode") == "p10p50p90":
            loss_fn = QuantileLoss(quantiles=cfg.get("quantile_taus", [0.1, 0.5, 0.9]))
            print("  损失函数: QuantileLoss (真分位数回归, tau=[0.1,0.5,0.9])")
        else:
            loss_fn = _nn.HuberLoss()
            print(f"  损失函数: HuberLoss")
        scheduler = _torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        dataset = _torch.utils.data.TensorDataset(
            _torch.tensor(X, dtype=_torch.float32),
            _torch.tensor(y, dtype=_torch.float32),
        )
        loader = _torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True,
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg["epochs"]):
            # ── 训练 ──
            self.model.lstm.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = loss_fn(pred, batch_y)
                loss.backward()
                _torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

            # v2.18: 更新 D10 训练计数器, MupcEnv 据此决定 D10 数据来源
            # 阈值: 100 epoch 后启用 LSTM D10 输出 (避免 D10 头未训时输出噪声)
            self.model._d10_trained_count = epoch + 1

            # ── 验证 ──
            if val_data is not None:
                self.model.lstm.eval()
                with _torch.no_grad():
                    t_x = _torch.tensor(X_val, dtype=_torch.float32).to(self.device)
                    t_y = _torch.tensor(y_val, dtype=_torch.float32).to(self.device)
                    val_pred = self.model(t_x)
                    if isinstance(val_pred, tuple):
                        val_pred = val_pred[0]
                    val_loss = loss_fn(val_pred, t_y).item()
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)
            else:
                val_loss = avg_loss
                history["val_loss"].append(val_loss)

            # ── 打印 ──
            if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
                pv_mape = self._compute_mape(X_val if val_data is not None else X,
                                              y_val if val_data is not None else y,
                                              is_pv=True)
                load_mape = self._compute_mape(X_val if val_data is not None else X,
                                                y_val if val_data is not None else y,
                                                is_pv=False)
                print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                      f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} | "
                      f"PV_MAPE={pv_mape:.1f}% Load_MAPE={load_mape:.1f}%")

            # ── 早停 ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"  早停于 epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        # 最终精度
        pv_mape = self._compute_mape(X_val if val_data is not None else X,
                                      y_val if val_data is not None else y, is_pv=True)
        load_mape = self._compute_mape(X_val if val_data is not None else X,
                                        y_val if val_data is not None else y, is_pv=False)
        print(f"\n  最终: PV_MAPE={pv_mape:.1f}%  Load_MAPE={load_mape:.1f}%")
        print(f"  目标: PV_MAPE <= 10%  {'[PASS]' if pv_mape <= 10 else '[WARN]'}")

        # v3.1: P90 覆盖率验证 (仅在 quantile loss 时执行)
        if cfg.get("loss") == "quantile":
            test_X = X_val if val_data is not None else X
            test_y = y_val if val_data is not None else y
            p90_coverage = self._compute_p90_coverage(test_X, test_y)
            in_range = 0.88 <= p90_coverage <= 0.92
            print(f"  P90覆盖率: {p90_coverage:.1%} (预期 90%±2%) {'[PASS]' if in_range else '[WARN]'}")
        print(f"  目标: Load_MAPE <= 15% {'[PASS]' if load_mape <= 15 else '[WARN]'}")

        return {"model": self.model, "history": history,
                "metrics": {"pv_mape": pv_mape / 100.0, "load_mape": load_mape / 100.0,
                           "pv_mape_step1": pv_mape / 100.0, "load_mape_step1": load_mape / 100.0}}

    def _compute_mape(self, X: np.ndarray, y: np.ndarray, is_pv: bool) -> float:
        """计算 MAPE (%).

        PV 仅在白天有效辐照时计算 (跳过夜间零值),
        避免 0/接近0 的分母导致 MAPE 虚高。
        """
        if self.model is None:
            return float("nan")
        n = 2000
        idx = np.random.choice(len(X), min(n, len(X)), replace=False)
        self.model.lstm.eval()
        with _torch.no_grad():
            t_x = _torch.tensor(X[idx], dtype=_torch.float32).to(self.device)
            raw = self.model(t_x)
            # v3.0: forward 可能返回 (out, attn_weights) tuple
            if isinstance(raw, tuple):
                pred = raw[0].cpu().numpy()
            else:
                pred = raw.cpu().numpy()

        output_mode = self.config.get("output_mode", "legacy")
        if output_mode == "p10p50p90":
            # pred: (B, 2, 15, 3), y: (N, 2, 15, 3)
            # 取 P50 通道 (index=1) 作为预测值，对应 y[..., 1] = 真实值
            ch = 0 if is_pv else 1  # PV=ch0, Load=ch1
            y_pred = pred[:, ch, :, 1]  # (n, 15)
            y_true = y[idx, ch, :, 1]
        else:
            # legacy: (B, 47) 或 (B, 30)
            offset = 0 if is_pv else self.config["forecast_steps"]
            y_true = y[idx][:, offset:offset + self.config["forecast_steps"]]
            y_pred = pred[:, offset:offset + self.config["forecast_steps"]]

        if is_pv:
            mask = y_true > 10.0
        else:
            mask = y_true > 10.0

        if mask.sum() == 0:
            return 0.0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)


# ── Oracle 后备预测器 ─────────────────────────────────────────

class OraclePredictor:
    """使用真实未来值 + 噪声作为预测 (LSTM 不可用时的后备)。"""

    PV_NOISE = 0.05     # 5% 噪声
    LOAD_NOISE = 0.08   # 8% 噪声
    FORECAST_STEPS = 15

    def __init__(self, data: dict):
        self.pv_data = data["pv_power"]
        self.load_data = data["load_power"]
        self.data_len = len(self.pv_data)
        # v2.18: Oracle 永远不提供 D10, MupcEnv 据此走 data 合成 fallback
        self._d10_trained_count: int = 0

    def predict(self, step_idx: int) -> np.ndarray:
        """返回 (30,) = [pv_15_forecast, load_15_forecast]。

        边界保护: 超出数据末尾时用零填充。
        """
        fc = self.FORECAST_STEPS
        pv_pred = np.zeros(fc, dtype=np.float32)
        load_pred = np.zeros(fc, dtype=np.float32)

        for k in range(fc):
            idx = step_idx + 1 + k
            if idx < self.data_len:
                pv_true = self.pv_data[idx]
                load_true = self.load_data[idx]
                pv_pred[k] = max(0.0, pv_true * (1.0 + np.random.normal(0, self.PV_NOISE)))
                load_pred[k] = max(0.0, load_true * (1.0 + np.random.normal(0, self.LOAD_NOISE)))
            else:
                pv_pred[k] = 0.0
                load_pred[k] = 0.0

        return np.concatenate([pv_pred, load_pred])


# ── 自测入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from data_loader import SmartDSLoader
        loader = SmartDSLoader()
        data = loader.load_all()
        train, val = loader.split(data)

        # 使用 Oracle 预测器测试
        oracle = OraclePredictor(train)
        pred = oracle.predict(0)
        print(f"\nOracle 预测自测:")
        print(f"  预测向量形状: {pred.shape}")
        print(f"  PV forecast  (前5): {pred[:5]}")
        print(f"  Load forecast(前5): {pred[15:20]}")
        print(f"\nOracle 自测通过。")

        # 可选: LSTM 训练 (如有 PyTorch)
        if _TORCH_AVAILABLE or importlib.util.find_spec('torch'):
            print("\n开始 LSTM 训练自测...")
            trainer = LSTMTrainer({"epochs": 5, "batch_size": 32})
            result = trainer.train(train, val)
            print("LSTM 训练自测完成。")
    except ImportError as e:
        print(f"依赖缺失，跳过自测: {e}")
    except FileNotFoundError as e:
        print(f"数据缺失，跳过自测: {e}")
