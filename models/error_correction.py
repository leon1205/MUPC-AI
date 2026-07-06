"""
误差修正 BiLSTM 模型 (v3.0 R2).

用途: 训练独立轻量 BiLSTM, 修正主 LSTM 预测的系统性偏差.
      主模型 Bias > 3% MAPE 时启用; 否则可跳过训练.

架构: BiLSTM (hidden=32, num_layers=1) + Linear (2H→1)
输入: (B, residual_window, 1) 历史残差序列
输出: (B, 15, 1) 未来残差预测
"""

import math
import numpy as np
from pathlib import Path

# PyTorch 延迟导入
_TORCH = None
_nn = None

def _ensure_torch():
    global _TORCH, _nn
    if _TORCH is None:
        try:
            import torch
            import torch.nn as nn
            _TORCH = torch
            _nn = nn
        except ImportError:
            _TORCH = None


class ErrorCorrectionBiLSTM:
    """独立轻量 BiLSTM, 用于残差修正 (v3.0 R2).

    参数量约束: ≤ 3MB (≤ 主模型参数量的 50%).
    默认 hidden=32, num_layers=1 → ~50K 参数.
    """

    def __init__(self, hidden_dim: int = 32, num_layers: int = 1,
                 residual_window: int = 24, output_horizon: int = 15):
        _ensure_torch()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual_window = residual_window
        self.output_horizon = output_horizon

        self.bilstm = _nn.LSTM(
            1, hidden_dim, num_layers,
            bidirectional=True, batch_first=True,
        )
        self.head = _nn.Linear(hidden_dim * 2, 1)
        self.device = "cpu"

    def to(self, device: str):
        self.bilstm.to(device)
        self.head.to(device)
        self.device = device
        return self

    def parameters(self):
        return list(self.bilstm.parameters()) + list(self.head.parameters())

    def state_dict(self):
        return {"bilstm": self.bilstm.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, d: dict):
        self.bilstm.load_state_dict(d["bilstm"])
        self.head.load_state_dict(d["head"])

    def eval(self):
        self.bilstm.eval()
        self.head.eval()
        return self

    def train(self, mode: bool = True):
        self.bilstm.train(mode)
        self.head.train(mode)
        return self

    def forward(self, x):
        """x: (B, residual_window, 1) → out: (B, output_horizon, 1)."""
        h, _ = self.bilstm(x)                # (B, T, 2H)
        out = self.head(h[:, -self.output_horizon:])  # (B, 15, 1)
        return out.squeeze(-1)               # (B, 15)

    def predict(self, residual_history: np.ndarray) -> np.ndarray:
        """推理接口: 输入 (residual_window,) → 输出 (output_horizon,)."""
        _ensure_torch()
        self.eval()
        with _TORCH.no_grad():
            t = _TORCH.tensor(residual_history, dtype=_TORCH.float32)
            t = t.view(1, self.residual_window, 1).to(self.device)
            out = self.forward(t)
        return out.cpu().numpy().flatten()


# ═══════════════════════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════════════════════

ERROR_CORRECTION_CONFIG = {
    "hidden_dim": 32,
    "num_layers": 1,
    "residual_window": 24,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "epochs": 100,
    "patience": 15,
    "bias_threshold_pct": 3.0,   # Bias > 3% MAPE 才启用
}


class ErrorCorrectionTrainer:
    """误差修正 BiLSTM 训练器 (v3.0 R2)."""

    def __init__(self, config: dict | None = None):
        _ensure_torch()
        self.config = {**ERROR_CORRECTION_CONFIG, **(config or {})}
        self.model: ErrorCorrectionBiLSTM | None = None
        self.device = "cuda" if _TORCH.cuda.is_available() else "cpu"

    def train(self, data: dict,
              main_model_predict: callable) -> dict:
        """训练误差修正模型.

        Args:
            data: 训练数据 dict {pv_power, load_power, ...}
            main_model_predict: 主模型 predict_numpy(X)→(N, 2, 15, 3) 接口

        Returns:
            {"skip": bool, "bias": float, "model": ErrorCorrectionBiLSTM | None,
             "history": dict}
        """
        cfg = self.config
        print("\n" + "=" * 56)
        print("  误差修正 BiLSTM 训练 (v3.0 R2)")
        print("=" * 56)

        # Step 1: 用主模型对训练集做预测
        pv = data["pv_power"]
        load = data["load_power"]
        n = len(pv)
        forecast = 15
        rw = cfg["residual_window"]

        # 构建训练集 X (同 prepare_data 逻辑)
        X, _ = _build_prediction_input(data, rw, forecast)

        # 主模型推理
        print("  [1/5] 主模型推理...")
        main_pred = main_model_predict(X)  # (N, 2, 15, 3) or (N, 47)

        # 提取 P50
        if main_pred.ndim == 4:  # (N, 2, 15, 3)
            pv_p50 = main_pred[:, 0, :, 1]    # (N, 15)
            load_p50 = main_pred[:, 1, :, 1]  # (N, 15)
        else:  # legacy (N, 47)
            pv_p50 = main_pred[:, :15]
            load_p50 = main_pred[:, 15:30]

        # Step 2: 计算残差
        print("  [2/5] 计算残差序列...")
        pv_residuals = _compute_residuals(X, pv_p50, pv, rw, forecast)
        load_residuals = _compute_residuals(X, load_p50, load, rw, forecast)

        # Step 3: 偏差检查
        pv_bias = float(np.mean(np.abs(pv_residuals)))
        load_bias = float(np.mean(np.abs(load_residuals)))
        pv_mape_abs = pv_bias / max(np.mean(pv), 1e-6) * 100
        load_mape_abs = load_bias / max(np.mean(load), 1e-6) * 100

        print(f"    光伏 Bias={pv_bias:.3f} kW ({pv_mape_abs:.1f}% MAPE)")
        print(f"    负荷 Bias={load_bias:.3f} kW ({load_mape_abs:.1f}% MAPE)")

        if pv_mape_abs <= cfg["bias_threshold_pct"] and load_mape_abs <= cfg["bias_threshold_pct"]:
            print(f"    Bias <= {cfg['bias_threshold_pct']}%, 跳过误差修正训练")
            return {"skip": True, "bias_pv": pv_bias, "bias_load": load_bias,
                    "model": None, "history": {}}

        # Step 4: 准备残差训练数据
        print(f"  [3/5] 准备残差训练数据 (窗口={rw})...")
        target = "load" if load_mape_abs > pv_mape_abs else "pv"
        residuals = load_residuals if target == "load" else pv_residuals
        print(f"    训练目标: {target} (Bias 更大)")

        X_err, y_err = _prepare_residual_data(residuals, rw, forecast)
        if len(X_err) < 100:
            print(f"    残差样本不足 ({len(X_err)} < 100), 跳过")
            return {"skip": True, "bias_pv": pv_bias, "bias_load": load_bias,
                    "model": None, "history": {}}

        # Step 5: 训练 BiLSTM
        print(f"  [4/5] 训练误差修正 BiLSTM (hidden={cfg['hidden_dim']})...")
        self.model = ErrorCorrectionBiLSTM(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            residual_window=rw,
        ).to(self.device)

        optimizer = _TORCH.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        loss_fn = _nn.HuberLoss()
        scheduler = _TORCH.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        dataset = _TORCH.utils.data.TensorDataset(
            _TORCH.tensor(X_err, dtype=_TORCH.float32),
            _TORCH.tensor(y_err, dtype=_TORCH.float32),
        )
        loader = _TORCH.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": []}

        for epoch in range(cfg["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model.forward(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                _TORCH.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)
            scheduler.step(avg_loss)

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == cfg["epochs"] - 1:
                print(f"    Epoch {epoch:3d}/{cfg['epochs']} | loss={avg_loss:.6f}")

            if patience_counter >= cfg["patience"]:
                print(f"    早停于 epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        print(f"  [5/5] 误差修正训练完成, best_loss={best_val_loss:.6f}")
        return {"skip": False, "bias_pv": pv_bias, "bias_load": load_bias,
                "model": self.model, "history": history,
                "target": target}


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _build_prediction_input(data: dict, seq_len: int,
                             forecast: int) -> tuple[np.ndarray, np.ndarray]:
    """构建主模型预测输入 X, 同 LSTMTrainer.prepare_data 逻辑."""
    pv = data["pv_power"]
    n = len(pv)
    max_samples = n - seq_len - forecast
    if max_samples <= 0:
        return np.zeros((0, seq_len, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # 简化: 均匀采样而非昼夜均衡 (误差修正不需要均衡)
    step = max(1, max_samples // 5000)
    indices = list(range(0, max_samples, step))[:5000]

    X = np.zeros((len(indices), seq_len, 7), dtype=np.float32)
    for out_i, src_i in enumerate(indices):
        for j in range(seq_len):
            idx = src_i + j
            X[out_i, j, 0] = pv[idx]
            if "load_power" in data:
                X[out_i, j, 1] = data["load_power"][idx]
            if "solar_irradiance" in data:
                X[out_i, j, 2] = data["solar_irradiance"][idx]
            if "temperature" in data:
                X[out_i, j, 3] = data["temperature"][idx]

    return X, None


def _compute_residuals(X: np.ndarray, pred: np.ndarray,
                        true: np.ndarray, seq_len: int,
                        forecast: int) -> np.ndarray:
    """计算预测残差序列: e = true - pred (仅 P50 或第一步)."""
    n = len(X)
    residuals = np.zeros(n + seq_len + forecast, dtype=np.float32)
    # 用预测第一步作为残差近似
    for i in range(n):
        t_idx = i + seq_len
        residuals[t_idx] = true[t_idx] - pred[i, 0]
    return residuals


def _prepare_residual_data(residuals: np.ndarray, window: int,
                            horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """从残差序列构建 (X_err, y_err).

    X_err: (N, window, 1) 历史残差
    y_err: (N, horizon) 未来残差
    """
    n = len(residuals) - window - horizon
    if n <= 0:
        return np.zeros((0, window, 1), dtype=np.float32), np.zeros((0, horizon), dtype=np.float32)

    X = np.zeros((n, window, 1), dtype=np.float32)
    y = np.zeros((n, horizon), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = residuals[i:i + window]
        y[i] = residuals[i + window:i + window + horizon]
    return X, y


def export_error_correction_onnx(model: ErrorCorrectionBiLSTM,
                                  output_dir: str = "./exported_models/",
                                  residual_window: int = 24) -> str:
    """导出误差修正 BiLSTM 为 ONNX (v3.0 R2).

    Returns:
        ONNX 文件路径
    """
    import os
    import datetime
    _ensure_torch()

    try:
        import onnx
    except ImportError:
        print("[WARN] onnx 未安装, 跳过导出")
        return ""

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path = os.path.join(output_dir, f"error_correction_{ts}.onnx")

    dummy = _TORCH.randn(1, residual_window, 1)

    # wrap in nn.Module for ONNX export
    class ECWrapper(_nn.Module):
        def __init__(self, m):
            super().__init__()
            self.bilstm = m.bilstm
            self.head = m.head
            self.output_horizon = m.output_horizon

        def forward(self, x):
            h, _ = self.bilstm(x)
            return self.head(h[:, -self.output_horizon:]).squeeze(-1)

    wrapper = ECWrapper(model)

    _TORCH.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["residual_history"],
        output_names=["residual_forecast"],
        opset_version=13,
        dynamic_axes={"residual_history": {0: "batch"},
                       "residual_forecast": {0: "batch"}},
        metadata_props={
            "mupc_model_type": "error_correction",
            "mupc_residual_window": str(residual_window),
            "mupc_output_horizon": str(model.output_horizon),
            "mupc_version": "v3.1.0",
        },
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"误差修正 ONNX 导出: {onnx_path}")
    print("  ONNX checker: 通过")
    return onnx_path
