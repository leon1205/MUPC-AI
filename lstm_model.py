"""
LSTM 时序预测模型 — 光伏出力与负荷功率联合预测。

用途: 为 RL 环境提供 D2 预测数据 (pv_forecast_15min + load_forecast_15min)。
      也可由 train.py 独立训练并导出 ONNX。

架构: 2 层 LSTM (hidden=64) + 双头 Linear (pv_head, load_head)
输入: (batch, seq_len=4, 6) — 过去 60 分钟历史
输出: (batch, 30) — 未来 15 分钟 pv(15维) + load(15维)
"""

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

# ── 模型定义 ──────────────────────────────────────────────────

class LSTMForecast:
    """LSTM 时序预测模型 (PyTorch)。"""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                 num_layers: int = 2, forecast_steps: int = 15,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
                [pv_power, load_power, ghi, temperature, sin_hour, cos_hour]
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            forecast_steps: 预测步数 (默认 15 = 15 分钟)
            dropout: dropout 比率
        """
        _ensure_torch()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.output_dim = forecast_steps * 2  # pv(15) + load(15)

        self.lstm = _nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_pv = _nn.Linear(hidden_dim, forecast_steps)
        self.head_load = _nn.Linear(hidden_dim, forecast_steps)

        self.device = "cpu"
        self._initialized = True

    def to(self, device: str) -> "LSTMForecast":
        self.lstm.to(device)
        self.head_pv.to(device)
        self.head_load.to(device)
        self.device = device
        return self

    def parameters(self):
        return list(self.lstm.parameters()) + \
               list(self.head_pv.parameters()) + \
               list(self.head_load.parameters())

    def state_dict(self) -> dict:
        return {
            "lstm": self.lstm.state_dict(),
            "head_pv": self.head_pv.state_dict(),
            "head_load": self.head_load.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.lstm.load_state_dict(d["lstm"])
        self.head_pv.load_state_dict(d["head_pv"])
        self.head_load.load_state_dict(d["head_load"])

    def eval(self) -> "LSTMForecast":
        """切换到评估模式。"""
        self.lstm.eval()
        self.head_pv.eval()
        self.head_load.eval()
        return self

    def train(mode: bool = True) -> "LSTMForecast":
        """切换到训练/评估模式。"""
        self.lstm.train(mode)
        self.head_pv.train(mode)
        self.head_load.train(mode)
        return self

    def forward(self, x) -> "torch.Tensor":
        """
        Args:
            x: (batch, seq_len, input_dim) = (batch, 4, 6)
        Returns:
            (batch, forecast_steps * 2) = (batch, 30)
            前 15 维 = pv_forecast, 后 15 维 = load_forecast
        """
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        pv_pred = self.head_pv(last_hidden)
        load_pred = self.head_load(last_hidden)
        return _torch.cat([pv_pred, load_pred], dim=-1)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy 接口: 输入 (batch, 4, 6) → 输出 (batch, 30)。"""
        _ensure_torch()
        self.lstm.eval()
        with _torch.no_grad():
            t = _torch.tensor(x, dtype=_torch.float32).to(self.device)
            out = self.forward(t)
        return out.cpu().numpy()

    def __call__(self, x):
        return self.forward(x)


# ── 训练器 ─────────────────────────────────────────────────────

LSTM_TRAIN_CONFIG = {
    "input_seq_len": 4,          # 60 分钟 / 15 分钟
    "forecast_steps": 15,        # 预测 15 步
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 100,
    "patience": 15,              # 早停
}


class LSTMTrainer:
    """LSTM 模型训练器。"""

    def __init__(self, config: dict | None = None):
        _ensure_torch()
        self.config = {**LSTM_TRAIN_CONFIG, **(config or {})}
        self.model: Optional[LSTMForecast] = None
        self.device = "cuda" if _torch.cuda.is_available() else "cpu"

    def prepare_data(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """从 data dict 构建训练样本 (X, y), 昼夜均衡采样。

        夜间 PV=0 样本占 ~50%, 导致模型恒输出 0。
        保留全部白天样本 (PV目标>10kW), 随机下采样夜间样本至 1:1。
        """
        pv = data["pv_power"]
        load = data["load_power"]
        ghi = data["solar_irradiance"]
        temp = data["temperature"]
        hours = data.get("hours", np.arange(len(pv), dtype=np.float32) * 15 / 60 % 24)

        seq_len = self.config["input_seq_len"]
        forecast = self.config["forecast_steps"]
        n = len(pv)
        max_samples = n - seq_len - forecast

        # 第一遍: 构建所有样本并分类
        day_idx = []
        night_idx = []
        for i in range(max_samples):
            # 检查预测窗口内是否有有效光伏 (>10kW)
            pv_target_max = np.max(pv[i + seq_len : i + seq_len + forecast])
            if pv_target_max > 10.0:
                day_idx.append(i)
            else:
                night_idx.append(i)

        # 均衡: 白天全保留, 夜间下采样至白天数量的1倍
        n_day = len(day_idx)
        n_night = min(len(night_idx), n_day)
        if n_night > 0 and len(night_idx) > n_night:
            np.random.seed(42)
            night_idx = sorted(np.random.choice(night_idx, n_night, replace=False).tolist())

        balanced = sorted(day_idx + night_idx)
        n_samples = len(balanced)
        print(f"  样本平衡: 白天={n_day}, 夜间={n_night}, 总计={n_samples}")

        X = np.zeros((n_samples, seq_len, 6), dtype=np.float32)
        y = np.zeros((n_samples, forecast * 2), dtype=np.float32)

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

            for k in range(forecast):
                target_idx = src_i + seq_len + k
                y[out_i, k] = pv[target_idx]
                y[out_i, k + forecast] = load[target_idx]

        return X, y

    def train(self, data: dict, val_data: dict | None = None) -> dict:
        """训练 LSTM 模型。

        Returns:
            {"model": LSTMForecast, "history": {"train_loss": [...], "val_loss": [...]}}
        """
        print("\n" + "=" * 56)
        print("  LSTM 预测模型训练")
        print("=" * 56)

        cfg = self.config
        X, y = self.prepare_data(data)
        print(f"  训练样本: {len(X)}")
        if val_data is not None:
            X_val, y_val = self.prepare_data(val_data)
            print(f"  验证样本: {len(X_val)}")

        self.model = LSTMForecast(
            input_dim=6, hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"], forecast_steps=cfg["forecast_steps"],
            dropout=cfg["dropout"],
        ).to(self.device)

        optimizer = _torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        loss_fn = _nn.HuberLoss()
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
                loss = loss_fn(pred, batch_y)
                loss.backward()
                _torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

            # ── 验证 ──
            if val_data is not None:
                self.model.lstm.eval()
                with _torch.no_grad():
                    t_x = _torch.tensor(X_val, dtype=_torch.float32).to(self.device)
                    t_y = _torch.tensor(y_val, dtype=_torch.float32).to(self.device)
                    val_pred = self.model(t_x)
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
        print(f"  目标: Load_MAPE <= 15% {'[PASS]' if load_mape <= 15 else '[WARN]'}")

        return {"model": self.model, "history": history}

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
            pred = self.model(t_x).cpu().numpy()
        offset = 0 if is_pv else self.config["forecast_steps"]
        y_true = y[idx][:, offset:offset + self.config["forecast_steps"]]
        y_pred = pred[:, offset:offset + self.config["forecast_steps"]]

        if is_pv:
            # PV: 仅统计实际出力 > 10kW (5%容量) 的点 — 夜间不计
            mask = y_true > 10.0
        else:
            # 负荷: 仅统计 > 10kW 的点
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
        if _TORCH_AVAILABLE or __import__('importlib').util.find_spec('torch'):
            print("\n开始 LSTM 训练自测...")
            trainer = LSTMTrainer({"epochs": 5, "batch_size": 32})
            result = trainer.train(train, val)
            print("LSTM 训练自测完成。")
    except ImportError as e:
        print(f"依赖缺失，跳过自测: {e}")
    except FileNotFoundError as e:
        print(f"数据缺失，跳过自测: {e}")
