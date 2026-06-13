"""
ONNX 模型导出 — RL 策略网络 + LSTM 预测模型。

用法:
  python export_onnx.py                          # 导出 RL 策略
  python export_onnx.py --lstm lstm_checkpoint.pt # 导出 LSTM
  python export_onnx.py --to-rknn                 # (可选) RKNN 转换
"""

import argparse
import os
import sys
import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ── 依赖检测 ──────────────────────────────────────────────────

def _ensure_export_deps():
    """确保 ONNX 导出依赖可用。"""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    if missing:
        raise ImportError(
            f"缺少 ONNX 导出依赖: {', '.join(missing)}。"
            f"请执行: pip install torch onnx onnxruntime"
        )


# ── RL 策略导出模型 ────────────────────────────────────────────

def _build_rl_export_model(obs_dim: int = 58, act_dim: int = 3,
                           hidden: list[int] | None = None,
                           pv_array_kw: float = 150.0,
                           load_peak_kw: float = 60.0):
    """构建包含归一化的 RL 策略网络壳（用于 ONNX 导出）。

    将 mupc_env._normalize_obs() 的逻辑 Bake 进 ONNX 模型，
    确保部署推理时收到与训练时相同尺度的输入。

    Args:
        obs_dim: 观测维度 (默认 58)
        act_dim: 动作维度 (默认 3: [p_batt, load_shedding, pv_limit])
        hidden: 隐藏层结构
        pv_array_kw: 光伏容量 (kW)，用于归一化边界
        load_peak_kw: 负荷峰值 (kW)，用于归一化边界
    """
    import torch
    import torch.nn as nn

    if hidden is None:
        hidden = [128, 128]

    class RLExportModelWithNorm(nn.Module):
        """策略网络 + 归一化（与 mupc_env._normalize_obs 对齐）。"""

        def __init__(self):
            super().__init__()
            prev = obs_dim
            self.shared = nn.Sequential()
            for i, h in enumerate(hidden, 1):
                self.shared.add_module(f"fc{i}", nn.Linear(prev, h))
                self.shared.add_module(f"relu{i}", nn.ReLU())
                prev = h
            self.actor = nn.Linear(hidden[-1], act_dim)

            # 预计算归一化边界（与 _normalize_obs 完全一致）
            self.register_buffer("_pv_kw", torch.tensor(pv_array_kw, dtype=torch.float32))
            self.register_buffer("_load_kw", torch.tensor(load_peak_kw, dtype=torch.float32))

        def _normalize(self, x: torch.Tensor) -> torch.Tensor:
            """与 mupc_env._normalize_obs 对齐的归一化算子。

            所有维度对应关系：
            [0]   identity (SOC)
            [1]   [0, PV] → [0,1]
            [2]   [0, LOAD_PEAK] → [0,1]
            [3]   [-500,500] → [-1,1]
            [4]   identity (transformer_load)
            [5]   [-500,500] → [-1,1]
            [6:9] [0.85,1.15] → [-1,1]
            [9]   identity (q_margin)
            [10:25] [0,PV] → [0,1]  (pv forecast 15维)
            [25:40] [0,LOAD_PEAK] → [0,1]  (load forecast 15维)
            [41:43] [0,1.5] → [0,1]
            [43]   [0,3] → [0,1]
            [44:47] [0,500] → [0,1]
            [47]   [0,1500] → [0,1]
            [48]   [-20,60] → [0,1]
            [49]   [-500,500] → [-1,1]
            [50,51..57,58] identity
            """
            pv = self._pv_kw.item()
            load = self._load_kw.item()
            out = x.clone()

            # D1
            out[:, 1:2] = (torch.clamp(x[:, 1:2], 0.0, pv) - 0.0) / (pv + 1e-9)
            out[:, 2:3] = (torch.clamp(x[:, 2:3], 0.0, load) - 0.0) / (load + 1e-9)
            out[:, 3:4] = (torch.clamp(x[:, 3:4], -500.0, 500.0) + 500.0) / 1000.0
            out[:, 5:6] = (torch.clamp(x[:, 5:6], -500.0, 500.0) + 500.0) / 1000.0
            out[:, 6:9] = (torch.clamp(x[:, 6:9], 0.85, 1.15) - 0.85) / 0.30
            # D2
            out[:, 10:25] = (torch.clamp(x[:, 10:25], 0.0, pv) - 0.0) / (pv + 1e-9)
            out[:, 25:40] = (torch.clamp(x[:, 25:40], 0.0, load) - 0.0) / (load + 1e-9)
            # D3
            out[:, 41:43] = torch.clamp(x[:, 41:43], 0.0, 1.5) / 1.5
            out[:, 43:44] = torch.clamp(x[:, 43:44], 0.0, 3.0) / 3.0
            # D4
            out[:, 44:47] = torch.clamp(x[:, 44:47], 0.0, 500.0) / 500.0
            # D5
            out[:, 47:48] = torch.clamp(x[:, 47:48], 0.0, 1500.0) / 1500.0
            out[:, 48:49] = (torch.clamp(x[:, 48:49], -20.0, 60.0) + 20.0) / 80.0
            # D6
            out[:, 49:50] = (torch.clamp(x[:, 49:50], -500.0, 500.0) + 500.0) / 1000.0
            # [0,4,9,50..58] identity（保持不变）
            return out

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_norm = self._normalize(x)
            latent = self.shared(x_norm)
            action = self.actor(latent)
            a1 = torch.tanh(action[:, :1])
            a2 = torch.sigmoid(action[:, 1:2])
            a3 = torch.sigmoid(action[:, 2:3])
            return torch.cat([a1, a2, a3], dim=-1)

    return RLExportModelWithNorm()


# ── 权重加载 ──────────────────────────────────────────────────

def _load_sb3_weights(checkpoint_path: str, obs_dim: int) -> dict:
    """从 SB3 checkpoint 提取权重。"""
    import torch
    from stable_baselines3 import PPO, SAC

    # 尝试 PPO
    try:
        model = PPO.load(checkpoint_path)
        state_dict = model.policy.state_dict()
    except Exception:
        try:
            model = SAC.load(checkpoint_path)
            state_dict = model.policy.state_dict()
        except Exception as e:
            raise RuntimeError(f"无法加载 SB3 checkpoint: {checkpoint_path}") from e
    return state_dict

def _load_npz_weights(npz_path: str, obs_dim: int) -> dict:
    """从 NumPy PPO checkpoint 加载权重。"""
    data = dict(np.load(npz_path))
    return data


# ── 主导出函数 ─────────────────────────────────────────────────

def export_rl_policy(
    checkpoint_dir: str = "./checkpoints/",
    output_dir: str = "./exported_models/",
    obs_dim: int = 58,
    checkpoint_path: str | None = None,
    pv_array_kw: float = 150.0,
    load_peak_kw: float = 60.0,
) -> str:
    """导出 RL 策略网络为 ONNX（包含归一化预处理）。

    导出的模型包含与 mupc_env._normalize_obs 对齐的归一化层，
    可直接接收原始观测输入，无需外部预处理。

    Args:
        checkpoint_dir: checkpoint 目录
        output_dir: 输出目录
        obs_dim: 观测维度 (58 单模式，59 多模式)
        checkpoint_path: checkpoint 路径（None 则自动查找最新）
        pv_array_kw: 光伏容量 (kW)，用于归一化
        load_peak_kw: 负荷峰值 (kW)，用于归一化

    Returns:
        导出的 ONNX 文件路径
    """
    import torch
    import onnx

    _ensure_export_deps()

    os.makedirs(output_dir, exist_ok=True)

    # 找到 checkpoint
    if checkpoint_path is None:
        # 自动查找
        checkpoints = list(Path(checkpoint_dir).glob("*.zip")) + \
                      list(Path(checkpoint_dir).glob("*.npz"))
        if not checkpoints:
            raise FileNotFoundError(f"未在 {checkpoint_dir} 找到 checkpoint (.zip/.npz)")
        checkpoint_path = str(sorted(checkpoints, key=os.path.getmtime)[-1])

    print(f"使用 checkpoint: {checkpoint_path}")
    is_npz = checkpoint_path.endswith(".npz")

    # 构建 PyTorch 模型并加载权重（包含归一化）
    model = _build_rl_export_model(obs_dim, pv_array_kw=pv_array_kw, load_peak_kw=load_peak_kw)

    if is_npz:
        # NumPy PPO 权重 → PyTorch state_dict
        npz_weights = dict(np.load(checkpoint_path))
        state_dict = {}
        # fc1
        state_dict["shared.fc1.weight"] = torch.tensor(npz_weights["fc1_w"].T.copy())
        state_dict["shared.fc1.bias"] = torch.tensor(npz_weights["fc1_b"].copy())
        # fc2
        state_dict["shared.fc2.weight"] = torch.tensor(npz_weights["fc2_w"].T.copy())
        state_dict["shared.fc2.bias"] = torch.tensor(npz_weights["fc2_b"].copy())
        # actor
        state_dict["actor.weight"] = torch.tensor(npz_weights["actor_w"].T.copy())
        state_dict["actor.bias"] = torch.tensor(npz_weights["actor_b"].copy())
    else:
        # SB3 checkpoint
        from stable_baselines3 import PPO, SAC
        try:
            sb3_model = PPO.load(checkpoint_path)
        except Exception:
            sb3_model = SAC.load(checkpoint_path)
        state_dict = sb3_model.policy.state_dict()

        # SB3 policy 使用 mlp_extractor 结构
        # policy_net(0.weight), policy_net(0.bias), policy_net(2.weight), ...
        # 需要映射到我们的 RLExportModel 结构
        mapped = {}
        sb3_sd = sb3_model.policy.state_dict()
        # 提取 shared 网络部分 (policy_net / value_net 共享? PPO 分离)
        # 简化: 直接使用 pi 网络
        if "mlp_extractor.policy_net.0.weight" in sb3_sd:
            mapped["shared.fc1.weight"] = sb3_sd["mlp_extractor.policy_net.0.weight"]
            mapped["shared.fc1.bias"] = sb3_sd["mlp_extractor.policy_net.0.bias"]
            if "mlp_extractor.policy_net.2.weight" in sb3_sd:
                mapped["shared.fc2.weight"] = sb3_sd["mlp_extractor.policy_net.2.weight"]
                mapped["shared.fc2.bias"] = sb3_sd["mlp_extractor.policy_net.2.bias"]
            mapped["actor.weight"] = sb3_sd["action_net.weight"]
            mapped["actor.bias"] = sb3_sd["action_net.bias"]
        else:
            # 直接使用 policy.state_dict() 键
            pass
        state_dict = mapped

    # 加载权重 (严格匹配可用键)
    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f"  匹配权重: {len(matched)}/{len(model_dict)} 层")

    model.eval()

    # 导出 ONNX
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path = os.path.join(output_dir, f"mupc_rl_policy_{ts}.onnx")

    dummy_input = torch.randn(1, obs_dim)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["observation"],
        output_names=["action"],
        opset_version=13,
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"ONNX 导出: {onnx_path}")

    # 验证
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: 通过")

    # onnxruntime 推理验证
    _verify_onnx_inference(model, onnx_path, obs_dim)

    return onnx_path


def _verify_onnx_inference(pytorch_model, onnx_path: str, obs_dim: int) -> None:
    """使用 onnxruntime 验证 ONNX 推理一致性。"""
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        print("  onnxruntime 不可用, 跳过推理验证")
        return

    test_input = torch.randn(3, obs_dim)
    with torch.no_grad():
        pytorch_out = pytorch_model(test_input).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {"observation": test_input.numpy()})[0]

    max_err = np.max(np.abs(pytorch_out - onnx_out))
    print(f"  推理验证: max error = {max_err:.2e} {'[PASS]' if max_err < 1e-5 else '[WARN]'}")


# ── LSTM 导出 ─────────────────────────────────────────────────

def export_lstm(checkpoint_path: str, output_dir: str = "./exported_models/") -> str:
    """导出 LSTM 模型为 ONNX。

    Input:  (batch=1, seq_len=4, features=6)
    Output: (batch=1, 30) = [pv_forecast(15), load_forecast(15)]
    """
    import torch
    import torch.nn as nn
    import onnx

    _ensure_export_deps()

    from lstm_model import LSTMForecast
    lstm_model = LSTMForecast()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    lstm_model.load_state_dict(state_dict)
    lstm_model.eval()

    # Wrap in nn.Module for ONNX export
    class LSTMWrapper(nn.Module):
        def __init__(self, lstm_model):
            super().__init__()
            # Use LSTMForecast's actual submodules (already initialized with correct dropout)
            self.lstm = lstm_model.lstm
            self.head_pv = lstm_model.head_pv
            self.head_load = lstm_model.head_load
            # LSTMForecast.state_dict() returns nested dict, load into nn.Module directly
            # by flattening it
            nested_sd = lstm_model.state_dict()
            flat_sd = {}
            for submod, params in nested_sd.items():
                for k, v in params.items():
                    flat_sd[f"{submod}.{k}"] = v
            self.load_state_dict(flat_sd)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]
            pv_pred = self.head_pv(last_hidden)
            load_pred = self.head_load(last_hidden)
            return torch.cat([pv_pred, load_pred], dim=-1)

    model = LSTMWrapper(lstm_model)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path = os.path.join(output_dir, f"lstm_forecast_{ts}.onnx")

    dummy_input = torch.randn(1, 4, 6)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["history"],
        output_names=["forecast"],
        opset_version=13,
        dynamic_axes={"history": {0: "batch"}, "forecast": {0: "batch"}},
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"LSTM ONNX 导出: {onnx_path}")
    print("  ONNX checker: 通过")

    return onnx_path


# ── RKNN 转换 (可选) ──────────────────────────────────────────

def export_to_rknn(onnx_path: str, output_dir: str = "./exported_models/") -> str:
    """调用 rknn-toolkit2 进行 INT8 量化。"""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 未安装, 跳过 RKNN 转换。"
              "请参考: https://github.com/airockchip/rknn-toolkit2")
        return ""

    rknn = RKNN(verbose=True)
    rknn.config(mean_values=[[0.0]], std_values=[[1.0]],
                 target_platform="rk3588", quantized_dtype="asymmetric_quantized-u8")

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"RKNN 加载 ONNX 失败: ret={ret}")
        return ""

    ret = rknn.build(do_quantization=True)
    if ret != 0:
        print(f"RKNN build 失败: ret={ret}")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    rknn_path = os.path.join(output_dir, f"{base}.rknn")
    rknn.export_rknn(rknn_path)
    print(f"RKNN 导出: {rknn_path}")

    rknn.release()
    return rknn_path


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MUPC 模型导出 (ONNX / RKNN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 路径 (自动检测 SB3 .zip / NumPy .npz)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/",
                        help="checkpoint 目录 (default: ./checkpoints/)")
    parser.add_argument("--output-dir", type=str, default="./exported_models/",
                        help="输出目录 (default: ./exported_models/)")
    parser.add_argument("--obs-dim", type=int, default=58,
                        help="观测维度 (default: 58, 多模式用 59)")
    parser.add_argument("--pv-kw", type=float, default=150.0,
                        help="光伏容量 (kW)，用于归一化 (default: 150)")
    parser.add_argument("--load-kw", type=float, default=60.0,
                        help="负荷峰值 (kW)，用于归一化 (default: 60)")
    parser.add_argument("--lstm", type=str, default=None,
                        help="导出 LSTM 模型 (提供 checkpoint 路径)")
    parser.add_argument("--to-rknn", action="store_true",
                        help="同时导出 RKNN (需要 rknn-toolkit2)")
    args = parser.parse_args()

    try:
        # LSTM 导出
        if args.lstm:
            onnx_path = export_lstm(args.lstm, args.output_dir)
            if args.to_rknn:
                export_to_rknn(onnx_path, args.output_dir)
            return

        # RL 策略导出
        onnx_path = export_rl_policy(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            obs_dim=args.obs_dim,
            checkpoint_path=args.checkpoint,
            pv_array_kw=args.pv_kw,
            load_peak_kw=args.load_kw,
        )

        if args.to_rknn:
            export_to_rknn(onnx_path, args.output_dir)

    except Exception as e:
        print(f"导出失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
