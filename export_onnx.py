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

def _build_rl_export_model(obs_dim: int = 58,
                           hidden: list[int] | None = None):
    """构建仅用于 ONNX 导出的策略网络壳 (2 维动作, 对齐下游 v2.15)。

    下游 MUPC AI 引擎 PRD v2.15: 动作空间精简为 2 维 [p_ref, k_droop]。
    load_shedding/pv_limit 下沉至 strategy-engine，confidence 移至 ModelOutput。

    Args:
        obs_dim: 观测维度 (default 58, 兼容旧 checkpoint)
        hidden: 隐藏层维度列表
    """
    import torch
    import torch.nn as nn

    if hidden is None:
        hidden = [128, 128]

    act_dim = 2  # v2.15 部署动作空间: [p_ref(tanh), k_droop(tanh)]

    class RLExportModelWithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            prev = obs_dim
            self.shared = nn.Sequential()
            for i, h in enumerate(hidden, 1):
                self.shared.add_module(f"fc{i}", nn.Linear(prev, h))
                self.shared.add_module(f"relu{i}", nn.ReLU())
                prev = h
            self.actor = nn.Linear(hidden[-1], act_dim)

        def _normalize(self, x):
            # 观测归一化: 仅对 D1 前 8 维标量归一化
            # (D2-D9 维度已在训练时预归一化到 [0,1])
            d1_norm = torch.tensor([
                0.5,          # SOC ~ 50%
                1/150,        # 光伏功率
                1/60,         # 负荷功率
                1/200,        # 电网功率
                1/200,        # 变压器负载
                1/50,         # 电池功率
                1/200,        # 三相电压 (A相)
                1/200,        # 三相电压 (B相)
            ], device=x.device)
            x_norm = x.clone()
            x_norm[..., :8] = x[..., :8] * d1_norm
            # Q 裕度 (D7 [49]) 直接恒等
            return x_norm

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_norm = self._normalize(x)
            latent = self.shared(x_norm)
            action = self.actor(latent)
            # 2 维 (v2.15): [p_ref(tanh), k_droop(tanh)]
            return torch.tanh(action)

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
    dual_mode: bool = False,
) -> str:
    """导出 RL 策略网络为 ONNX。

    自动检测 checkpoint 格式 (SB3 .zip 或 NumPy .npz)。

    Args:
        checkpoint_dir: checkpoint 目录
        output_dir: 输出目录
        obs_dim: 观测维度
        checkpoint_path: checkpoint 路径 (None 则自动查找)
        dual_mode: 启用双参数下垂模式 (5维训练空间, 4维导出)

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

    # 构建 PyTorch 模型并加载权重
    model = _build_rl_export_model(obs_dim)

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
    parser.add_argument("--obs-dim", type=int, default=48,
                        help="观测维度 (default: 48, 多模式用 49)")
    parser.add_argument("--lstm", type=str, default=None,
                        help="导出 LSTM 模型 (提供 checkpoint 路径)")
    parser.add_argument("--to-rknn", action="store_true",
                        help="同时导出 RKNN (需要 rknn-toolkit2)")
    parser.add_argument("--dual-mode", action="store_true",
                        help="启用双参数下垂模式（2维训练空间，2维导出，v2.15）")
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
            dual_mode=args.dual_mode,
        )

        if args.to_rknn:
            export_to_rknn(onnx_path, args.output_dir)

    except Exception as e:
        print(f"导出失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
