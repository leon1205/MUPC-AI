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

def _build_rl_export_model(obs_dim: int = 78,
                           hidden: list[int] | None = None):
    """构建仅用于 ONNX 导出的策略网络壳 (2 维动作, 对齐下游 v2.15)。

    下游 MUPC AI 引擎 PRD v2.15: 动作空间精简为 2 维 [p_ref, k_droop]。
    load_shedding/pv_limit 下沉至 strategy-engine，confidence 移至 ModelOutput。

    v2.14 观测空间: 78 维单模式 (D1~D10) / 79 维多模式 (+mode_id).

    Args:
        obs_dim: 观测维度 (default 78, v2.14 单模式)
        hidden: 隐藏层维度列表
    """
    import torch
    import torch.nn as nn

    if hidden is None:
        hidden = [128, 128]

    act_dim = 2  # v2.15 部署动作空间: [p_ref(tanh), k_droop(tanh)]

    class RLExportModel(nn.Module):
        """纯策略网络 (v3.1): 输入已归一化的 78/79 维观测，输出 tanh 动作。

        归一化由下游 Rust normalize_observation() 负责，完全镜像训练环境的
        mupc_env/observation.py:normalize_obs()。
        """
        def __init__(self):
            super().__init__()
            prev = obs_dim
            self.shared = nn.Sequential()
            for i, h in enumerate(hidden, 1):
                self.shared.add_module(f"fc{i}", nn.Linear(prev, h))
                self.shared.add_module(f"relu{i}", nn.ReLU())
                prev = h
            self.actor = nn.Linear(hidden[-1], act_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.shared(x)
            action = self.actor(latent)
            # 2 维 (v2.15): [p_ref(tanh), k_droop(tanh)]
            return torch.tanh(action)

    return RLExportModel()


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
    obs_dim: int = 78,
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

    # 验证 + 注入 metadata
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key="mupc_input_norm", value="external"))
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key="mupc_version", value="v3.1.0"))
    onnx.save(onnx_model, onnx_path)
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

def export_lstm(checkpoint_path: str, output_dir: str = "./exported_models/",
                input_window: int = 4, with_attention: bool = False,
                bidirection: bool = False, with_tcn: bool = False,
                metadata: dict | None = None) -> str:
    """导出 LSTM 模型为 ONNX (v3.0: 支持 Attention + metadata_props).

    Args:
        checkpoint_path: LSTMForecast 权重文件路径
        input_window: 输入窗口步数 (legacy=4, v3.0=12/24/36)
        with_attention: 是否嵌入 Attention 计算图
        bidirection: 是否导出 BiLSTM (R2)
        with_tcn: 是否前置 TCN 特征提取层 (v3.1 R2)
        metadata: metadata_props 字典 (v3.0 必须), None 则用默认值
    """
    import torch
    import torch.nn as nn
    import onnx

    _ensure_export_deps()

    from models.lstm import LSTMForecast
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # v3.0: auto-detect output_mode from checkpoint
    has_attn = "attention" in state_dict and state_dict["attention"] is not None
    has_p10p50p90 = "head_pv_p10" in state_dict
    detected_mode = "p10p50p90" if has_p10p50p90 else "legacy"
    detected_attn = has_attn
    print(f"  检测到模型模式: {detected_mode}, attention={detected_attn}")
    lstm_model = LSTMForecast(
        with_attention=detected_attn,
        output_mode=detected_mode,
        with_tcn=with_tcn,
    )
    lstm_model.load_state_dict(state_dict)
    lstm_model.eval()

    # 构建 ONNX 可导出包装器
    class LSTMWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.tcn = m.tcn  # v3.1: TCN 前置特征提取层
            self.lstm = m.lstm
            if m.attention is not None:
                self.attn_W = m.attention.W
                self.attn_v = m.attention.v
                self.has_attn = True
            else:
                self.has_attn = False
            self.output_mode = m.output_mode
            if self.output_mode == "p10p50p90":
                self.head_pv_p10 = m.head_pv_p10
                self.head_pv_p50 = m.head_pv_p50
                self.head_pv_p90 = m.head_pv_p90
                self.head_load_p10 = m.head_load_p10
                self.head_load_p50 = m.head_load_p50
                self.head_load_p90 = m.head_load_p90
            else:
                self.head_pv = m.head_pv
                self.head_load = m.head_load
                self.with_d10 = m.with_d10
                if m.with_d10:
                    self.head_d10_quantiles = m.head_d10_quantiles
                    self.head_d10_shock = m.head_d10_shock
                    self.head_d10_base = m.head_d10_base

        def forward(self, x):
            if self.tcn is not None:
                x = self.tcn.forward(x)  # v3.1: TCN 前置特征提取
            h_seq, (h_n, _) = self.lstm(x)
            if self.has_attn:
                score = self.attn_v(torch.tanh(self.attn_W(h_seq)))
                weights = torch.softmax(score.squeeze(-1), dim=1)
                ctx = torch.sum(weights.unsqueeze(-1) * h_seq, dim=1)
            else:
                ctx = h_n[-1]

            if self.output_mode == "p10p50p90":
                pv_p10 = torch.relu(self.head_pv_p10(ctx))
                pv_p50 = torch.relu(self.head_pv_p50(ctx))
                pv_p90 = torch.relu(self.head_pv_p90(ctx))
                load_p10 = torch.relu(self.head_load_p10(ctx))
                load_p50 = torch.relu(self.head_load_p50(ctx))
                load_p90 = torch.relu(self.head_load_p90(ctx))
                pv = torch.stack([pv_p10, pv_p50, pv_p90], dim=-1)
                lo = torch.stack([load_p10, load_p50, load_p90], dim=-1)
                out = torch.stack([pv, lo], dim=1)  # (B, 2, 15, 3)
                if self.has_attn:
                    return out, weights
                return out
            else:
                pv = torch.relu(self.head_pv(ctx))
                lo = torch.relu(self.head_load(ctx))
                return torch.cat([pv, lo], dim=-1)

    model = LSTMWrapper(lstm_model)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "bilstm_attn" if bidirection else ("lstm_attn" if with_attention else "lstm_forecast")
    onnx_path = os.path.join(output_dir, f"{prefix}_{ts}.onnx")

    dummy_input = torch.randn(1, input_window, 7)
    output_names = ["forecast", "attention_weights"] if with_attention else ["forecast"]

    # v3.0: metadata_props 注入
    export_metadata = metadata or {
        "mupc_model_type": "bilstm" if bidirection else "lstm",
        "mupc_with_attention": str(with_attention).lower(),
        "mupc_with_vmd": "false",     # v3.1: VMD 预处理默认关闭 (R2 可选)
        "mupc_mic_topk": "7",
        "mupc_output_horizon": "15",
        "mupc_input_window": str(input_window),
        "mupc_hidden_size": str(lstm_model.hidden_dim),
        "mupc_num_layers": str(lstm_model.num_layers),
        "mupc_with_tcn": str(with_tcn).lower(),        # v3.1: TCN 特征提取
        "mupc_direction": "bidirectional" if bidirection else "forward",
        "mupc_version": "v3.0.1",
    }

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["history"],
        output_names=output_names,
        opset_version=13,
        dynamic_axes={"history": {0: "batch"}, "forecast": {0: "batch"}},
    )

    onnx_model = onnx.load(onnx_path)
    # v3.0: 注入 metadata_props
    for k, v in export_metadata.items():
        onnx_model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_model)

    meta = {p.key: p.value for p in onnx_model.metadata_props}
    print(f"LSTM ONNX 导出: {onnx_path}")
    print(f"  metadata keys: {sorted(meta.keys())}")
    print("  ONNX checker: 通过")

    return onnx_path


# ── 误差修正 BiLSTM 导出 (v3.1) ───────────────────────────────

def export_error_correction(checkpoint_path: str, output_dir: str = "./exported_models/",
                             residual_window: int = 24) -> str:
    """导出误差修正 BiLSTM 模型为 ONNX (v3.1)。

    Args:
        checkpoint_path: ErrorCorrectionBiLSTM 权重文件路径
        output_dir: 输出目录
        residual_window: 残差历史窗口步数 (default 24)
    """
    import torch
    from models.error_correction import ErrorCorrectionBiLSTM, export_error_correction_onnx

    _ensure_export_deps()

    model = ErrorCorrectionBiLSTM(residual_window=residual_window)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  误差修正模型已加载: {checkpoint_path}")
    return export_error_correction_onnx(model, output_dir, residual_window)


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
    parser.add_argument("--obs-dim", type=int, default=78,
                        help="观测维度 (default: 78, v2.14 单模式; 多模式用 79)")
    parser.add_argument("--lstm", type=str, default=None,
                        help="导出 LSTM 模型 (提供 checkpoint 路径)")
    parser.add_argument("--input-window", type=int, default=4,
                        help="输入窗口步数 (legacy=4, v3.0=12/24/36)")
    parser.add_argument("--with-attention", action="store_true",
                        help="嵌入 Attention 计算图 (v3.0)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="导出 BiLSTM (v3.0 R2)")
    parser.add_argument("--with-tcn", action="store_true",
                        help="前置 TCN 特征提取层 (v3.1 R2)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="metadata JSON 文件路径 (v3.0)")
    parser.add_argument("--error-correction", type=str, default=None,
                        help="导出误差修正 BiLSTM 模型 (提供 checkpoint 路径, v3.1)")
    parser.add_argument("--to-rknn", action="store_true",
                        help="同时导出 RKNN (需要 rknn-toolkit2)")
    parser.add_argument("--dual-mode", action="store_true",
                        help="启用双参数下垂模式（2维训练空间，2维导出，v2.15）")
    args = parser.parse_args()

    try:
        # LSTM 导出
        if args.lstm:
            import json
            metadata = None
            if getattr(args, 'metadata', None):
                with open(args.metadata) as f:
                    metadata = json.load(f)
            onnx_path = export_lstm(args.lstm, args.output_dir,
                                    input_window=getattr(args, 'input_window', 4),
                                    with_attention=getattr(args, 'with_attention', False),
                                    bidirection=getattr(args, 'bidirectional', False),
                                    with_tcn=getattr(args, 'with_tcn', False),
                                    metadata=metadata)
            if args.to_rknn:
                export_to_rknn(onnx_path, args.output_dir)
            return

        # 误差修正导出
        if args.error_correction:
            ec_path = export_error_correction(args.error_correction, args.output_dir)
            if ec_path and args.to_rknn:
                export_to_rknn(ec_path, args.output_dir)
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
