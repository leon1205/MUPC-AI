"""
MUPC RL 模型训练主入口。

用法:
  #快速测试（5万步，Oracle预测）
  python train.py --mode MODE-01 --total-timesteps 50000 --no-lstm

  # LSTM训练 + RL训练（合并数据，1M步）
  python train.py --mode MODE-01 --data-source merged --train-lstm \
         --lstm-params hidden_dim=128,num_layers=3,epochs=200,patience=30 \
         --total-timesteps 1000000 --export-onnx

  # 5个独立模型（SMART-DS数据，20万步/每模式）
  python train.py --mode single --data-source smartds --total-timesteps 200000

  # 多模式单模型（Oracle预测，100万步）
  python train.py --mode all --no-lstm --total-timesteps 1000000

  # 中国数据 + 低学习率衰减
  python train.py --mode all --data-source china --lr-decay --total-timesteps 500000

  # 独立训练 LSTM（仅LSTM，不跑RL）
  python train.py --train-lstm --data-source merged --lstm-params epochs=100
"""

import argparse
import os
import sys
import time
import datetime
import json
from pathlib import Path

import numpy as np

# ── 依赖检测 ──────────────────────────────────────────────────

def _check_sb3() -> bool:
    try:
        import stable_baselines3
        return True
    except ImportError:
        return False

def _check_gymnasium() -> bool:
    try:
        import gymnasium
        return True
    except ImportError:
        return False

def _check_torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


# ── 参数解析 ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MUPC 强化学习模型训练管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py --mode MODE-01 --total-timesteps 50000 --no-lstm
  python train.py --mode all --data-source merged --train-lstm --total-timesteps 1000000
  python train.py --train-lstm --data-source merged --lstm-params epochs=100
        """,
    )

    # ── 模式 ──
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05", "single"],
                   help="训练模式: all=多模式单模型, single=5个独立模型, MODE-0X=单模式 (default: all)")
    p.add_argument("--algo", type=str, default="ppo",
                   choices=["ppo", "sac"],
                   help="RL 算法 (default: ppo)")

    # ── 数据 ──
    p.add_argument("--data-source", type=str, default="smartds",
                   choices=["smartds", "china", "merged", "unified"],
                   help="数据源: smartds=SMART-DS, china=中国合成, merged=SMART-DS+中国合并, unified=三源合并 (default: smartds)")
    p.add_argument("--lat", type=float, default=31.23, help="中国数据纬度 (default: 31.23)")
    p.add_argument("--lon", type=float, default=121.47, help="中国数据经度 (default: 121.47)")
    # ── 兼容旧参数 ──
    p.add_argument("--merge", action="store_true",
                   help="[兼容] 等同于 --data-source merged")

    # ── LSTM ──
    p.add_argument("--train-lstm", action="store_true",
                   help="仅训练 LSTM 预测器（不跑 RL 训练）")
    p.add_argument("--lstm-checkpoint", type=str, default=None,
                   help="预训练 LSTM 模型路径")
    p.add_argument("--lstm-params", type=str, default="",
                   help="LSTM 超参，格式: key=val[,key=val] (hidden_dim,num_layers,epochs,batch_size,patience)")
    p.add_argument("--no-lstm", action="store_true",
                   help="使用 Oracle 预测 (真实值+噪声) 代替 LSTM")

    # ── RL 训练 ──
    p.add_argument("--total-timesteps", type=int, default=175200,
                   help="总训练步数 (default: 175200)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="学习率 (default: 3e-4)")
    p.add_argument("--lr-decay", action="store_true",
                   help="启用学习率线性衰减 (1e-4 → 1e-5)")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="熵系数 (default: 0.01)")
    p.add_argument("--net-arch", type=str, default="128,128",
                   help="网络结构，逗号分隔 (default: 128,128)")
    p.add_argument("--eval-freq", type=int, default=10000,
                   help="评估频率 (default: 10000)")

    # ──通用 ──
    p.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    p.add_argument("--tensorboard-log", type=str, default="./tensorboard_logs/",
                   help="TensorBoard 日志目录")
    p.add_argument("--checkpoint-path", type=str, default="./checkpoints/",
                   help="模型保存目录")
    p.add_argument("--export-onnx", action="store_true",
                   help="训练完成后自动导出 ONNX")
    p.add_argument("--reward-weights", type=str, default="",
                   help="自定义权重, e.g. MODE-01=1.5,0.3,3.0 (逗号分隔)")
    return p.parse_args()


# ── 权重解析 ──────────────────────────────────────────────────

def parse_custom_weights(raw: str) -> dict[str, list[float]]:
    """解析 --reward-weights 参数。"""
    if not raw:
        return {}
    result = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, vals = part.split("=", 1)
        key = key.strip()
        result[key] = [float(v) for v in vals.split("/")]
    return result


# ── LSTM 参数解析 ──────────────────────────────────────────────

def parse_lstm_params(raw: str) -> dict:
    """解析 --lstm-params 参数。"""
    defaults = {
        "hidden_dim": 64, "num_layers": 2, "epochs": 200,
        "batch_size": 64, "patience": 20,
    }
    if not raw:
        return defaults
    for part in raw.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        defaults[k.strip()] = float(v.strip()) if "." in v else int(v.strip())
    return defaults


# ── 网络结构解析 ──────────────────────────────────────────────

def parse_net_arch(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── 兼容旧参数 ──────────────────────────────────────
    if args.merge:
        args.data_source = "merged"

    # ── 依赖检测 ────────────────────────────────────────
    print("=" * 56)
    print("  MUPC RL 训练管线")
    print("=" * 56)

    has_sb3 = _check_sb3()
    has_gym = _check_gymnasium()
    has_torch = _check_torch()
    print(f"\n依赖检测: SB3={'可用' if has_sb3 else '不可用(将使用NumPy PPO)'}, "
          f"Gymnasium={'可用' if has_gym else '不可用(使用_gym_stub)'}, "
          f"PyTorch={'可用' if has_torch else '不可用'}")

    # ── 数据加载 ────────────────────────────────────────
    print("\n── 数据加载 ──")
    from data_loader import SmartDSLoader, ChinaDataLoader, UnifiedDataLoader

    if args.data_source == "smartds":
        loader = SmartDSLoader()
        data = loader.load_all()
    elif args.data_source == "china":
        loader = ChinaDataLoader()
        data = loader.load_all()
    elif args.data_source in ("merged", "unified"):
        loader = UnifiedDataLoader(
            lat=args.lat, lon=args.lon,
            merge_data=True,
            auto_generate=True,
        )
        data = loader.load_all()
    else:
        raise ValueError(f"未知 data_source: {args.data_source}")

    train_data, val_data = loader.split(data)
    print(f" 训练: {train_data['n_steps']} 步 ({train_data['n_steps']*15/60/24:.0f} 天), "
          f"验证: {val_data['n_steps']} 步 ({val_data['n_steps']*15/60/24:.0f} 天)")

    # ── LSTM / Oracle ────────────────────────────────────
    predictor = None
    lstm_trained_path = None

    if args.train_lstm:
        # 仅训练 LSTM，不跑 RL
        _train_lstm_only(train_data, val_data, args)
        print("\nLSTM 训练完成（无 RL 训练）。")

        # LSTM 训练后导出 ONNX（如果指定了 --export-onnx）
        if args.export_onnx and has_torch:
            print("\n── LSTM ONNX 导出 ──")
            try:
                import export_onnx
                lstm_path = os.path.join(args.checkpoint_path, "lstm_checkpoint.pt")
                onnx_path = export_onnx.export_lstm(lstm_path, "./exported_models/")
                print(f"  LSTM ONNX 已导出: {onnx_path}")
            except Exception as e:
                print(f"  LSTM ONNX 导出失败: {e}")
        return

    if not args.no_lstm and args.lstm_checkpoint:
        import torch
        from lstm_model import LSTMForecast
        print(f"\n── LSTM 模型加载 ──")
        model = LSTMForecast()
        model.load_state_dict(torch.load(args.lstm_checkpoint, map_location="cpu"))
        model.lstm.eval()
        predictor = model
        predictor.set_data(train_data)
        lstm_trained_path = args.lstm_checkpoint
    elif not args.no_lstm and has_torch:
        print("\n── LSTM 模型训练 ──")
        from lstm_model import LSTMTrainer
        lstm_cfg = parse_lstm_params(args.lstm_params)
        print(f"  LSTM 参数: {lstm_cfg}")
        trainer = LSTMTrainer(lstm_cfg)
        result = trainer.train(train_data, val_data)
        predictor = result["model"]
        predictor.set_data(train_data)
        # 保存 LSTM checkpoint
        os.makedirs(args.checkpoint_path, exist_ok=True)
        lstm_path = os.path.join(args.checkpoint_path, "lstm_checkpoint.pt")
        import torch
        torch.save(predictor.state_dict(), lstm_path)
        print(f"  LSTM checkpoint 已保存: {lstm_path}")
        lstm_trained_path = lstm_path
    else:
        from lstm_model import OraclePredictor
        print("\n── 使用 Oracle 预测 ──")
        predictor = OraclePredictor(train_data)

    # ── 确定训练模式列表 ────────────────────────────────
    if args.mode == "single":
        modes = ["MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"]
    elif args.mode == "all":
        modes = ["all"]
    else:
        modes = [args.mode]

    # ── 训练 ────────────────────────────────────────────
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    custom_weights = parse_custom_weights(args.reward_weights)

    for mode in modes:
        print(f"\n{'=' * 56}")
        print(f"  训练模式: {mode}")
        print(f"{'=' * 56}")

        from mupc_env import MupcEnv
        env = MupcEnv(train_data, mode=mode, lstm_predictor=predictor,
                      reward_weights=custom_weights if custom_weights else None)
        eval_env = MupcEnv(val_data, mode=mode, lstm_predictor=predictor,
                          reward_weights=custom_weights if custom_weights else None)

        obs_dim = env.observation_space.shape[0]
        print(f"  观测空间: Box({obs_dim},), 动作空间: Box(2,), 算法: {args.algo}")

        if has_sb3 and has_gym:
            _train_sb3(env, eval_env, args, obs_dim, mode)
        else:
            _train_numpy_ppo(env, eval_env, args, obs_dim, mode)

        # ONNX 导出
        if args.export_onnx and has_torch:
            print("\n── ONNX 导出 ──")
            try:
                import export_onnx
                onnx_path = export_onnx.export_rl_policy(
                    checkpoint_dir=args.checkpoint_path,
                    output_dir="./exported_models/",
                    obs_dim=obs_dim,
                )
                print(f"  ONNX 模型已导出: {onnx_path}")
            except Exception as e:
                print(f"  ONNX 导出失败: {e}")

    print(f"\n{'=' * 56}")
    print(f"  训练完成。checkpoint: {args.checkpoint_path}")
    print(f"{'=' * 56}")


# ── LSTM 独立训练 ─────────────────────────────────────────────

def _train_lstm_only(train_data: dict, val_data: dict, args):
    """仅训练 LSTM，不进行 RL 训练。"""
    print("\n" + "=" * 56)
    print("  LSTM 独立训练")
    print("=" * 56)

    from lstm_model import LSTMTrainer
    import torch

    lstm_cfg = parse_lstm_params(args.lstm_params)
    print(f"  LSTM 参数: {lstm_cfg}")

    t0 = time.time()
    trainer = LSTMTrainer(lstm_cfg)
    result = trainer.train(train_data, val_data)
    elapsed = time.time() - t0

    os.makedirs(args.checkpoint_path, exist_ok=True)
    lstm_path = os.path.join(args.checkpoint_path, "lstm_checkpoint.pt")
    torch.save(result["model"].state_dict(), lstm_path)
    print(f"\n  LSTM 训练完成，耗时: {elapsed:.0f}s")
    print(f"  checkpoint: {lstm_path}")


# ── SB3 训练 ──────────────────────────────────────────────────

def _train_sb3(env, eval_env, args, obs_dim: int, mode: str):
    import gymnasium
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    algo_cls = PPO if args.algo == "ppo" else SAC
    algo_name = "PPO" if args.algo == "ppo" else "SAC"

    print(f"\n── SB3 {algo_name} 训练 ──")
    print(f"  总步数: {args.total_timesteps}, 模式: {mode}")

    vec_env = DummyVecEnv([lambda: Monitor(env)])
    vec_eval = DummyVecEnv([lambda: Monitor(eval_env)])

    eval_callback = EvalCallback(
        vec_eval, best_model_save_path=args.checkpoint_path,
        log_path=args.checkpoint_path, eval_freq=args.eval_freq,
        deterministic=True, render=False,
    )

    import torch.nn as nn
    net_arch = parse_net_arch(args.net_arch)
    policy_kwargs = {
        "net_arch": {"pi": net_arch, "vf": net_arch},
        "activation_fn": nn.ReLU,
    }

    # 学习率
    if args.lr_decay:
        from stable_baselines3.common.utils import get_linear_fn
        lr = get_linear_fn(args.lr, 1e-5, 1.0)
    else:
        lr = args.lr

    model = algo_cls(
        "MlpPolicy", vec_env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2 if args.algo == "ppo" else None,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        verbose=1,
    )

    ts_name = f"{args.algo}_{mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            tb_log_name=f"mupc_{ts_name}",
        )
        checkpoint_path = os.path.join(args.checkpoint_path, f"final_model")
        model.save(checkpoint_path)
        print(f"  最终模型已保存: {checkpoint_path}.zip")
    except KeyboardInterrupt:
        checkpoint_path = os.path.join(args.checkpoint_path, "interrupted_model")
        model.save(checkpoint_path)
        print(f"\n  训练中断, 模型已保存: {checkpoint_path}.zip")
        sys.exit(0)


# ── NumPy PPO 训练 ────────────────────────────────────────────

def _train_numpy_ppo(env, eval_env, args, obs_dim: int, mode: str):
    from _ppo_core import NumPyPPO

    print(f"\n── NumPy PPO 训练 (SB3 不可用) ──")
    print(f"  总步数: {args.total_timesteps}, 模式: {mode}")

    model = NumPyPPO(env, obs_dim=obs_dim)

    try:
        log = model.learn(args.total_timesteps)
        weights_path = os.path.join(args.checkpoint_path, "ppo_weights.npz")
        model.save_weights(weights_path)
        print(f"  权重已保存: {weights_path}")
    except KeyboardInterrupt:
        weights_path = os.path.join(args.checkpoint_path, "ppo_weights_interrupted.npz")
        model.save_weights(weights_path)
        print(f"\n  训练中断, 权重已保存: {weights_path}")
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()