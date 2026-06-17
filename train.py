"""
MUPC RL 模型训练主入口。

用法:
  python train.py                          # 默认多模式 PPO 训练
  python train.py --mode MODE-01           # 单模式 (农网灌溉)
  python train.py --algo sac               # SAC 算法
  python train.py --total-timesteps 50000  # 快速测试
  python train.py --reward-weights MODE-01=1.5,0.3,3.0  # 自定义权重
  python train.py --no-lstm                # 使用 Oracle 预测
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
  python train.py                                          # 默认多模式 PPO
  python train.py --mode MODE-01 --algo sac               # 单模式 SAC
  python train.py --total-timesteps 50000                 # 快速测试
  python train.py --reward-weights MODE-01=1.5,0.3,3.0   # 自定义权重
        """,
    )
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"],
                   help="训练模式 (default: all = 多模式单模型)")
    p.add_argument("--algo", type=str, default="ppo",
                   choices=["ppo", "sac"],
                   help="RL 算法 (default: ppo)")
    p.add_argument("--total-timesteps", type=int, default=175200,
                   help="总训练步数 (default: 175200)")
    p.add_argument("--reward-weights", type=str, default="",
                   help="自定义权重, e.g. MODE-01=1.5,0.3,3.0 (逗号分隔)")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (default: 42)")
    p.add_argument("--tensorboard-log", type=str, default="./tensorboard_logs/",
                   help="TensorBoard 日志目录")
    p.add_argument("--checkpoint-path", type=str, default="./checkpoints/",
                   help="模型保存目录")
    p.add_argument("--no-lstm", action="store_true",
                   help="使用 Oracle 预测 (真实值+噪声) 代替 LSTM")
    p.add_argument("--use-grid2op", action="store_true", default=True,
                   help="使用 Grid2Op + Pandapower 三相潮流仿真 (默认)")
    p.add_argument("--no-grid2op", action="store_false", dest="use_grid2op",
                   help="使用 VoltageSimulator 简化电压模型")
    p.add_argument("--algo-backend", type=str, default="auto",
                   choices=["auto", "sb3", "numpy"],
                   help="RL 后端选择: auto=SB3 可用时优先 SB3, 否则 NumPy PPO; "
                        "sb3=强制 SB3; numpy=强制 NumPy PPO (默认 auto)")
    p.add_argument("--lstm-checkpoint", type=str, default=None,
                   help="预训练 LSTM 模型路径")
    p.add_argument("--export-onnx", action="store_true",
                   help="训练完成后自动导出 ONNX")
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


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── 依赖检测 ────────────────────────────────────────
    print("=" * 56)
    print("  MUPC RL 训练管线")
    print("=" * 56)

    has_sb3 = _check_sb3() and _check_gymnasium() and _check_torch()
    has_gym = _check_gymnasium()
    has_torch = _check_torch()
    print(f"\n依赖检测: SB3={'可用' if has_sb3 else '不可用(将使用NumPy PPO)'}, "
          f"Gymnasium={'可用' if has_gym else '不可用(使用_gym_stub)'}, "
          f"PyTorch={'可用' if has_torch else '不可用'}")

    # ── 数据加载 ────────────────────────────────────────
    print("\n── 数据加载 ──")
    from data_loader import SmartDSLoader
    loader = SmartDSLoader()
    data = loader.load_all()
    train_data, val_data = loader.split(data)

    # ── LSTM / Oracle ────────────────────────────────────
    predictor = None
    if not args.no_lstm and args.lstm_checkpoint:
        from lstm_model import LSTMForecast
        import torch as _torch_lstm
        print(f"\n── LSTM 模型加载 ──")
        model = LSTMForecast()
        model.load_state_dict(_torch_lstm.load(args.lstm_checkpoint, map_location="cpu"))
        model.lstm.eval()
        model.set_data(train_data)
        predictor = model
    elif not args.no_lstm and has_torch:
        print("\n── LSTM 模型训练 ──")
        from lstm_model import LSTMTrainer
        trainer = LSTMTrainer({"epochs": 50, "batch_size": 64})
        result = trainer.train(train_data, val_data)
        predictor = result["model"]
        predictor.set_data(train_data)  # 绑定 data 引用供 predict(step_idx) 使用
        # 保存 LSTM checkpoint
        lstm_path = os.path.join(args.checkpoint_path, "lstm_checkpoint.pt")
        os.makedirs(args.checkpoint_path, exist_ok=True)
        import torch
        torch.save(predictor.state_dict(), lstm_path)
        print(f"LSTM checkpoint 已保存: {lstm_path}")
    else:
        from lstm_model import OraclePredictor
        print("\n── 使用 Oracle 预测 ──")
        predictor = OraclePredictor(train_data)

    # ── 环境创建 ────────────────────────────────────────
    print("\n── 环境创建 ──")
    custom_weights = parse_custom_weights(args.reward_weights)

    from mupc_env import MupcEnv
    from config.config_manager import get_config
    cfg = get_config()
    env = MupcEnv(train_data, mode=args.mode, lstm_predictor=predictor,
                  config=cfg, use_grid2op=args.use_grid2op,
                  reward_weights=custom_weights if custom_weights else None)
    eval_env = MupcEnv(val_data, mode=args.mode, lstm_predictor=predictor,
                       config=cfg, use_grid2op=args.use_grid2op,
                       reward_weights=custom_weights if custom_weights else None)

    obs_dim = env.observation_space.shape[0]
    print(f"  观测空间: Box({obs_dim},)")
    act_dim = env.action_space.shape[0]
    print(f"  动作空间: Box({act_dim},), 模式: {args.mode}, 算法: {args.algo}")
    if args.use_grid2op:
        if env.use_grid2op:
            print(f"  电压仿真: Grid2Op + Pandapower 三相潮流")
        else:
            print(f"  电压仿真: VoltageSimulator 简化模型 (Grid2Op 不可用，已降级)")

    # ── 训练 ────────────────────────────────────────────
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    # v2.15 2 维同质 tanh 动作空间, SB3 MlpPolicy 完全支持
    backend = _resolve_backend(args.algo_backend, has_sb3)
    if backend == "sb3":
        print(f"  训练后端: stable-baselines3 (主路径)")
        _train_sb3(env, eval_env, args)
    else:
        print(f"  训练后端: NumPy PPO (后备路径)")
        _train_numpy_ppo(env, eval_env, args, obs_dim)

    print(f"\n{'=' * 56}")
    print(f"  训练完成。checkpoint: {args.checkpoint_path}")
    print(f"{'=' * 56}")

    # ── ONNX 导出 ──────────────────────────────────────
    if args.export_onnx and has_torch:
        print("\n── ONNX 导出 ──")
        try:
            import export_onnx
            onnx_path = export_onnx.export_rl_policy(
                checkpoint_dir=args.checkpoint_path,
                output_dir="./exported_models/",
                obs_dim=obs_dim,
            )
            print(f"ONNX 模型已导出: {onnx_path}")
        except Exception as e:
            print(f"ONNX 导出失败: {e}")


# ── SB3 训练 ──────────────────────────────────────────────────

def _resolve_backend(requested: str, has_sb3: bool) -> str:
    """解析训练后端选择: auto → sb3 (若可用) / numpy, 否则按请求。"""
    if requested == "auto":
        return "sb3" if has_sb3 else "numpy"
    if requested == "sb3" and not has_sb3:
        print("[WARN] --algo-backend=sb3 但 SB3/Gymnasium/Torch 不可用, 降级到 NumPy PPO",
              file=sys.stderr)
        return "numpy"
    return requested


def _train_sb3(env, eval_env, args):
    import gymnasium
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    algo_cls = PPO if args.algo == "ppo" else SAC
    algo_name = "PPO" if args.algo == "ppo" else "SAC"

    print(f"\n── SB3 {algo_name} 训练 ──")
    print(f"  总步数: {args.total_timesteps}, 模式: {args.mode}")

    # 包装环境
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    vec_eval = DummyVecEnv([lambda: Monitor(eval_env)])

    eval_callback = EvalCallback(
        vec_eval, best_model_save_path=args.checkpoint_path,
        log_path=args.checkpoint_path, eval_freq=10000,
        deterministic=True, render=False,
    )

    # 网络架构
    import torch.nn as nn
    policy_kwargs = {
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "activation_fn": nn.ReLU,
    }

    model = algo_cls(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2 if args.algo == "ppo" else None,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            tb_log_name=f"mupc_{args.algo}_{args.mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}",
        )
        # 保存最终模型
        checkpoint_path = os.path.join(args.checkpoint_path, "final_model")
        model.save(checkpoint_path)
        print(f"最终模型已保存: {checkpoint_path}.zip")
    except KeyboardInterrupt:
        checkpoint_path = os.path.join(args.checkpoint_path, "interrupted_model")
        model.save(checkpoint_path)
        print(f"\n训练中断, 模型已保存: {checkpoint_path}.zip")
        sys.exit(0)


# ── NumPy PPO 训练 ────────────────────────────────────────────

def _train_numpy_ppo(env, eval_env, args, obs_dim):
    from _ppo_core import NumPyPPO

    print(f"\n── NumPy PPO 训练 ──")
    print(f"  总步数: {args.total_timesteps}, 模式: {args.mode}")

    model = NumPyPPO(env, obs_dim=obs_dim)

    try:
        log = model.learn(args.total_timesteps)
        # 保存权重
        weights_path = os.path.join(args.checkpoint_path, "ppo_weights.npz")
        model.save_weights(weights_path)
        print(f"权重已保存: {weights_path}")
    except KeyboardInterrupt:
        weights_path = os.path.join(args.checkpoint_path, "ppo_weights_interrupted.npz")
        model.save_weights(weights_path)
        print(f"\n训练中断, 权重已保存: {weights_path}")
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
