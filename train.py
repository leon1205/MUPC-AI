"""
MUPC RL 模型训练主入口。

用法:
  python train.py                          # 默认多模式 PPO 训练
  python train.py --mode MODE-01           # 单模式 (农网灌溉)
  python train.py --algo sac               # SAC 算法
  python train.py --total-timesteps 50000  # 快速测试
  python train.py --reward-weights MODE-01=1.5,0.3,3.0  # 自定义权重
  python train.py --no-lstm                # 使用 Oracle 预测
  python train.py --config tmp.json        # v3.0: MSSA 超参配置文件
  python train.py --mic mic_result.json    # v3.0: MIC 特征筛选结果
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
    p.add_argument("--train-lstm", action="store_true",
                   help="仅训练 LSTM + 误差修正, 不跑 RL (供 MSSA/MIC 使用)")
    p.add_argument("--no-error-correction", action="store_true",
                   help="跳过 BiLSTM 误差修正训练 (v3.1, 默认启用, 约增1-2分钟)")
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
    p.add_argument("--data-source", type=str, default="smartds",
                   choices=["smartds", "china", "unified"],
                   help="数据源: smartds/china/unified (default: smartds)")
    p.add_argument("--export-onnx", action="store_true",
                   help="训练完成后自动导出 ONNX")
    # v3.0: MSSA 接口
    p.add_argument("--config", type=str, default=None,
                   help="MSSA 生成的临时训练配置文件 (JSON)")
    p.add_argument("--mic", type=str, default=None,
                   help="MIC 离线分析输出 JSON 文件路径 (v3.0)")
    p.add_argument("--mssa-result", type=str, default=None,
                   help="MSSA 搜索结果 JSON 文件路径 (v3.0)")
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


# ── v3.0: MSSA 配置解析 ────────────────────────────────────

def parse_mssa_config(config_path: str) -> dict:
    """解析 MSSA 生成的临时训练配置文件 (JSON).

    Raises SystemExit(1) 若缺少必须字段.
    """
    REQUIRED = ["hidden_size", "num_layers", "lr", "batch_size"]
    with open(config_path) as f:
        cfg = json.load(f)
    for k in REQUIRED:
        if k not in cfg:
            sys.stderr.write(f"[FATAL] Missing required config key: {k}\n")
            sys.exit(1)
    return cfg

def load_mic_features(json_path: str) -> tuple[list[str], int]:
    """读取 MIC 分析结果, 返回 (selected 特征名列表, top_k).

    若实际选中数 != top_k, 输出 WARN.
    """
    with open(json_path) as f:
        mic = json.load(f)
    selected = [feat["name"] for feat in mic["features"] if feat["selected"]]
    top_k = mic["top_k"]
    if len(selected) != top_k:
        sys.stderr.write(f"[WARN] MIC selected {len(selected)} features != top_k {top_k}\n")
    return selected, top_k

def load_mssa_result(json_path: str) -> dict:
    """读取 MSSA 搜索结果, 返回最优超参字典."""
    with open(json_path) as f:
        mssa = json.load(f)
    if mssa.get("quality_flag") == "unusable":
        sys.stderr.write("[WARN] MSSA quality_flag=unusable, using manual baseline\n")
    return mssa["best_hyperparameters"]


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
    if args.data_source == "china":
        from data_loader import UnifiedDataLoader
        loader = UnifiedDataLoader(lat=31.23, lon=121.47, auto_generate=True)
        print("  数据源: 中国合成数据 (上海)")
    elif args.data_source == "unified":
        from data_loader import UnifiedDataLoader
        loader = UnifiedDataLoader(lat=31.23, lon=121.47, auto_generate=True)
        print("  数据源: 统一自动检测")
    else:
        from data_loader import SmartDSLoader
        loader = SmartDSLoader()
        print("  数据源: SMART-DS")
    data = loader.load_all()
    train_data, val_data = loader.split(data)

    # ── v3.0: MSSA 配置覆盖 ────────────────────────────────
    lstm_cfg = {
        "epochs": 200, "batch_size": 64,
        "output_mode": "p10p50p90",     # v3.0: 分位数三通道预测
        "with_attention": True,          # v3.0: AdditiveAttention
        "loss": "quantile",              # v3.1: 真分位数回归 (QuantileLoss)
        "input_window": 24,              # v3.0: 输入窗口步数
        "patience": 30,                  # 早停
        "with_tcn": True,                # v3.1: TCN 特征提取 (R2, 默认开启)
    }
    if args.config:
        mssa_cfg = parse_mssa_config(args.config)
        # v3.0: MSSA 键名 → 训练配置键名映射 (全部 10 个超参)
        KEY_MAP = {
            "hidden_size": "hidden_dim",
            "lr": "learning_rate",
            "attn_score": "attn_score",
            "vmd_k": "vmd_k",
            "vmd_alpha": "vmd_alpha",
            "optimizer": "optimizer",
        }
        KNOWN_KEYS = {"hidden_dim", "num_layers", "learning_rate", "batch_size",
                       "dropout", "input_window", "output_mode", "with_attention",
                       "attn_score", "vmd_k", "vmd_alpha", "vmd_enabled", "optimizer",
                       "with_tcn"}    # v3.1: TCN 特征提取 (R2)
        mapped = {}
        for k, v in mssa_cfg.items():
            if k not in KEY_MAP and k not in KNOWN_KEYS:
                sys.stderr.write(f"[WARN] MSSA key '{k}' not recognized, skipping\n")
                continue
            target_k = KEY_MAP.get(k, k)
            mapped[target_k] = v
        lstm_cfg.update({k: v for k, v in mapped.items() if k in KNOWN_KEYS})

    # v3.1: MIC 特征筛选 → 传递给 LSTMTrainer
    mic_features = None
    if args.mic:
        selected, top_k = load_mic_features(args.mic)
        print(f"\n── MIC 特征筛选 ──")
        print(f"  选中 {len(selected)}/{top_k} 特征: {selected}")
        mic_features = selected
        lstm_cfg["mic_features"] = selected
        lstm_cfg["mic_top_k"] = top_k

    # v3.1: MSSA 搜索结果 → 覆盖 lstm_cfg
    if args.mssa_result:
        mssa_best = load_mssa_result(args.mssa_result)
        print(f"\n── MSSA 搜索结果加载 ──")
        # 将 MSSA 最优超参映射到 lstm_cfg (仅覆盖未在 --config 中指定的键)
        mssa_key_map = {"hidden_size": "hidden_dim", "lr": "learning_rate"}
        for mssa_k, cfg_k in mssa_key_map.items():
            if mssa_k in mssa_best and cfg_k not in lstm_cfg:
                lstm_cfg[cfg_k] = mssa_best[mssa_k]
        for k in ["num_layers", "batch_size", "input_window", "dropout"]:
            if k in mssa_best and k not in lstm_cfg:
                lstm_cfg[k] = mssa_best[k]
        print(f"  已合并超参: { {k: lstm_cfg[k] for k in ['hidden_dim','num_layers','input_window'] if k in lstm_cfg} }")

    # ── LSTM / Oracle ────────────────────────────────────
    predictor = None
    lstm_metrics = {}  # v3.0: 训练指标供 stdout 输出
    if not args.no_lstm and args.lstm_checkpoint:
        from models.lstm import LSTMForecast
        import torch as _torch_lstm
        print(f"\n── LSTM 模型加载 ──")
        model = LSTMForecast()
        model.load_state_dict(_torch_lstm.load(args.lstm_checkpoint, map_location="cpu"))
        model.lstm.eval()
        model.set_data(train_data)
        predictor = model
    elif not args.no_lstm and has_torch:
        print("\n── LSTM 模型训练 ──")
        try:
            from models.lstm import LSTMTrainer
            trainer = LSTMTrainer(lstm_cfg)
            result = trainer.train(train_data, val_data)
            predictor = result["model"]
            predictor.set_data(train_data)
            lstm_metrics = result.get("metrics", {})

            # 保存 LSTM checkpoint
            lstm_path = os.path.join(args.checkpoint_path, "lstm_checkpoint.pt")
            os.makedirs(args.checkpoint_path, exist_ok=True)
            import torch
            torch.save(predictor.state_dict(), lstm_path)
            print(f"LSTM checkpoint 已保存: {lstm_path}")

            # v3.1: BiLSTM 误差修正训练
            if not args.no_error_correction:
                try:
                    from models.error_correction import ErrorCorrectionTrainer
                    print("\n── BiLSTM 误差修正训练 ──")
                    ec_trainer = ErrorCorrectionTrainer()
                    ec_result = ec_trainer.train(train_data, predictor.predict_numpy)
                    if ec_result.get("model") is not None:
                        ec_path = os.path.join(args.checkpoint_path, "error_correction.pt")
                        torch.save(ec_result["model"].state_dict(), ec_path)
                        print(f"误差修正 checkpoint 已保存: {ec_path}")
                        bias_pv = ec_result.get("bias_pv", 0.0)
                        bias_load = ec_result.get("bias_load", 0.0)
                        print(f"  Bias Gate: 已启用 (PV={bias_pv:.1%}, Load={bias_load:.1%} > 3%)")
                    else:
                        bias_pv = ec_result.get("bias_pv", 0.0)
                        bias_load = ec_result.get("bias_load", 0.0)
                        skip_reason = "Bias不足" if ec_result.get("skip") else "未知"
                        print(f"  误差修正跳过: {skip_reason} (PV={bias_pv:.1%}, Load={bias_load:.1%})")
                except (ValueError, RuntimeError, ImportError) as e:
                    print(f"[WARN] 误差修正训练失败 (不影响主流程): {e}")
        except Exception as e:
            import traceback
            sys.stderr.write(f"[FATAL] LSTM training failed: {e}\n")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        from models.lstm import OraclePredictor
        print("\n── 使用 Oracle 预测 ──")
        predictor = OraclePredictor(train_data)

    # v3.1: --train-lstm 仅训练 LSTM + 误差修正, 不跑 RL (MSSA/MIC 工作流)
    if args.train_lstm:
        print(f"\n{'=' * 56}")
        print(f"  LSTM 训练完成 (--train-lstm, 跳过 RL)")
        if lstm_metrics:
            pv_mape = lstm_metrics.get("pv_mape_step1", lstm_metrics.get("pv_mape", -1.0))
            load_mape = lstm_metrics.get("load_mape_step1", lstm_metrics.get("load_mape", -1.0))
            print(f"PV_MAPE={pv_mape:.4f} LOAD_MAPE={load_mape:.4f}")
        print(f"{'=' * 56}")
        return

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

    # v3.0: stdout MAPE 输出 (供 MSSA 解析)
    if lstm_metrics:
        pv_mape = lstm_metrics.get("pv_mape_step1", lstm_metrics.get("pv_mape", -1.0))
        load_mape = lstm_metrics.get("load_mape_step1", lstm_metrics.get("load_mape", -1.0))
        print(f"PV_MAPE={pv_mape:.4f} LOAD_MAPE={load_mape:.4f}")

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
            # v3.1: 同时导出误差修正模型
            ec_checkpoint = os.path.join(args.checkpoint_path, "error_correction.pt")
            if os.path.exists(ec_checkpoint):
                try:
                    ec_onnx = export_onnx.export_error_correction(
                        ec_checkpoint, "./exported_models/")
                    print(f"误差修正 ONNX 已导出: {ec_onnx}")
                except Exception as e:
                    print(f"[WARN] 误差修正 ONNX 导出失败: {e}")
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


# ── v3.1: 调度时段统计回调 ──────────────────────────────────

class DispatchStatsCallback:
    """v3.1: 累计调度接管时段占比，训练结束时输出。
    通过包装 EvalCallback 的 _on_step 收集 dispatched 标记。
    """
    def __init__(self):
        self._total_steps = 0
        self._dispatched_steps = 0

    def record(self, infos: list[dict]):
        for info in infos:
            self._total_steps += 1
            if info.get("dispatched", False):
                self._dispatched_steps += 1

    @property
    def dispatch_ratio(self) -> float:
        if self._total_steps == 0:
            return 0.0
        return self._dispatched_steps / self._total_steps

    def summary(self) -> str:
        return (f"调度接管: {self._dispatched_steps}/{self._total_steps} 步 "
                f"({self.dispatch_ratio:.1%})")


def _train_sb3(env, eval_env, args):
    import gymnasium
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # v3.1: 调度时段统计
    dispatch_stats = DispatchStatsCallback()

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

    # v3.1: 包装 eval 环境以收集 dispatch stats
    from stable_baselines3.common.callbacks import BaseCallback
    class _DispatchWrapper(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dispatch_stats.record(infos)
            return True

    dispatch_cb = _DispatchWrapper()
    callbacks = [eval_callback, dispatch_cb]

    model = algo_cls(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,       # v2.18 调优: 原始默认, unified数据量充足
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
            callback=callbacks,
            tb_log_name=f"mupc_{args.algo}_{args.mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}",
        )
        # 保存最终模型
        checkpoint_path = os.path.join(args.checkpoint_path, "final_model")
        model.save(checkpoint_path)
        print(f"最终模型已保存: {checkpoint_path}.zip")
        if dispatch_stats._total_steps > 0:
            print(f"  {dispatch_stats.summary()}")
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
