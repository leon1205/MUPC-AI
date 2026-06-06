"""方案 A v2: 为每个场景训练独立模型, 使用 EnvMaker 避免闭包问题。"""
import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from data_loader import SmartDSLoader
from mupc_env import MupcEnv
from lstm_model import OraclePredictor

print("=" * 56)
print("  Plan A v2: Single-mode models for MODE-02~05")
print("=" * 56)

loader = SmartDSLoader()
train_data, val_data = loader.split(loader.load_all())
predictor = OraclePredictor(train_data)
predictor_eval = OraclePredictor(val_data)

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

class EnvMaker:
    def __init__(self, env): self.env = env
    def __call__(self): return Monitor(self.env)

MODES = ["MODE-02", "MODE-03", "MODE-04", "MODE-05"]
TOTAL_STEPS = 80_000

for mode in MODES:
    print(f"\n{'='*56}")
    print(f"  {mode}")
    print(f"{'='*56}")

    env = MupcEnv(train_data, mode=mode, lstm_predictor=predictor)
    eval_env = MupcEnv(val_data, mode=mode, lstm_predictor=predictor_eval)

    vec_env = DummyVecEnv([EnvMaker(env)])
    vec_eval = DummyVecEnv([EnvMaker(eval_env)])

    eval_cb = EvalCallback(vec_eval, best_model_save_path="checkpoints/",
                           log_path="checkpoints/", eval_freq=20000,
                           deterministic=True)

    policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                     "activation_fn": nn.ReLU}

    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs,
                tensorboard_log="tensorboard_logs/", seed=42, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb,
                tb_log_name=f"mupc_v2_{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    elapsed = time.time() - t0

    model_path = f"checkpoints/{mode}_model"
    model.save(model_path)
    print(f"  {mode} done in {elapsed:.0f}s -> {model_path}.zip")

print(f"\n{'='*56}")
print("  Plan A v2 Complete!")
print(f"{'='*56}")
print("  MODE-01: checkpoints/final_model.zip (Round 1)")
for m in MODES:
    print(f"  {m}: checkpoints/{m}_model.zip")
