"""方案 A: 中国数据集训练 — 12城市 × 5建筑 × 12省电价。"""
import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from data_loader import ChinaDataLoader
from mupc_env import MupcEnv
from lstm_model import OraclePredictor

print("=" * 56)
print("  Plan A: China Dataset Training (12 cities)")
print("=" * 56)

loader = ChinaDataLoader()
train_data, val_data = loader.split(loader.load_all())
print(f"Train: {train_data['n_steps']} steps ({train_data['n_steps']*15/60/24:.0f} days)")
print(f"Val:   {val_data['n_steps']} steps ({val_data['n_steps']*15/60/24:.0f} days)")

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

MODES = ["MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"]
TOTAL_STEPS = 100_000

for mode in MODES:
    print(f"\n{'='*56}")
    print(f"  China: {mode} ({TOTAL_STEPS} steps)")
    print(f"{'='*56}")

    env = MupcEnv(train_data, mode=mode, lstm_predictor=predictor)
    eval_env = MupcEnv(val_data, mode=mode, lstm_predictor=predictor_eval)
    vec_env = DummyVecEnv([EnvMaker(env)])
    vec_eval = DummyVecEnv([EnvMaker(eval_env)])

    eval_cb = EvalCallback(vec_eval, best_model_save_path="checkpoints/",
                           log_path="checkpoints/", eval_freq=25000,
                           deterministic=True)

    policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                     "activation_fn": nn.ReLU}

    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs,
                tensorboard_log="tensorboard_logs/", seed=42, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb,
                tb_log_name=f"china_{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    elapsed = time.time() - t0

    model_path = f"checkpoints/china_{mode}_model"
    model.save(model_path)
    print(f"  {mode} done in {elapsed:.0f}s -> {model_path}.zip")

print(f"\n{'='*56}")
print("  China training complete!")
print(f"{'='*56}")
for m in MODES:
    print(f"  {m}: checkpoints/china_{m}_model.zip")
