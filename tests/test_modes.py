"""Verify different modes give different training results."""
import os, sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from data_loader import SmartDSLoader
from mupc_env import MupcEnv
from models.lstm import OraclePredictor
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

if __name__ != "__main__":
    raise SystemExit("This script is meant to be run directly, not imported")

loader = SmartDSLoader()
train, val = loader.split(loader.load_all())

class EnvMaker:
    def __init__(self, env): self.env = env
    def __call__(self): return Monitor(self.env)

policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": nn.ReLU}

for mode in ["MODE-02", "MODE-05"]:
    env = MupcEnv(train, mode=mode, lstm_predictor=OraclePredictor(train))
    eval_env = MupcEnv(val, mode=mode, lstm_predictor=OraclePredictor(val))
    vec_eval = DummyVecEnv([EnvMaker(eval_env)])

    model = PPO("MlpPolicy", DummyVecEnv([EnvMaker(env)]),
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                policy_kwargs=policy_kwargs, seed=42, verbose=0)
    model.learn(total_timesteps=20000)

    obs = vec_eval.reset()
    rewards = []
    for _ in range(960):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = vec_eval.step(act)
        rewards.extend(rew.tolist())
    print(f"{mode}: eval={sum(rewards)/10:.1f}")
