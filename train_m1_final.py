"""MODE-01 最终训练: LSTM预测 + 三数据集合并 (SMART-DS + Aug + China)。"""
import sys, os, time, math
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 56)
print("  MODE-01 Final: LSTM + Unified Data (3 sources)")
print("=" * 56)

# 1. 加载统一数据
from train_unified import _load_all
train_data, val_data = _load_all()

# 2. 加载 LSTM + 适配器
import torch
from lstm_model import LSTMForecast

print("\nLoading LSTM model...")
lstm_model = LSTMForecast()
lstm_model.load_state_dict(torch.load("checkpoints/lstm_model.pt", map_location="cpu"))
lstm_model.to("cpu")
lstm_model.lstm.eval()

class LSTMAdapter:
    def __init__(self, model, data):
        self.model = model
        self.pv = data["pv_power"]
        self.load = data["load_power"]
        self.ghi = data.get("solar_irradiance", np.zeros_like(self.pv))
        self.temp = data.get("temperature", np.zeros_like(self.pv))
        hours_raw = np.arange(len(self.pv)) * 15 / 60 % 24
        self.hours = hours_raw.astype(np.float32)
        self.n = len(self.pv)

    def predict(self, step_idx: int) -> np.ndarray:
        seq_len = 4
        x = np.zeros((1, seq_len, 6), dtype=np.float32)
        for j in range(seq_len):
            idx = max(0, min(step_idx - seq_len + 1 + j, self.n - 1))
            h = self.hours[idx]
            x[0, j, 0] = self.pv[idx]
            x[0, j, 1] = self.load[idx]
            x[0, j, 2] = self.ghi[idx]
            x[0, j, 3] = self.temp[idx]
            x[0, j, 4] = math.sin(h * 2 * math.pi / 24)
            x[0, j, 5] = math.cos(h * 2 * math.pi / 24)
        pred = self.model.predict_numpy(x)
        return pred.ravel().astype(np.float32)

lstm_train = LSTMAdapter(lstm_model, train_data)
lstm_val = LSTMAdapter(lstm_model, val_data)
print("LSTM adapter ready")

# 3. 训练
from mupc_env import MupcEnv
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

class EnvMaker:
    def __init__(self, env): self.env = env
    def __call__(self): return Monitor(self.env)

env = MupcEnv(train_data, mode="MODE-01", lstm_predictor=lstm_train)
eval_env = MupcEnv(val_data, mode="MODE-01", lstm_predictor=lstm_val)
vec_env = DummyVecEnv([EnvMaker(env)])
vec_eval = DummyVecEnv([EnvMaker(eval_env)])

eval_cb = EvalCallback(vec_eval, best_model_save_path="checkpoints/",
                        log_path="checkpoints/", eval_freq=50000,
                        deterministic=True)

pk = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": nn.ReLU}
model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
            batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, policy_kwargs=pk,
            tensorboard_log="tensorboard_logs/", seed=42, verbose=1)

STEPS = 300_000
print(f"\nTraining: MODE-01, {STEPS} steps, LSTM predictor, unified data")
t0 = time.time()
model.learn(total_timesteps=STEPS, callback=eval_cb,
            tb_log_name=f"m1_final_lstm_{time.strftime('%Y%m%d_%H%M%S')}")
elapsed = time.time() - t0

model.save("checkpoints/unified_MODE-01_model")
print(f"\nDone in {elapsed:.0f}s -> checkpoints/unified_MODE-01_model.zip")
