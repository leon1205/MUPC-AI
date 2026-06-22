"""统一模型测试集评估 — 每模型跑 100 episodes 统计各项指标。"""
import sys, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  Unified Model Evaluation on Test Set")
print("=" * 60)

# 加载验证数据
from train_unified import _load_all
_, val_data = _load_all()
from models.lstm import OraclePredictor
predictor = OraclePredictor(val_data)
print(f"\nTest set: {val_data['n_steps']} steps")

from mupc_env import MupcEnv
from stable_baselines3 import PPO

MODES = ["MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"]
MODE_LABELS = {
    "MODE-01": "农网灌溉", "MODE-02": "自主套利",
    "MODE-03": "需量控制", "MODE-04": "虚拟电厂", "MODE-05": "极致绿色",
}
N_EPISODES = 100

import gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class EnvMaker:
    def __init__(self, env): self.env = env
    def __call__(self): return Monitor(self.env)

all_results = {}

for mode in MODES:
    model_path = f"checkpoints/unified_{mode}_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"\n  {mode}: model not found, skip")
        continue

    model = PPO.load(model_path)
    env = MupcEnv(val_data, mode=mode, lstm_predictor=predictor)

    ep_rewards = []
    ep_soc_final = []
    ep_overloads = []
    ep_pv_mean = []
    ep_p_batt_mean = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        total_r = 0.0
        overload_count = 0
        soc_vals = []
        pv_vals = []
        p_batt_vals = []

        for _ in range(96):  # 1 day
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            total_r += r
            soc_vals.append(info["soc"])
            pv_vals.append(info.get("pv_power", 0))
            p_batt_vals.append(info["p_batt"])
            if info["load_rate"] > 0.85:
                overload_count += 1
            if term or trunc:
                break

        ep_rewards.append(total_r)
        ep_soc_final.append(soc_vals[-1])
        ep_overloads.append(overload_count)
        ep_pv_mean.append(np.mean(pv_vals))
        ep_p_batt_mean.append(np.mean(p_batt_vals))

    results = {
        "reward": (np.mean(ep_rewards), np.std(ep_rewards)),
        "soc_final": (np.mean(ep_soc_final), np.std(ep_soc_final)),
        "overloads_per_ep": np.mean(ep_overloads),
        "p_batt_mean_kw": np.mean(ep_p_batt_mean),
        "max_reward": np.max(ep_rewards),
        "min_reward": np.min(ep_rewards),
    }
    all_results[mode] = results

# 打印对比表
print(f"\n{'='*60}")
print(f"  Results ({N_EPISODES} episodes per mode, deterministic policy)")
print(f"{'='*60}")

header = f"{'Mode':<8s} {'场景':<10s} {'Reward':>12s} {'Max':>8s} {'Min':>8s} {'SOC_final':>10s} {'过载/集':>8s} {'P_batt均值':>10s}"
print(header)
print("-" * len(header))

for mode in MODES:
    r = all_results.get(mode)
    if r is None:
        continue
    print(f"{mode:<8s} {MODE_LABELS[mode]:<10s} "
          f"{r['reward'][0]:>7.1f}±{r['reward'][1]:>3.1f} "
          f"{r['max_reward']:>7.1f} "
          f"{r['min_reward']:>7.1f} "
          f"{r['soc_final'][0]:>7.2f}±{r['soc_final'][1]:.2f} "
          f"{r['overloads_per_ep']:>7.1f} "
          f"{r['p_batt_mean_kw']:>7.0f}kW")

print(f"\n{'='*60}")
print("  整体统计")
print(f"{'='*60}")
rewards_all = [r["reward"][0] for r in all_results.values()]
print(f"  平均奖励: {np.mean(rewards_all):.1f}")
print(f"  最稳定:    {min(all_results.items(), key=lambda x: x[1]['reward'][1])[0]} "
      f"(std={min(r['reward'][1] for r in all_results.values()):.2f})")
