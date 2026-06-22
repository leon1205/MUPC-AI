"""MODE-01 诊断脚本。"""
import os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from data_loader import SmartDSLoader
from mupc_env import MupcEnv
from models.lstm import OraclePredictor
from stable_baselines3 import PPO

loader = SmartDSLoader()
_, val = loader.split(loader.load_all())
model = PPO.load("checkpoints/unified_MODE-01_model")
env = MupcEnv(val, mode="MODE-01", lstm_predictor=OraclePredictor(val))

all_r = []
for ep in range(10):
    obs, _ = env.reset()
    pvs, loads, pbs, ols, rwds = [], [], [], [], []
    for t in range(96):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, _, _, info = env.step(act)
        pvs.append(val["pv_power"][env._step_idx - 1])
        loads.append(val["load_power"][env._step_idx - 1])
        pbs.append(info["p_batt"])
        ols.append(info["load_rate"] > 0.85)
        rwds.append(r)
    pv = np.array(pvs)
    ld = np.array(loads)
    pb = np.array(pbs)
    ol = np.array(ols)
    sun = pv > 30
    if sun.sum() > 0:
        pv_wasted = np.maximum(0, pv[sun] - ld[sun])
        util = 100 * (1 - pv_wasted.sum() / pv[sun].sum())
    else:
        util = 0
    all_r.append((sum(rwds), util, ol.mean() * 100))

rr = np.array(all_r)
print(f"MODE-01 10ep avg:")
print(f"  Reward:     {rr[:,0].mean():.1f} +/- {rr[:,0].std():.1f}")
print(f"  PV util:    {rr[:,1].mean():.0f}% +/- {rr[:,1].std():.0f}%")
print(f"  Overload:   {rr[:,2].mean():.0f}%")
print(f"  Theoretical max: 96.0  Achieved: {rr[:,0].mean()/96*100:.0f}%")
