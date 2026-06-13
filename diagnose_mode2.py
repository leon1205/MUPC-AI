"""MODE-02 套利模式诊断 — 分析为什么 reward 偏低。"""
import os, math
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data_loader import SmartDSLoader
from mupc_env import MupcEnv
from lstm_model import OraclePredictor
from stable_baselines3 import PPO

loader = SmartDSLoader()
data = loader.load_all()
_, val = loader.split(data)
predictor = OraclePredictor(val)

model = PPO.load("checkpoints/unified_MODE-02_model")
env = MupcEnv(val, mode="MODE-02", lstm_predictor=predictor)

# 跑 1 个 episode, 记录每步详细信息
obs, _ = env.reset()
steps = []
for t in range(96):
    act, _ = model.predict(obs, deterministic=True)
    obs, r, _, _, info = env.step(act)
    steps.append({
        "t": t,
        "hour": (info.get("hour_encoded", 0) if "hour_encoded" in info
                 else t * 0.25 % 24),
        "price": info.get("current_price", val["current_electricity_price"][env._step_idx-1]),
        "tariff": info.get("tariff_id", 0),
        "p_batt": info["p_batt"],
        "q_batt": info["q_batt"],
        "pv": val["pv_power"][env._step_idx-1] if env._step_idx > 0 else 0,
        "load": val["load_power"][env._step_idx-1] if env._step_idx > 0 else 0,
        "soc": info["soc"],
        "load_rate": info["load_rate"],
        "reward": r,
        "overload": info["load_rate"] > 0.85,
    })

# 分析
prices = np.array([s["price"] for s in steps])
p_batts = np.array([s["p_batt"] for s in steps])
socs = np.array([s["soc"] for s in steps])
rewards = np.array([s["reward"] for s in steps])
overloads = np.array([s["overload"] for s in steps])
load_rates = np.array([s["load_rate"] for s in steps])

# 分时段统计
peak_mask = prices > 1.0   # 峰+尖峰
valley_mask = prices < 0.6  # 谷
flat_mask = ~peak_mask & ~valley_mask

print("=" * 60)
print("  MODE-02 套利模式诊断")
print("=" * 60)

print(f"\n电价分布: 峰={peak_mask.sum()}步 平={flat_mask.sum()}步 谷={valley_mask.sum()}步")
print(f"价格范围: {prices.min():.2f}~{prices.max():.2f} yuan, 均值={prices.mean():.2f}")

print(f"\n分时段动作分析:")
for label, mask in [("峰时(>1.0)", peak_mask), ("平时", flat_mask), ("谷时(<0.6)", valley_mask)]:
    if mask.sum() == 0:
        continue
    p_avg = p_batts[mask].mean()
    charging = (p_batts[mask] < 0).sum()   # p_batt<0 = 充电
    discharging = (p_batts[mask] > 0).sum()  # p_batt>0 = 放电
    r_avg = rewards[mask].mean()
    ol = overloads[mask].sum()
    print(f"  {label:<12s}: {mask.sum():2d}步 | P_batt均值={p_avg:6.0f}kW | "
          f"充电{charging}步 放电{discharging}步 | 均奖励={r_avg:.2f} | 过载{ol}次")

print(f"\n物理约束分析:")
print(f"  电池容量: 200kWh, 可用: 80% = 160kWh")
print(f"  SOC范围: {socs.min():.2f}~{socs.max():.2f}")
print(f"  SOC变化: {socs[-1]-socs[0]:.2f} (净{'充电' if socs[-1]>socs[0] else '放电'})")
print(f"  最大单步充放电: {abs(p_batts).max():.0f}kW")
print(f"  过载步数: {overloads.sum()}/{len(steps)}")

print(f"\n理论最大套利收益:")
# 峰谷价差 × 可用容量
price_spread = 1.2 - 0.4  # yuan
max_energy_shift = 160  # kWh per cycle
max_daily = price_spread * max_energy_shift  # yuan/day
print(f"  峰谷价差: {price_spread:.1f} yuan/kWh")
print(f"  单次满充放收益: {max_daily:.0f} yuan (160kWh × 0.8yuan)")
print(f"  归一化奖励约: {max_daily / (500 * 0.4):.1f} (每步理论上限)")

print(f"\n总奖励: {rewards.sum():.1f}")
print(f"过载率: {overloads.mean()*100:.0f}%")

# 关键判断
print(f"\n诊断结论:")
if overloads.sum() > 30:
    print(f"  ❌ 过载过多 ({overloads.sum()}步) — 充放电导致变压器超载, 安全惩罚吃掉套利收益")
if abs(p_batts[valley_mask].mean()) < 50 and valley_mask.sum() > 0:
    print(f"  ❌ 谷时充电不够积极 ({abs(p_batts[valley_mask].mean()):.0f}kW)")
if abs(p_batts[peak_mask].mean()) < 50 and peak_mask.sum() > 0:
    print(f"  ❌ 峰时放电不够积极 ({abs(p_batts[peak_mask].mean()):.0f}kW)")
if peak_mask.sum() + valley_mask.sum() < 20:
    print(f"  ⚠️ 峰谷时段太少 ({peak_mask.sum()+valley_mask.sum()}步) — 套利机会不足")

# 计算如果完美套利能得多少
ideal_r = 0
soc_sim = 0.5
for t in range(96):
    p = prices[t]
    # 完美策略: 谷充峰放, 受 SOC 约束
    if p < 0.6 and soc_sim < 0.9:
        charge = min(500, (0.9 - soc_sim) * 200 / 0.25)
        r_spread = -charge * (p - 0.8) / (500 * 0.4)
        soc_sim += charge * 0.25 / 200
    elif p > 1.0 and soc_sim > 0.1:
        discharge = min(500, (soc_sim - 0.1) * 200 / 0.25)
        r_spread = discharge * (p - 0.8) / (500 * 0.4)
        soc_sim -= discharge * 0.25 / 200
    else:
        r_spread = 0
    ideal_r += r_spread - 0.5 * abs(soc_sim - 0.5) * 0.1
print(f"\n理想完美策略奖励 (估算): {ideal_r:.1f}")
print(f"模型实际奖励: {rewards.sum():.1f} ({rewards.sum()/max(ideal_r,1)*100:.0f}% of ideal)")
