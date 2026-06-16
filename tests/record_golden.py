"""录制 Golden Test 基线数据。拆分前运行一次，锁定当前行为。

Usage: python tests/record_golden.py
Output: tests/golden/mupc_env_baseline.npz
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from data_loader import SmartDSLoader
from mupc_env import MupcEnv


def record_single_mode(env, n_steps=100, seed=42):
    """录制约定性单模式轨迹。"""
    np.random.seed(seed)
    obs_list, reward_list, info_keys = [], [], []

    # 同时记录 reset 后的初始 obs
    obs, info = env.reset(seed=seed)
    obs_list.append(obs.copy())

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs.copy())
        reward_list.append(reward)
        # 只记录数值型 info key
        info_snapshot = {}
        for k, v in info.items():
            if isinstance(v, (float, int, np.floating, np.integer, bool)):
                info_snapshot[k] = float(v)
        info_keys.append(info_snapshot)
        if terminated or truncated:
            break

    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "rewards": np.array(reward_list, dtype=np.float32),
        "info_keys": info_keys,
        "n_steps": len(reward_list),
    }


def record_multi_mode(env, n_steps=200, seed=123):
    """录制多模式轨迹。"""
    np.random.seed(seed)
    obs_list, reward_list, modes = [], [], []

    obs, info = env.reset(seed=seed)
    obs_list.append(obs.copy())

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs.copy())
        reward_list.append(reward)
        modes.append(str(info.get("mode", "")))
        if terminated or truncated:
            env.reset()

    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "rewards": np.array(reward_list, dtype=np.float32),
        "modes": np.array(modes),
        "n_steps": len(reward_list),
    }


def main():
    print("加载数据...")
    loader = SmartDSLoader()
    data = loader.load_all()
    train, _val = loader.split(data)

    # 单模式基线
    print("录制单模式 (MODE-01) baseline...")
    env1 = MupcEnv(train, mode="MODE-01")
    single = record_single_mode(env1, n_steps=100, seed=42)
    print(f"  obs shape: {single['obs'].shape}, rewards: {single['n_steps']} steps")

    # 多模式基线
    print("录制多模式 (all) baseline...")
    env_all = MupcEnv(train, mode="all")
    multi = record_multi_mode(env_all, n_steps=200, seed=123)
    print(f"  obs shape: {multi['obs'].shape}, rewards: {multi['n_steps']} steps")
    print(f"  modes: {sorted(set(str(m) for m in multi['modes']))}")

    # 保存
    outpath = "tests/golden/mupc_env_baseline.npz"
    np.savez_compressed(
        outpath,
        # single mode
        single_obs=single["obs"],
        single_rewards=single["rewards"],
        single_n_steps=single["n_steps"],
        # multi mode
        multi_obs=multi["obs"],
        multi_rewards=multi["rewards"],
        multi_n_steps=multi["n_steps"],
        multi_modes=multi["modes"],
    )
    print(f"\n[OK] Golden baseline saved to {outpath}")


if __name__ == "__main__":
    main()
