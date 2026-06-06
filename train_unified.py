"""统一模型训练 — SMART-DS + 增强 + China 三数据集合并。

为每个场景训练一个模型，数据覆盖:
  - SMART-DS 原始 (旧金山真实测量, 14万步)
  - SMART-DS 增强 (时间平移/PV缩放, 56万步)
  - China 合成 (12城市, 126万步)
  总计约 196 万步
"""

import sys, os, time
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _concat_datasets(*datasets: dict) -> dict:
    """沿时间轴拼接多个数据集。"""
    keys = datasets[0].keys()
    merged = {}
    for key in keys:
        if key in ("n_steps", "norm_params"):
            continue
        arrays = []
        for d in datasets:
            if key in d and isinstance(d[key], np.ndarray):
                arrays.append(d[key])
        if arrays:
            merged[key] = np.concatenate(arrays).astype(np.float32)

    # 重新生成时间特征 (统一长度)
    n = len(merged["pv_power"])
    merged["n_steps"] = n
    merged["timestamps"] = np.arange(n, dtype=np.float64)
    hours = (np.arange(n) * 15 / 60) % 24
    merged["hours"] = hours.astype(np.float32)
    merged["month"] = (np.arange(n) // (96 * 30) % 12 + 1).astype(np.float32)
    merged["hour_encoded"] = np.sin(hours * 2 * np.pi / 24).astype(np.float32)

    # 补充源数据中确实不存在的字段 (仅当 key 不存在时才设置)
    _defaults = {
        "grid_power": np.zeros(n, dtype=np.float32),
        "transformer_load": np.zeros(n, dtype=np.float32),
        "battery_power": np.zeros(n, dtype=np.float32),
        "voltage_phase_a": np.ones(n, dtype=np.float32),
        "voltage_phase_b": np.ones(n, dtype=np.float32),
        "voltage_phase_c": np.ones(n, dtype=np.float32),
        "pv_forecast": np.zeros(n, dtype=np.float32),
        "load_forecast": np.zeros(n, dtype=np.float32),
        "current_electricity_price": np.zeros(n, dtype=np.float32),
        "next_period_price": np.zeros(n, dtype=np.float32),
        "price_tariff_id": np.zeros(n, dtype=np.int32),
        "current_demand": np.zeros(n, dtype=np.float32),
        "contract_demand": np.full(n, 300.0, dtype=np.float32),
        "peak_demand_this_month": np.zeros(n, dtype=np.float32),
        "solar_irradiance": np.zeros(n, dtype=np.float32),
        "temperature": np.zeros(n, dtype=np.float32),
        "dispatch_p_set": np.zeros(n, dtype=np.float32),
    }
    for key, default_val in _defaults.items():
        if key not in merged:
            merged[key] = default_val

    return merged


def _load_all() -> tuple[dict, dict]:
    """加载三数据集并合并。"""
    print("=" * 56)
    print("  Loading Unified Dataset (3 sources)")
    print("=" * 56)

    # 1. SMART-DS 原始
    print("\n[1/3] SMART-DS original...")
    from data_loader import SmartDSLoader
    ds1 = SmartDSLoader().load_all()
    print(f"  -> {ds1['n_steps']} steps")

    # 2. SMART-DS 增强
    print("\n[2/3] SMART-DS augmented...")
    aug_dir = "data/smart_ds_augmented"
    if os.path.isdir(aug_dir):
        ds2 = SmartDSLoader(data_dir=aug_dir).load_all()
        print(f"  -> {ds2['n_steps']} steps")
    else:
        print("  WARN: augmented data not found, skipping")
        ds2 = None

    # 3. China 合成
    print("\n[3/3] China synthetic...")
    from data_loader import ChinaDataLoader
    ds3 = ChinaDataLoader().load_all()
    print(f"  -> {ds3['n_steps']} steps")

    # 合并
    datasets = [ds1]
    if ds2 is not None:
        datasets.append(ds2)
    datasets.append(ds3)

    merged = _concat_datasets(*datasets)
    print(f"\n  Unified total: {merged['n_steps']} steps "
          f"({merged['n_steps'] * 15 / 60 / 24:.0f} days)")

    # 8:2 切分
    n = merged["n_steps"]
    split = int(n * 0.8)
    train = {}
    val = {}
    for key in merged:
        if key == "n_steps":
            continue
        arr = merged[key]
        train[key] = arr[:split]
        val[key] = arr[split:]
    train["n_steps"] = split
    val["n_steps"] = n - split
    print(f"  Train: {split} steps ({split * 15 / 60 / 24:.0f} days)")
    print(f"  Val:   {n - split} steps ({(n - split) * 15 / 60 / 24:.0f} days)")

    return train, val


def main():
    train_data, val_data = _load_all()
    from mupc_env import MupcEnv
    from lstm_model import OraclePredictor

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
    TOTAL_STEPS = 200_000  # 数据量大, 每模式多训一些

    for mode in MODES:
        print(f"\n{'='*56}")
        print(f"  Unified: {mode} ({TOTAL_STEPS} steps)")
        print(f"{'='*56}")

        env = MupcEnv(train_data, mode=mode, lstm_predictor=predictor)
        eval_env = MupcEnv(val_data, mode=mode, lstm_predictor=predictor_eval)
        vec_env = DummyVecEnv([EnvMaker(env)])
        vec_eval = DummyVecEnv([EnvMaker(eval_env)])

        eval_cb = EvalCallback(vec_eval, best_model_save_path="checkpoints/",
                               log_path="checkpoints/", eval_freq=50000,
                               deterministic=True)

        policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                         "activation_fn": nn.ReLU}

        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs,
                    tensorboard_log="tensorboard_logs/", seed=42, verbose=1)

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb,
                    tb_log_name=f"unified_{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
        elapsed = time.time() - t0

        model_path = f"checkpoints/unified_{mode}_model"
        model.save(model_path)
        print(f"  {mode} done in {elapsed:.0f}s -> {model_path}.zip")

    print(f"\n{'='*56}")
    print("  Unified training complete!")
    print(f"{'='*56}")
    for m in MODES:
        print(f"  {m}: checkpoints/unified_{m}_model.zip")


if __name__ == "__main__":
    main()
