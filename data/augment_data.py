"""
数据增强脚本 — 扩大训练数据集 3~5 倍。

策略:
  1. 时间平移: 将数据前移/后移 ±1~4 小时 (模拟不同经度/夏令时)
  2. 光伏缩放: PV ±20% 线性缩放 (模拟不同容量/衰减)
  3. 负荷缩放: 负荷 ±15% 缩放 + 噪声 (模拟不同台区)
  4. 噪声注入: 辐照/温度加入高斯噪声 (模拟测量误差)
  5. 模式混合: 将不同城市/建筑类型的数据混合

用法: python data/augment_data.py
输出: data/smart_ds_augmented/ (增强后的 SMART-DS 兼容格式)
"""

import math
import os
from pathlib import Path
import numpy as np

SOURCE_DIR = Path(__file__).parent / "smart_ds"
OUTPUT_DIR = Path(__file__).parent / "smart_ds_augmented"
AUGMENT_FACTOR = 4  # 扩增倍数 (原始 1x + 增强 3x = 总共 4x)


def load_source_data() -> dict:
    """加载 SMART-DS 原始数据。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_loader import SmartDSLoader
    loader = SmartDSLoader(data_dir=str(SOURCE_DIR))
    return loader.load_all()


def augment_time_shift(data: dict, shift_hours: float, seed: int) -> dict:
    """时间平移: 将序列前后移动 shift_hours。

    模拟不同时区/经度的光伏-负荷相位差。
    """
    shift_steps = int(shift_hours * 4)  # 15分钟步长
    n = data["n_steps"]
    shifted = dict(data)

    for key in ["pv_power", "load_power", "solar_irradiance", "temperature"]:
        if key in data and isinstance(data[key], np.ndarray):
            arr = data[key]
            if shift_steps > 0:
                shifted[key] = np.concatenate([arr[shift_steps:], arr[:shift_steps]])
            elif shift_steps < 0:
                shifted[key] = np.concatenate([arr[shift_steps:], arr[:shift_steps]])
            else:
                shifted[key] = arr.copy()

    # 重新生成电价 (时间变了, 电价时段也变)
    shifted["current_electricity_price"] = np.roll(data["current_electricity_price"], shift_steps)
    shifted["next_period_price"] = np.roll(data["next_period_price"], shift_steps)
    shifted["price_tariff_id"] = np.roll(data["price_tariff_id"], shift_steps)
    shifted["hour_encoded"] = np.sin(
        ((np.arange(n) + shift_steps) * 15 / 60 % 24) * 2 * math.pi / 24
    ).astype(np.float32)

    shifted["augmentation"] = f"time_shift_{shift_hours}h"
    shifted["seed"] = seed
    return shifted


def augment_pv_scale(data: dict, scale_factor: float, seed: int) -> dict:
    """光伏缩放: 将 PV 出力乘以 scale_factor。

    模拟: 不同光伏容量 (0.8x~1.2x), 组件衰减, 不同朝向。
    """
    np.random.seed(seed)
    aug = dict(data)
    for key in ["pv_power"]:
        if key in data:
            aug[key] = data[key].copy() * scale_factor
    # 辐照不变 (物理上不可缩放)
    aug["augmentation"] = f"pv_scale_{scale_factor:.2f}"
    aug["seed"] = seed
    return aug


def augment_load_scale(data: dict, scale_factor: float, seed: int) -> dict:
    """负荷缩放: 模拟不同台区容量。"""
    np.random.seed(seed)
    aug = dict(data)
    for key in ["load_power"]:
        if key in data:
            noise = np.random.normal(0, 0.02, len(data[key]))
            aug[key] = data[key].copy() * scale_factor * (1.0 + noise)
    aug["augmentation"] = f"load_scale_{scale_factor:.2f}"
    aug["seed"] = seed
    return aug


def augment_noise(data: dict, noise_level: float, seed: int) -> dict:
    """测量噪声注入: 模拟传感器误差。"""
    np.random.seed(seed)
    aug = dict(data)
    for key in ["solar_irradiance", "temperature", "pv_power", "load_power"]:
        if key in data:
            arr = data[key]
            std = noise_level * (np.abs(arr).mean() + 1e-6)
            aug[key] = arr + np.random.normal(0, std, len(arr))
    aug["augmentation"] = f"noise_{noise_level:.2f}"
    aug["seed"] = seed
    return aug


def save_augmented(data: dict, output_dir: Path) -> None:
    """保存增强数据为 SMART-DS 兼容格式。"""
    tag = data.get("augmentation", "original")
    solar_dir = output_dir / "solar"
    profile_dir = output_dir / "load_profiles"
    solar_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    n = data["n_steps"]
    # 保存光伏
    solar_fp = solar_dir / f"aug_{tag}.csv"
    from datetime import datetime, timedelta
    with open(solar_fp, "w") as f:
        f.write("Year,Month,Day,Hour,Minute,GHI,DNI,DHI,"
                "Wind_Speed,Temperature,PoA_Irradiance,kW_Generated\n")
        for i in range(n):
            dt = datetime(2022, 1, 1) + timedelta(minutes=i * 15)
            ghi = data.get("solar_irradiance", np.zeros(n))[i]
            temp = data.get("temperature", np.zeros(n))[i]
            pv = data["pv_power"][i]
            f.write(f"{dt.year},{dt.month},{dt.day},{dt.hour},{dt.minute},"
                    f"{ghi:.1f},0.0,0.0,3.0,{temp:.1f},{ghi:.1f},{pv:.1f}\n")

    # 保存负荷
    load_fp = profile_dir / f"aug_{tag}_pu.csv"
    with open(load_fp, "w") as f:
        for v in data.get("load_power", np.zeros(n)) / 400.0:
            f.write(f"{v:.8f}\n")


def main():
    print("=" * 56)
    print("  Data Augmentation")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 56)

    print("\n[1/3] Loading source data...")
    data = load_source_data()
    print(f"  Original: {data['n_steps']} steps")

    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # 保存原始副本
    print("\n[2/3] Saving original + augmented copies...")
    save_augmented(data, OUTPUT_DIR)
    print(f"  [1/{AUGMENT_FACTOR}] Original saved")

    augmentations = [
        ("time_shift_-2h", lambda d, s: augment_time_shift(d, -2.0, s)),
        ("time_shift_+2h", lambda d, s: augment_time_shift(d, 2.0, s)),
        ("pv_scale_0.8",   lambda d, s: augment_pv_scale(d, 0.8, s)),
        ("pv_scale_1.2",   lambda d, s: augment_pv_scale(d, 1.2, s)),
        ("load_scale_0.85", lambda d, s: augment_load_scale(d, 0.85, s)),
        ("load_scale_1.15", lambda d, s: augment_load_scale(d, 1.15, s)),
        ("noise_0.03",     lambda d, s: augment_noise(d, 0.03, s)),
    ]

    count = 1
    seeds = iter([100, 200, 300, 400, 500, 600, 700])
    for name, aug_fn in augmentations:
        if count >= AUGMENT_FACTOR:
            break
        seed = next(seeds)
        aug = aug_fn(data, seed)
        save_augmented(aug, OUTPUT_DIR)
        count += 1
        print(f"  [{count}/{AUGMENT_FACTOR}] {name}")

    print(f"\n[3/3] Summary")
    print(f"  Total files: {count}x original")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Use with: data_loader.SmartDSLoader(data_dir='{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
