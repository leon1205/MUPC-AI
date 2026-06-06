"""
SMART-DS 数据加载与状态合成。

职责:
  1. 加载 SMART-DS 光伏 CSV + 负荷 per-unit profile + 气象(辐照/温度)
  2. 合成 TOU 电价、需量、调度指令等部署端实时数据源字段
  3. 归一化 + 训练/验证集按时间 8:2 切分
"""

import os
import math
import json
from pathlib import Path

import numpy as np

# ── 物理常量 ──────────────────────────────────────────────────

PV_ARRAY_KW_REF = 1000.0        # CSV 中光伏阵列参考容量
PV_ARRAY_KW_TARGET = 200.0      # MUPC 实际光伏容量 (缩放因子 = 0.2)
PV_SCALE = PV_ARRAY_KW_TARGET / PV_ARRAY_KW_REF  # 0.2

LOAD_PEAK_KW = 400.0            # 负荷峰值 (kW)
CONTRACT_DEMAND_KW = 300.0      # 默认合同需量 (kW)
GRID_EMISSION_FACTOR = 0.581    # kg CO2/kWh

DATA_DIR = Path(__file__).parent / "data" / "smart_ds"
SOLAR_DIR = DATA_DIR / "solar"
PROFILE_DIR = DATA_DIR / "load_profiles"


# ── TOU 电价配置 ──────────────────────────────────────────────

# (起始小时, 结束小时, tariff_id, 价格)
# tariff_id: 0=谷, 1=平, 2=峰, 3=尖峰
TOU_SCHEDULE = [
    ( 0,  8, 0, 0.4),   # 谷：23:00-08:00 → 0.4 元/kWh
    ( 8, 10, 1, 0.8),   # 平：08:00-10:00
    (10, 12, 2, 1.2),   # 峰：10:00-12:00
    (12, 14, 3, 1.5),   # 尖峰：12:00-14:00 (夏季 6-9 月)
    (14, 17, 1, 0.8),   # 平：14:00-17:00
    (17, 21, 2, 1.2),   # 峰：17:00-21:00
    (21, 23, 1, 0.8),   # 平：21:00-23:00
    (23, 24, 0, 0.4),   # 谷：23:00-24:00
]


def _get_tou(hour: int, month: int) -> tuple[int, float, float]:
    """返回 (tariff_id, current_price, next_period_price)。"""
    current = _tou_for_hour(hour, month)
    next_hour = (hour + 1) % 24
    next_tid, next_price = _tou_for_hour(next_hour, month)
    return current[0], current[1], next_price


def _tou_for_hour(hour: int, month: int) -> tuple[int, float]:
    """返回单一时段的 (tariff_id, price)。"""
    for start, end, tid, price in TOU_SCHEDULE:
        if start <= hour < end:
            if tid == 3 and month not in (6, 7, 8, 9):
                return 2, 1.2  # 非夏季降级为峰
            return tid, price
    return 0, 0.4  # 兜底


# ── 数据加载 ─────────────────────────────────────────────────

class SmartDSLoader:
    """SMART-DS 数据集加载 + 状态合成。"""

    def __init__(self, data_dir: str | None = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = DATA_DIR

    # ── 主加载入口 ──────────────────────────────────────

    def load_all(self) -> dict[str, np.ndarray]:
        """加载全部数据并合成缺失字段。

        Returns keys:
            D1: pv_power, load_power, grid_power, transformer_load,
                battery_power, voltage_phase_a/b/c
            D2: pv_forecast_15min (30维 = 15 pv + 15 load)
            D3: current_electricity_price, next_period_price, price_tariff_id
            D4: current_demand, contract_demand, peak_demand_this_month
            D5: solar_irradiance, temperature
            D6: dispatch_p_set

            + 辅助: timestamps, hour_encoded
            + norm_params: 归一化参数 dict
        """
        print("=" * 56)
        print("  SMART-DS 数据加载与状态合成")
        print("=" * 56)

        # Step 1: 加载真实数据
        print("\n[1/5] 加载光伏与负荷真实数据...")
        solar_data = self._load_solar_files()
        load_data = self._load_load_files()

        # 对齐长度
        n_steps = min(len(solar_data["pv_power"]), len(load_data["load_power"]))
        print(f"  有效步数 (对齐后): {n_steps} ({n_steps * 15 / 60 / 24:.1f} 天)")

        # Step 2: 生成时间戳与时间特征
        print("\n[2/5] 生成时间特征...")
        timestamps = np.arange(n_steps, dtype=np.float64)
        hours = (timestamps * 15 / 60) % 24
        months = np.ones(n_steps) * 7  # 默认 7 月 (尖峰生效)
        hour_encoded = np.sin(hours * 2 * math.pi / 24).astype(np.float32)

        # Step 3: TOU 电价
        print("[3/5] 合成 TOU 电价...")
        prices = self._generate_tou_prices(hours, months)

        # Step 4: 需量合成
        print("[4/5] 合成需量数据...")
        demand = self._generate_demand(load_data["load_power"][:n_steps])

        # Step 5: 调度指令合成
        print("[5/5] 合成调度指令...")
        dispatch = self._generate_dispatch(n_steps)

        # 截断各数组到 n_steps
        result = {
            # D1: 实时数据
            "pv_power": solar_data["pv_power"][:n_steps],
            "load_power": load_data["load_power"][:n_steps],
            "solar_irradiance": solar_data["ghi"][:n_steps],
            "temperature": solar_data["temperature"][:n_steps],
            # D3: 电价
            "current_electricity_price": prices["current"],
            "next_period_price": prices["next"],
            "price_tariff_id": prices["tariff_id"],
            # D4: 需量
            "current_demand": demand["current_demand"],
            "contract_demand": demand["contract_demand"],
            "peak_demand_this_month": demand["peak_demand"],
            # D6: 调度
            "dispatch_p_set": dispatch["dispatch_p_set"],
            # 辅助
            "timestamps": timestamps,
            "hour_encoded": hour_encoded,
            "hours": hours.astype(np.float32),
            "months": months.astype(np.float32),
            "n_steps": n_steps,
        }
        result["norm_params"] = self._compute_norm_params(result)

        # 打印摘要
        self._print_summary(result)
        return result

    def split(self, data: dict, train_ratio: float = 0.8
              ) -> tuple[dict, dict]:
        """按时间顺序切分训练/验证集。

        Returns: (train_dict, val_dict) 各包含与原 data 相同的 keys。
        """
        n = data["n_steps"]
        split_idx = int(n * train_ratio)
        train = {}
        val = {}
        for key in data:
            if key in ("n_steps", "norm_params"):
                continue
            arr = data[key]
            train[key] = arr[:split_idx]
            val[key] = arr[split_idx:]
        train["n_steps"] = split_idx
        val["n_steps"] = n - split_idx
        train["norm_params"] = data["norm_params"]
        val["norm_params"] = data["norm_params"]
        print(f"\n训练集: {split_idx} 步 ({split_idx * 15 / 60 / 24:.1f} 天)")
        print(f"验证集: {n - split_idx} 步 ({(n - split_idx) * 15 / 60 / 24:.1f} 天)")
        return train, val

    # ── 光伏数据 ────────────────────────────────────────

    def _load_solar_files(self) -> dict[str, np.ndarray]:
        """加载 solar/ 目录下所有 CSV, 提取关键列并缩放。"""
        pv_list, ghi_list, temp_list = [], [], []

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                f"请先运行: python data/download_smart_ds.py"
            )

        solar_dir = self.data_dir / "solar"
        csv_files = sorted(solar_dir.glob("*.csv"))
        # 排除 .csv.new 文件
        csv_files = [f for f in csv_files if not f.name.endswith(".new")]

        if not csv_files:
            raise FileNotFoundError(f"未找到光伏 CSV 文件于 {solar_dir}")

        for fp in csv_files:
            print(f"  读取: {fp.name}")
            try:
                # 读取整个 CSV, 提取列 11(kW_generated), 9(GHI), 8(Temperature)
                data = np.genfromtxt(
                    fp, delimiter=",", skip_header=1,
                    usecols=(11, 9, 8),
                    invalid_raise=False,
                    encoding="utf-8-sig",
                )
                # 过滤无效行 (全部 NaN 或包含 NaN)
                if data.ndim == 1:
                    data = data.reshape(-1, 3)
                valid_mask = ~np.isnan(data).any(axis=1)
                data = data[valid_mask]
                if len(data) > 0:
                    pv_list.append(data[:, 0] * PV_SCALE)  # 缩放到 200kW
                    ghi_list.append(data[:, 1])
                    temp_list.append(data[:, 2])
                    print(f"    {len(data)} 行 OK")
            except Exception as e:
                print(f"    WARN: 读取失败 - {e}")

        if not pv_list:
            raise RuntimeError("所有光伏 CSV 加载失败")

        # 拼接所有文件 (按时间顺序)
        return {
            "pv_power": np.concatenate(pv_list).astype(np.float32),
            "ghi": np.concatenate(ghi_list).astype(np.float32),
            "temperature": np.concatenate(temp_list).astype(np.float32),
        }

    # ── 负荷数据 ─────────────────────────────────────────

    def _load_load_files(self) -> dict[str, np.ndarray]:
        """加载 load_profiles/ 目录下所有 per-unit CSV。"""
        load_list = []

        profile_dir = self.data_dir / "load_profiles"
        csv_files = sorted(profile_dir.glob("*.csv"))

        for fp in csv_files:
            print(f"  读取: {fp.name}")
            try:
                values = np.genfromtxt(fp, delimiter=",", invalid_raise=False)
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    # per-unit → 实际功率 (kW)
                    load_kw = values * LOAD_PEAK_KW
                    load_list.append(load_kw)
                    print(f"    {len(values)} 行 OK (range: {load_kw.min():.0f}-{load_kw.max():.0f} kW)")
            except Exception as e:
                print(f"    WARN: 读取失败 - {e}")

        if not load_list:
            raise RuntimeError("所有负荷 CSV 加载失败")

        return {
            "load_power": np.concatenate(load_list).astype(np.float32),
        }

    # ── 电价合成 ─────────────────────────────────────────

    def _generate_tou_prices(self, hours: np.ndarray,
                             months: np.ndarray) -> dict[str, np.ndarray]:
        """TOU 电价生成。"""
        n = len(hours)
        current = np.zeros(n, dtype=np.float32)
        next_p = np.zeros(n, dtype=np.float32)
        tariff = np.zeros(n, dtype=np.int32)

        for i in range(n):
            h = int(hours[i]) % 24
            m = int(months[i])
            tid, price, nprice = _get_tou(h, m)
            current[i] = price
            next_p[i] = nprice
            tariff[i] = tid

        return {"current": current, "next": next_p, "tariff_id": tariff}

    # ── 需量合成 ─────────────────────────────────────────

    def _generate_demand(self, load_power: np.ndarray) -> dict[str, np.ndarray]:
        """15 分钟滑动窗口均值作为需量。"""
        n = len(load_power)
        window = 4  # 4 步 × 15 分钟 = 1 小时

        current_demand = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - window + 1)
            current_demand[i] = float(np.mean(load_power[start:i + 1]))

        # 确保不低于合同需量的 30%
        current_demand = np.maximum(current_demand, CONTRACT_DEMAND_KW * 0.3)

        # 本月峰值需量 (累计最大值)
        peak_demand = np.maximum.accumulate(current_demand)

        return {
            "current_demand": current_demand,
            "contract_demand": np.full(n, CONTRACT_DEMAND_KW, dtype=np.float32),
            "peak_demand": peak_demand.astype(np.float32),
        }

    # ── 调度指令合成 ─────────────────────────────────────

    def _generate_dispatch(self, n_steps: int) -> dict[str, np.ndarray]:
        """默认无调度 (全为 0.0)，VPP 场景运行时动态生成。"""
        return {
            "dispatch_p_set": np.zeros(n_steps, dtype=np.float32),
        }

    # ── 归一化参数 ───────────────────────────────────────

    @staticmethod
    def _compute_norm_params_static(data: dict) -> dict:
        """静态版本，供 ChinaDataLoader 复用。"""
        return SmartDSLoader._do_compute_norm_params(data)

    def _compute_norm_params(self, data: dict) -> dict:
        return self._do_compute_norm_params(data)

    @staticmethod
    def _do_compute_norm_params(data: dict) -> dict:
        """计算归一化参数 (在训练集上统计)。"""
        params = {}
        # D1
        params["battery_soc"] = {"method": "identity"}  # [0,1] 原始
        pv = data["pv_power"]
        params["pv_power"] = {"method": "minmax", "min": 0.0, "max": float(PV_ARRAY_KW_TARGET)}
        load = data["load_power"]
        params["load_power"] = {"method": "minmax", "min": 0.0, "max": float(LOAD_PEAK_KW)}
        params["grid_power"] = {"method": "minmax", "min": -500.0, "max": 500.0}
        params["transformer_load"] = {"method": "identity"}
        params["battery_power"] = {"method": "minmax", "min": -500.0, "max": 500.0}
        for ph in ("a", "b", "c"):
            params[f"voltage_phase_{ph}"] = {"method": "minmax", "min": 0.85, "max": 1.15}
        # D2
        params["pv_forecast"] = {"method": "minmax", "min": 0.0, "max": float(PV_ARRAY_KW_TARGET)}
        params["load_forecast"] = {"method": "minmax", "min": 0.0, "max": float(LOAD_PEAK_KW)}
        # D3
        params["current_price"] = {"method": "minmax", "min": 0.0, "max": 1.5}
        params["next_price"] = {"method": "minmax", "min": 0.0, "max": 1.5}
        params["tariff_id"] = {"method": "minmax", "min": 0.0, "max": 3.0}
        # D4
        params["current_demand"] = {"method": "minmax", "min": 0.0, "max": 500.0}
        params["contract_demand"] = {"method": "minmax", "min": 0.0, "max": 500.0}
        params["peak_demand"] = {"method": "minmax", "min": 0.0, "max": 500.0}
        # D5
        params["solar_irradiance"] = {"method": "minmax", "min": 0.0, "max": 1500.0}
        params["temperature"] = {"method": "minmax", "min": -20.0, "max": 60.0}
        # D6
        params["dispatch_p_set"] = {"method": "minmax", "min": -500.0, "max": 500.0}
        # mode_id
        params["mode_id"] = {"method": "identity"}

        return params

    # ── 打印摘要 ─────────────────────────────────────────

    def _print_summary(self, data: dict) -> None:
        n = data["n_steps"]
        print(f"\n{'=' * 56}")
        print(f"  数据集摘要")
        print(f"{'=' * 56}")
        print(f"  总步数:        {n} ({n * 15 / 60 / 24:.1f} 天)")
        print(f"  光伏均值:      {data['pv_power'].mean():.1f} kW "
              f"(range: {data['pv_power'].min():.0f}-{data['pv_power'].max():.0f})")
        print(f"  负荷均值:      {data['load_power'].mean():.1f} kW "
              f"(range: {data['load_power'].min():.0f}-{data['load_power'].max():.0f})")
        print(f"  辐照均值:      {data['solar_irradiance'].mean():.1f} W/m^2")
        print(f"  温度均值:      {data['temperature'].mean():.1f} C")
        print(f"  电价均值:      {data['current_electricity_price'].mean():.2f} yuan/kWh")


# ═══════════════════════════════════════════════════════════════
# 中国数据集加载器
# ═══════════════════════════════════════════════════════════════

CHINA_DATA_DIR = Path(__file__).parent / "data" / "china_data"


class ChinaDataLoader:
    """中国区域数据集加载器 — 与 SmartDSLoader 相同接口。

    数据格式:
      - solar/*.csv: SMART-DS 兼容的光伏 CSV
      - load/*.csv: per-unit 负荷 (单列浮点值)
      - pricing/*.csv: current_price,next_price,tariff_id
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else CHINA_DATA_DIR

    def load_all(self) -> dict[str, np.ndarray]:
        print("=" * 56)
        print("  China Dataset Loading")
        print("=" * 56)

        print("\n[1/4] Loading solar + weather data...")
        solar_data = self._load_solar_files()

        print("\n[2/4] Loading load profiles...")
        load_data = self._load_load_files()

        n_steps = min(len(solar_data["pv_power"]), len(load_data["load_power"]))
        print(f"  Aligned steps: {n_steps} ({n_steps * 15 / 60 / 24:.1f} days)")

        print("\n[3/4] Loading TOU pricing...")
        prices = self._load_pricing_files(n_steps)

        print("\n[4/4] Generating demand + dispatch...")
        demand = self._generate_demand(load_data["load_power"][:n_steps])
        dispatch = self._generate_dispatch(n_steps)

        hours = (np.arange(n_steps, dtype=np.float32) * 15 / 60) % 24

        result = {
            "pv_power": solar_data["pv_power"][:n_steps],
            "load_power": load_data["load_power"][:n_steps],
            "solar_irradiance": solar_data["ghi"][:n_steps],
            "temperature": solar_data["temperature"][:n_steps],
            "current_electricity_price": prices["current"][:n_steps],
            "next_period_price": prices["next"][:n_steps],
            "price_tariff_id": prices["tariff_id"][:n_steps],
            "current_demand": demand["current_demand"],
            "contract_demand": demand["contract_demand"],
            "peak_demand_this_month": demand["peak_demand"],
            "dispatch_p_set": dispatch["dispatch_p_set"],
            "timestamps": np.arange(n_steps, dtype=np.float64),
            "hour_encoded": np.sin(hours * 2 * math.pi / 24).astype(np.float32),
            "hours": hours,
            "months": np.ones(n_steps, dtype=np.float32) * 7,
            "n_steps": n_steps,
        }
        result["norm_params"] = SmartDSLoader._compute_norm_params_static(result)
        return result

    def split(self, data: dict, train_ratio: float = 0.8) -> tuple[dict, dict]:
        n = data["n_steps"]
        split_idx = int(n * train_ratio)
        train, val = {}, {}
        for key in data:
            if key in ("n_steps", "norm_params"):
                continue
            arr = data[key]
            train[key] = arr[:split_idx]
            val[key] = arr[split_idx:]
        train["n_steps"] = split_idx
        val["n_steps"] = n - split_idx
        train["norm_params"] = data["norm_params"]
        val["norm_params"] = data["norm_params"]
        print(f"\nTrain: {split_idx} steps ({split_idx * 15 / 60 / 24:.1f} days)")
        print(f"Val: {n - split_idx} steps ({(n - split_idx) * 15 / 60 / 24:.1f} days)")
        return train, val

    def _load_solar_files(self) -> dict[str, np.ndarray]:
        d = self.data_dir / "solar"
        csv_files = sorted(d.glob("*.csv"))
        pv_list, ghi_list, temp_list = [], [], []
        for fp in csv_files:
            try:
                data = np.genfromtxt(fp, delimiter=",", skip_header=1,
                                     usecols=(11, 9, 8), invalid_raise=False,
                                     encoding="utf-8-sig")
                if data.ndim == 1:
                    data = data.reshape(-1, 3)
                valid = data[~np.isnan(data).any(axis=1)]
                if len(valid) > 0:
                    pv_list.append(valid[:, 0])
                    ghi_list.append(valid[:, 1])
                    temp_list.append(valid[:, 2])
                    print(f"  {fp.name}: {len(valid)} rows")
            except Exception as e:
                print(f"  WARN {fp.name}: {e}")
        return {
            "pv_power": np.concatenate(pv_list).astype(np.float32),
            "ghi": np.concatenate(ghi_list).astype(np.float32),
            "temperature": np.concatenate(temp_list).astype(np.float32),
        }

    def _load_load_files(self) -> dict[str, np.ndarray]:
        d = self.data_dir / "load"
        csv_files = sorted(d.glob("*.csv"))
        load_list = []
        for fp in csv_files:
            try:
                values = np.genfromtxt(fp, delimiter=",", invalid_raise=False)
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    load_kw = values * LOAD_PEAK_KW
                    load_list.append(load_kw)
                    print(f"  {fp.name}: {len(values)} rows ({load_kw.min():.0f}-{load_kw.max():.0f} kW)")
            except Exception as e:
                print(f"  WARN {fp.name}: {e}")
        return {"load_power": np.concatenate(load_list).astype(np.float32)}

    def _load_pricing_files(self, n_steps: int) -> dict[str, np.ndarray]:
        d = self.data_dir / "pricing"
        csv_files = sorted(d.glob("*.csv"))
        current_list, next_list, tariff_list = [], [], []
        for fp in csv_files:
            try:
                data = np.genfromtxt(fp, delimiter=",", skip_header=1,
                                     invalid_raise=False, encoding="utf-8-sig")
                if data.ndim == 1:
                    data = data.reshape(-1, 3)
                data = data[~np.isnan(data).any(axis=1)]
                current_list.append(data[:, 0])
                next_list.append(data[:, 1])
                tariff_list.append(data[:, 2])
                print(f"  {fp.name}: {len(data)} rows")
            except Exception as e:
                print(f"  WARN {fp.name}: {e}")
        # 取最短对齐
        min_len = min(len(a) for a in current_list)
        return {
            "current": np.concatenate([a[:min_len] for a in current_list]).astype(np.float32),
            "next": np.concatenate([a[:min_len] for a in next_list]).astype(np.float32),
            "tariff_id": np.concatenate([a[:min_len] for a in tariff_list]).astype(np.int32),
        }

    def _generate_demand(self, load_power: np.ndarray) -> dict:
        n = len(load_power)
        window = 4
        cd = np.array([np.mean(load_power[max(0,i-window+1):i+1]) for i in range(n)], dtype=np.float32)
        cd = np.maximum(cd, CONTRACT_DEMAND_KW * 0.3)
        return {
            "current_demand": cd,
            "contract_demand": np.full(n, CONTRACT_DEMAND_KW, dtype=np.float32),
            "peak_demand": np.maximum.accumulate(cd).astype(np.float32),
        }

    def _generate_dispatch(self, n_steps: int) -> dict:
        return {"dispatch_p_set": np.zeros(n_steps, dtype=np.float32)}


# ── 自测入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    loader = SmartDSLoader()
    data = loader.load_all()
    train, val = loader.split(data)

    # 显示前 10 步样本
    print(f"\n{'=' * 56}")
    print("  前 10 步数据样本")
    print(f"{'=' * 56}")
    keys_show = ["pv_power", "load_power", "solar_irradiance", "temperature",
                 "current_electricity_price", "current_demand", "price_tariff_id"]
    header = f"{'步':>4s}"
    for k in keys_show:
        header += f" {k:>14s}"
    print(header)
    print("-" * len(header))
    for i in range(min(10, train["n_steps"])):
        row = f"{i:4d}"
        for k in keys_show:
            row += f" {train[k][i]:14.3f}"
        print(row)
    print(f"\n数据加载自测通过。")
