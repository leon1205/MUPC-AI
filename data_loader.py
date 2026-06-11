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

# ── 懒加载配置 ──────────────────────────────────────────────────


def _cfg():
    """懒加载配置（支持无 --config 时的硬编码回退）。"""
    from config.config_manager import get_config
    return get_config()


# ── 物理常量 ──────────────────────────────────────────────────

_c = _cfg()
PV_ARRAY_KW_REF = 1000.0        # CSV 中光伏阵列参考容量（通常不需修改）
PV_ARRAY_KW_TARGET = _c.physical.pv_array_kw  # MUPC 实际光伏容量 (缩放因子)
PV_SCALE = PV_ARRAY_KW_TARGET / PV_ARRAY_KW_REF  # 动态计算

LOAD_PEAK_KW = _c.physical.load_peak_kw
CONTRACT_DEMAND_KW = _c.contract.contract_demand_kw
GRID_EMISSION_FACTOR = _c.contract.grid_emission_factor

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
# 中国合成数据生成器
# ═══════════════════════════════════════════════════════════════

class ChinaSyntheticDataGenerator:
    """基于中国典型气象和负荷特征生成合成数据。

    使用经纬度驱动的天照辐射模型 + 温度模型 + 工商业负荷模型
    生成与 NASA POWER 统计分布相似的全年数据（15分钟步长）。

    适用场景：内网环境无法访问外部 API 时，用合成数据验证完整训练流程。
    """

    # 中国典型城市参考经纬度 (lat, lon, 海拔m, 气候区)
    CITY_PROFILES = {
        "shanghai":   (31.23, 121.47, 4,   "subtropical"),
        "beijing":     (39.91, 116.39, 44,  "temperate"),
        "chengdu":     (30.66, 104.07, 506, "subtropical"),
        "kunming":     (25.04, 102.71, 1888, "plateau"),
        "xian":        (34.27, 108.95, 397, "temperate"),
        "guangzhou":   (23.13, 113.26, 7,    "tropical"),
        "harbin":      (45.80, 126.53, 142,  "cold"),
        "urumqi":      (43.83, 87.62,   918,  "dry"),
    }

    def __init__(self, lat: float, lon: float,
                 pv_capacity_kw: float = 200.0,
                 peak_load_kw: float = 400.0,
                 panel_efficiency: float = 0.18,
                 panel_area_m2_per_kw: float = 8.0,
                 derating: float = 0.8,
                 data_dir: str | None = None):
        """
        Args:
            lat, lon: 站点经纬度
            pv_capacity_kw: 光伏装机容量 (默认 200kW)
            peak_load_kw: 负荷峰值 (默认 400kW)
            panel_efficiency: 光伏板效率 (默认 18%)
            derating: 综合折减系数 (灰尘/温度/线损，默认 0.8)
        """
        self.lat = lat
        self.lon = lon
        self.pv_capacity_kw = pv_capacity_kw
        self.peak_load_kw = peak_load_kw
        self.panel_efficiency = panel_efficiency
        self.panel_area_m2_per_kw = panel_area_m2_per_kw
        self.derating = derating
        self.data_dir = Path(data_dir) if data_dir else CHINA_DATA_DIR

    # 省份和区域映射
    PROVINCE_MAP = {
        "shanghai": "Shanghai", "beijing": "Beijing", "tianjin": "Tianjin",
        "chongqing": "Chongqing", "chengdu": "Sichuan", "kunming": "Yunnan",
        "xian": "Shaanxi", "guangzhou": "Guangdong", "shenzhen": "Guangdong",
        "nanjing": "Jiangsu", "hangzhou": "Zhejiang", "suzhou": "Jiangsu",
        "harbin": "Heilongjiang", "changchun": "Jilin", "shenyang": "Liaoning",
        "urumqi": "Xinjiang", "lhasa": "Tibet", "zhengzhou": "Henan",
        "jinan": "Shandong", "qingdao": "Shandong", "changsha": "Hunan",
        "wuhan": "Hubei", "nanchang": "Jiangxi", "nanning": "Guangxi",
        "guiyang": "Guizhou", "fuzhou": "Fujian", "xiamen": "Fujian",
        "haikou": "Hainan", "yinchuan": "Ningxia", "xining": "Qinghai",
        "lanzhou": "Gansu", "hohhot": "Inner Mongolia", "baotou": "Inner Mongolia",
        "taiyuan": "Shanxi", "shijiazhuang": "Hebei", "tangshan": "Hebei",
        "dalian": "Liaoning", "ningbo": "Zhejiang", "wenzhou": "Zhejiang",
        "wuxi": "Jiangsu", "foshan": "Guangdong", "dongguan": "Guangdong",
        "zhuhai": "Guangdong", "macau": "Guangdong", "hongkong": "Guangdong",
    }

    def _lat_lon_to_province(self, lat: float, lon: float) -> str:
        """根据经纬度返回省份名（英文，首字母大写）。"""
        # 精确匹配主要城市
        for city, province in [
            ("shanghai", "Shanghai"), ("beijing", "Beijing"), ("chengdu", "Sichuan"),
            ("kunming", "Yunnan"), ("xian", "Shaanxi"), ("guangzhou", "Guangdong"),
            ("harbin", "Heilongjiang"), ("urumqi", "Xinjiang"), ("lhasa", "Tibet"),
            ("zhengzhou", "Henan"), ("nanjing", "Jiangsu"), ("hangzhou", "Zhejiang"),
        ]:
            if city in self.CITY_PROFILES:
                c_lat, c_lon = self.CITY_PROFILES[city][:2]
                if abs(lat - c_lat) < 0.5 and abs(lon - c_lon) < 0.5:
                    return province
        # 按经纬度区间粗匹配
        if 31.0 <= lat <= 31.5 and 120.5 <= lon <= 122.0:
            return "Shanghai"
        elif 39.5 <= lat <= 40.0 and 115.5 <= lon <= 117.5:
            return "Beijing"
        elif 29.5 <= lat <= 31.5 and 103.0 <= lon <= 108.0:
            return "Sichuan"
        elif 22.5 <= lat <= 25.0 and 110.0 <= lon <= 115.0:
            return "Guangdong"
        elif 43.0 <= lat <= 46.0 and 125.0 <= lon <= 130.0:
            return "Heilongjiang"
        elif 43.0 <= lat <= 45.0 and 85.0 <= lon <= 92.0:
            return "Xinjiang"
        elif 34.0 <= lat <= 35.0 and 108.0 <= lon <= 110.0:
            return "Shaanxi"
        elif 28.0 <= lat <= 30.0 and 112.0 <= lon <= 115.0:
            return "Henan"
        elif 30.0 <= lat <= 32.0 and 118.0 <= lon <= 122.0:
            return "Jiangsu"
        elif 25.0 <= lat <= 30.0 and 98.0 <= lon <= 106.0:
            return "Yunnan"
        else:
            return "Unknown"

    def _province_to_region(self, province: str) -> str:
        """省份名 → 城市名（取首字母大写形式）。"""
        return province  # 直接返回省份名作为城市名

    def _solar_declination(self, day_of_year: int) -> float:
        """计算太阳赤纬 (radians)。"""
        return 23.45 * np.pi / 180 * np.sin(2 * np.pi * (284 + day_of_year) / 365)

    def _hour_angle(self, hour: float, lon: float) -> float:
        """计算时角 (radians)。lon 以度为单位。"""
        # 时区估算: lon / 15 (小时)
        timezone_offset = lon / 15.0
        solar_time = hour + timezone_offset - 12.0
        return solar_time * 15.0 * np.pi / 180

    def _ghi_model(self, day_of_year: int, hour: float) -> float:
        """计算瞬时 GHI (W/m²)。

        基于太阳高度角 + 大气透射率 + 云量随机扰动。
        """
        lat_rad = self.lat * np.pi / 180
        dec = self._solar_declination(day_of_year)
        ha = self._hour_angle(hour, self.lon)

        # 太阳高度角 sin(elev) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(ha)
        sin_elev = (np.sin(lat_rad) * np.sin(dec) +
                    np.cos(lat_rad) * np.cos(dec) * np.cos(ha))
        elev = np.arcsin(np.clip(sin_elev, 0.0, 1.0))

        if elev <= 0:
            return 0.0

        # 大气透射率 (Clear Sky Index)
        # 海拔越高，透射率越高
        air_mass = 1.0 / np.sin(elev) if sin_elev > 0.1 else 20.0
        tau = np.exp(-0.000118 * air_mass) * 0.7  # 简化晴空模型

        # 日照常数 1361 W/m²
        ghi_clear = 1361.0 * sin_elev * tau

        # 季节性云量扰动 (随机噪声)
        # 夏季多雨云量多，冬季晴朗
        seasonal_cloud = 0.15 * np.cos(2 * np.pi * day_of_year / 365)
        noise = np.random.normal(0, 0.05)
        cloud_factor = np.clip(1.0 - seasonal_cloud + noise, 0.1, 1.0)

        return max(0.0, ghi_clear * cloud_factor)

    def _temperature_model(self, day_of_year: int, hour: float,
                          base_temp: float = 15.0) -> float:
        """计算瞬时温度 (°C)。

        基础温度 + 季节变化 + 日变化 + 随机扰动。
        """
        # 季节变化: 年振幅 ±15°C
        seasonal = 15.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # 日变化: 14:00 最热 (振幅 ±5°C)
        hourly = 5.0 * np.sin(2 * np.pi * (hour - 14) / 24)

        # 纬度修正: 高纬度年振幅更大
        lat_factor = abs(self.lat) / 90.0  # 0~1
        seasonal *= (0.5 + lat_factor * 0.5)

        noise = np.random.normal(0, 1.5)
        return base_temp + seasonal + hourly + noise

    def generate_year(self, year: int = 2023,
                      random_seed: int = 42) -> dict[str, np.ndarray]:
        """生成全年 15 分钟步长的合成数据。

        Args:
            year: 数据年份
            random_seed: 随机种子 (保证可复现)

        Returns:
            dict keys: pv_power, ghi, temperature, load_power,
                      timestamps, hours, months, norm_params
        """
        import datetime

        np.random.seed(random_seed)

        # 全年步长: 96 steps/day × 365 days = 35040
        n_days = 365
        if year % 4 == 0:
            n_days = 366  # 闰年
        steps_per_day = 96
        n_steps = n_days * steps_per_day

        print(f"\n{'=' * 56}")
        print(f"  Generating Synthetic China Data")
        print(f"  Location: lat={self.lat}, lon={self.lon}")
        print(f"  Year: {year} ({n_days} days, {n_steps} steps)")
        print(f"{'=' * 56}")

        # 预计算每日基础数据 (加速)
        daily_ghi_max = []
        for d in range(1, n_days + 1):
            # 取正午 GHI 代表日峰值
            ghi_noon = self._ghi_model(d, 12.0)
            daily_ghi_max.append(ghi_noon)

        # 生成所有步长数据
        pv_list, ghi_list, temp_list, load_list = [], [], [], []
        hours_arr = np.zeros(n_steps, dtype=np.float32)
        months_arr = np.zeros(n_steps, dtype=np.float32)

        for day_idx in range(n_days):
            day_of_year = day_idx + 1
            month = int(datetime.date(year, 1, 1).replace(
                month=((day_idx * 12) // n_days) + 1).month)
            base_temp = 15.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            # 基础温度随纬度修正
            base_temp += (self.lat - 30) * 0.3  # 南方更热

            for step_in_day in range(steps_per_day):
                hour = step_in_day * 15.0 / 60.0
                hours_arr[day_idx * steps_per_day + step_in_day] = hour
                months_arr[day_idx * steps_per_day + step_in_day] = month

                # GHI
                ghi = self._ghi_model(day_of_year, hour)
                ghi_list.append(ghi)

                # Temperature
                temp = self._temperature_model(day_of_year, hour, base_temp)
                temp_list.append(temp)

                # PV power: GHI → kW
                panel_area = self.pv_capacity_kw * self.panel_area_m2_per_kw
                pv = ghi * panel_area * self.panel_efficiency * self.derating / 1000.0
                pv = np.clip(pv, 0.0, self.pv_capacity_kw)
                pv_list.append(pv)

                # Load power (温度驱动 + 工作日)
                dow = datetime.date(year, 1, 1).weekday()
                day_offset = day_idx
                actual_dow = (dow + day_offset) % 7

                # 基础负荷曲线 (夜间低，白天双峰)
                hour_factor = 0.3 + 0.25 * np.sin((hour - 6) * np.pi / 12)
                # 温度驱动空调负荷
                temp_factor = np.clip((temp - 25.0) / 10.0, 0.0, 1.0) ** 0.8
                # 工作日因子
                weekday_factor = 1.0 if actual_dow < 5 else (0.7 if actual_dow == 5 else 0.5)

                load = self.peak_load_kw * hour_factor * (
                    1.0 + temp_factor * 0.6) * weekday_factor
                # 添加随机噪声 ±5%
                load *= (1.0 + np.random.normal(0, 0.03))
                load_list.append(max(0.0, load))

        # 转为 numpy arrays
        ghi_arr = np.array(ghi_list, dtype=np.float32)
        temp_arr = np.array(temp_list, dtype=np.float32)
        pv_arr = np.array(pv_list, dtype=np.float32)
        load_arr = np.array(load_list, dtype=np.float32)

        print(f"  GHI range: {ghi_arr.min():.0f} - {ghi_arr.max():.0f} W/m2")
        print(f"  Temp range: {temp_arr.min():.1f} - {temp_arr.max():.1f} degC")
        print(f"  PV range: {pv_arr.min():.1f} - {pv_arr.max():.1f} kW")
        print(f"  Load range: {load_arr.min():.1f} - {load_arr.max():.1f} kW")

        return {
            "pv_power": pv_arr,
            "ghi": ghi_arr,
            "temperature": temp_arr,
            "load_power": load_arr,
            "hours": hours_arr,
            "months": months_arr,
            "n_steps": n_steps,
        }

    def save_to_csv(self, year: int = 2023, random_seed: int = 42) -> None:
        """生成并保存 CSV 文件到 data/china_data/。

        生成:
          data/china_data/solar/{region}_{province}_pv.csv
          data/china_data/load/{region}_{province}_pu.csv
        """
        data = self.generate_year(year, random_seed)

        solar_dir = self.data_dir / "solar"
        load_dir = self.data_dir / "load"
        solar_dir.mkdir(parents=True, exist_ok=True)
        load_dir.mkdir(parents=True, exist_ok=True)

        # 使用区域名: 根据 lat/lon 判断省份
        province = self._lat_lon_to_province(self.lat, self.lon)
        region = self._province_to_region(province)
        prefix = f"{region}_{province}"

        # 保存 solar CSV: pv_power_kW, ghi_Wm2, temperature_C
        solar_fp = solar_dir / f"{prefix}_pv.csv"
        np.savetxt(solar_fp,
                   np.column_stack([data["pv_power"],
                                   data["ghi"],
                                   data["temperature"]]),
                   delimiter=",", fmt="%.4f",
                   header="pv_power_kW,ghi_Wm2,temperature_C",
                   comments="")
        print(f"\n  Saved: {solar_fp}")

        # 保存 load CSV: per-unit
        load_fp = load_dir / f"{prefix}_pu.csv"
        load_pu = data["load_power"] / self.peak_load_kw
        np.savetxt(load_fp, load_pu, delimiter=",", fmt="%.6f",
                   header="load_per_unit",
                   comments="")
        print(f"  Saved: {load_fp}")


# ═══════════════════════════════════════════════════════════════
# 统一数据加载器（单入口）
# ═══════════════════════════════════════════════════════════════

class UnifiedDataLoader:
    """统一数据加载入口，自动检测可用数据源。

    使用优先级:
      1. data/china_data/        (中国合成数据，已有 CSV)
      2. data/china_data/         (中国合成数据，未生成 → 自动生成)
      3. data/smart_ds/           (美国 SMART-DS)

    训练/验证集按时间 8:2 切分 (与 SmartDSLoader 一致)。

    Usage:
        loader = UnifiedDataLoader()
        data = loader.load_all()
        train, val = loader.split(data)
        print(f"Loaded: {data['n_steps']} steps from {loader.source_name}")
    """

    def __init__(self, lat: float = 31.23, lon: float = 121.47,
                 pv_capacity_kw: float = 200.0,
                 peak_load_kw: float = 400.0,
                 auto_generate: bool = True,
                 merge_data: bool = False,
                 data_dir: str | None = None):
        """
        Args:
            lat, lon: 当使用中国数据时的坐标 (默认上海)
            pv_capacity_kw: 光伏装机容量
            peak_load_kw: 负荷峰值
            auto_generate: 数据不存在时自动生成合成数据
            merge_data: 是否合并 SMART-DS + 中国合成数据
            data_dir: 数据根目录
        """
        self.lat = lat
        self.lon = lon
        self.pv_capacity_kw = pv_capacity_kw
        self.peak_load_kw = peak_load_kw
        self.auto_generate = auto_generate
        self.merge_data = merge_data
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.source_name = "unknown"

    def load_all(self) -> dict[str, np.ndarray]:
        """自动检测并加载可用数据源。

        Args:
            merge_data=True 时，优先尝试合并 SMART-DS + 中国合成数据

        Returns:
            与 SmartDSLoader 相同的 data dict，含 norm_params
        """
        # 优先使用合并模式
        if self.merge_data:
            smart_ds_dir = self.data_dir / "smart_ds"
            if smart_ds_dir.exists() and (smart_ds_dir / "solar").exists():
                return self._load_merged()
            # SMART-DS 不可用，降级到中国数据
            print("  [WARN] SMART-DS not found, falling back to China data only")

        # 检查中国数据
        china_solar = self.data_dir / "china_data" / "solar"
        china_load = self.data_dir / "china_data" / "load"
        if china_solar.exists() and any(china_solar.glob("*.csv")):
            return self._load_china()
        if self.auto_generate and (not china_solar.exists() or not any(china_solar.glob("*.csv"))):
            return self._generate_china()

        # 检查 SMART-DS 数据
        smart_ds_dir = self.data_dir / "smart_ds"
        if smart_ds_dir.exists() and (smart_ds_dir / "solar").exists():
            return self._load_smart_ds()

        # 无数据 → 报错
        raise FileNotFoundError(
            f"No data found.\n"
            f"  China data: {china_solar} (empty={not any(china_solar.glob('*.csv')) if china_solar.exists() else 'not exist'})\n"
            f"  SMART-DS: {smart_ds_dir / 'solar'} (exists={smart_ds_dir.exists()})\n"
            f"\nHint: Set auto_generate=True or run:"
            f"\n  python data_loader.py --download --lat {self.lat} --lon {self.lon}"
        )

    def _load_china(self) -> dict[str, np.ndarray]:
        """通过 ChinaDataLoader 加载中国数据（按 region 过滤）。"""
        # 根据 lat/lon 确定区域
        gen = ChinaSyntheticDataGenerator(lat=self.lat, lon=self.lon,
                                          data_dir=self.data_dir / "china_data")
        province = gen._lat_lon_to_province(self.lat, self.lon)
        region = province  # 城市名就是省份名
        loader = ChinaDataLoader(data_dir=self.data_dir / "china_data",
                                 region=region)
        result = loader.load_all()
        self.source_name = "china_data"
        return result

    def _generate_china(self) -> dict[str, np.ndarray]:
        """生成并加载中国合成数据。"""
        print(f"\n{'=' * 56}")
        print(f"  Auto-generating synthetic China data")
        print(f"  Location: lat={self.lat}, lon={self.lon}")
        print(f"{'=' * 56}")

        gen = ChinaSyntheticDataGenerator(
            lat=self.lat, lon=self.lon,
            pv_capacity_kw=self.pv_capacity_kw,
            peak_load_kw=self.peak_load_kw,
            data_dir=self.data_dir,
        )
        gen.save_to_csv(year=2023, random_seed=42)

        return self._load_china()

    def _load_smart_ds(self) -> dict[str, np.ndarray]:
        """通过 SmartDSLoader 加载 SMART-DS 数据。"""
        loader = SmartDSLoader(data_dir=self.data_dir / "smart_ds")
        result = loader.load_all()
        self.source_name = "smart_ds"
        return result

    def _load_merged(self) -> dict[str, np.ndarray]:
        """合并 SMART-DS + 中国合成数据（按时间轴拼接）。

        流程：
          1. 加载 SMART-DS 全量数据（作为主体）
          2. 生成并加载中国合成数据（作为补充）
          3. 拼接公共字段（沿时间轴 axis=0）
          4. 重新计算 hours / timestamps / norm_params
        """
        print(f"\n{'=' * 56}")
        print(f"  Merging SMART-DS + China Synthetic Data")
        print(f"{'=' * 56}")

        # ── Step 1: SMART-DS ────────────────────────────────
        print("\n[1/2] Loading SMART-DS data...")
        smart_ds = self._load_smart_ds()

        # ── Step 2: China Synthetic ────────────────────────
        print("\n[2/2] Generating China synthetic data...")
        china = self._generate_china()

        # ── Step 3: 拼接公共字段 ─────────────────────────────
        # 需要拼接的字段列表（排除 meta 字段）
        MERGE_KEYS = [
            "pv_power", "load_power", "solar_irradiance", "temperature",
            "current_electricity_price", "next_period_price", "price_tariff_id",
            "current_demand", "contract_demand", "peak_demand_this_month",
            "dispatch_p_set",
        ]
        merged = {}
        for key in MERGE_KEYS:
            a = smart_ds[key]
            b = china[key]
            merged[key] = np.concatenate([a, b], axis=0).astype(np.float32)
            print(f"  {key}: {len(a)} + {len(b)} = {len(merged[key])}")

        # ── Step 4: 重新计算派生字段 ─────────────────────────
        n_steps = len(merged["pv_power"])
        timestamps = np.arange(n_steps, dtype=np.float64)
        hours = (timestamps * 15 / 60) % 24
        months = np.ones(n_steps, dtype=np.float32) * 7
        hour_encoded = np.sin(hours * 2 * math.pi / 24).astype(np.float32)

        merged["timestamps"] = timestamps
        merged["hour_encoded"] = hour_encoded
        merged["hours"] = hours.astype(np.float32)
        merged["months"] = months
        merged["n_steps"] = n_steps

        # ── Step 5: 重新计算 norm_params ───────────────────
        merged["norm_params"] = SmartDSLoader._compute_norm_params_static(merged)
        self.source_name = "merged"

        print(f"\n  合并完成: {n_steps} steps ({n_steps * 15 / 60 / 24:.1f} 天)")
        return merged

    def split(self, data: dict,
             train_ratio: float = 0.8) -> tuple[dict, dict]:
        """按时间 8:2 切分训练/验证集。"""
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

    支持 region 过滤：只加载指定区域的数据。
    """

    def __init__(self, data_dir: str | None = None, region: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else CHINA_DATA_DIR
        self.region = region  # e.g., "Shanghai" or None (加载全部)

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
        # 按 region 过滤（如指定了 region）
        if self.region:
            csv_files = [f for f in csv_files if f.name.startswith(self.region + "_")]
        pv_list, ghi_list, temp_list = [], [], []
        for fp in csv_files:
            try:
                # 尝试检测列数：读第一行判断有几个字段
                with open(fp, encoding="utf-8-sig") as f:
                    header = f.readline()
                n_cols = len(header.strip().split(","))
                # 3列格式: pv_power_kW,ghi_Wm2,temperature_C (中国合成数据)
                # 12列格式: Year,Month,Day,...,kW_Generated (SMART-DS格式)
                if n_cols == 3:
                    usecols = (0, 1, 2)  # pv, ghi, temperature
                else:
                    usecols = (11, 9, 8)  # kW_Generated, Temperature, Wind_Speed
                data = np.genfromtxt(fp, delimiter=",", skip_header=1,
                                     usecols=usecols, invalid_raise=False,
                                     encoding="utf-8-sig")
                if data.ndim == 1:
                    data = data.reshape(-1, 3)
                valid = data[~np.isnan(data).any(axis=1)]
                if len(valid) > 0:
                    pv_list.append(valid[:, 0])
                    ghi_list.append(valid[:, 1])
                    temp_list.append(valid[:, 2])
                    print(f"  {fp.name}: {len(valid)} rows (cols={n_cols})")
            except Exception as e:
                print(f"  WARN {fp.name}: {e}")
        if not pv_list:
            return {"pv_power": np.array([], dtype=np.float32),
                    "ghi": np.array([], dtype=np.float32),
                    "temperature": np.array([], dtype=np.float32)}
        return {
            "pv_power": np.concatenate(pv_list).astype(np.float32),
            "ghi": np.concatenate(ghi_list).astype(np.float32),
            "temperature": np.concatenate(temp_list).astype(np.float32),
        }

    def _load_load_files(self) -> dict[str, np.ndarray]:
        d = self.data_dir / "load"
        csv_files = sorted(d.glob("*.csv"))
        # 按 region 过滤（如指定了 region）
        if self.region:
            csv_files = [f for f in csv_files if f.name.startswith(self.region + "_")]
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


# ═══════════════════════════════════════════════════════════════
# 中国气象数据下载说明 + 本地 CSV 解析器
# ═══════════════════════════════════════════════════════════════

class ChinaMeteorologicalDownloader:
    """中国气象数据本地解析器。

    由于 NASA POWER API 在内网环境不可直接访问，请按以下步骤获取数据：

    【步骤 1：手动下载 NASA POWER CSV】
    1. 访问: https://power.larc.nasa.gov/
    2. 进入 "Data Viewer" → 选择坐标 (lat, lon)
    3. 选择参数: ALLSKY_SFC_SW_DWN (GHI), T2M (温度), WS2M (风速)
    4. 选择时间范围: 1 年
    5. 下载格式选择 CSV

    【步骤 2：放置 CSV 文件】
    将下载的 CSV 放入以下目录:
      data/china_data/solar/           → 光伏+气象 CSV
      data/china_data/load/             → 负荷 per-unit CSV

    【步骤 3：配置坐标和年份】
    使用 --lat, --lon, --year 参数指定数据源

    【步骤 4：运行】
    python data_loader.py --download --lat 31.23 --lon 121.47 --year 2023

    输出格式与 ChinaDataLoader 兼容。
    """

    # NASA POWER CSV 列索引 (下载的 CSV 格式)
    NASA_POWER_USECOLS = (0, 1, 2, 3, 4, 5)  # YR, MO, DY, HR, GHI, TEMP
    # 0=年, 1=月, 2=日, 3=小时, 4=GHI(W/m²), 5=温度(°C)

    def __init__(self, lat: float, lon: float,
                 pv_capacity_kw: float = 200.0,
                 panel_efficiency: float = 0.18,
                 panel_area_m2_per_kw: float = 8.0,
                 data_dir: str | None = None):
        self.lat = lat
        self.lon = lon
        self.pv_capacity_kw = pv_capacity_kw
        self.panel_efficiency = panel_efficiency
        self.panel_area_m2_per_kw = panel_area_m2_per_kw
        self.data_dir = Path(data_dir) if data_dir else CHINA_DATA_DIR

    def parse_nasa_power_csv(self, csv_path: str | Path) -> dict[str, np.ndarray]:
        """解析 NASA POWER 下载的 CSV 文件，转换为项目数据格式。

        Args:
            csv_path: NASA POWER 下载的 CSV 文件路径

        Returns:
            {"pv_power": ..., "ghi": ..., "temperature": ...}
        """
        import datetime

        csv_path = Path(csv_path)
        print(f"  Parsing: {csv_path.name}")

        # 读取 CSV
        try:
            data = np.genfromtxt(csv_path, delimiter=",",
                                  skip_header=1, invalid_raise=False,
                                  encoding="utf-8-sig")
            if data.ndim == 1:
                data = data.reshape(-1, len(self.NASA_POWER_USECOLS))
            print(f"    Rows: {len(data)}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse {csv_path}: {e}")

        # 提取列
        ghi_raw = data[:, 4]   # W/m²
        temp_raw = data[:, 5]    # °C

        # 过滤无效值
        valid_mask = ~(np.isnan(ghi_raw) | np.isnan(temp_raw) | (ghi_raw < 0))
        ghi_raw = ghi_raw[valid_mask]
        temp_raw = temp_raw[valid_mask]

        # GHI 转换: W/m² → kW/m² (÷1000)
        ghi_arr = ghi_raw.astype(np.float32) / 1000.0

        # 温度保持 °C
        temp_arr = temp_raw.astype(np.float32)

        # GHI → 光伏功率
        # PV_power = GHI * panel_area * efficiency * derating
        # derating = 0.8 (灰尘/温度/线损等综合折减)
        derating = 0.8
        panel_area = self.pv_capacity_kw * self.panel_area_m2_per_kw
        pv_power = ghi_arr * panel_area * self.panel_efficiency * derating
        pv_power = np.clip(pv_power, 0.0, self.pv_capacity_kw).astype(np.float32)

        print(f"    GHI range: {ghi_arr.min()*1000:.0f} - {ghi_arr.max()*1000:.0f} W/m2")
        print(f"    Temp range: {temp_arr.min():.1f} - {temp_arr.max():.1f} degC")
        print(f"    PV power range: {pv_power.min():.1f} - {pv_power.max():.1f} kW")

        return {
            "pv_power": pv_power,
            "ghi": ghi_arr,
            "temperature": temp_arr,
        }

    def generate_load_from_temperature(self, n_steps: int,
                                       temperature: np.ndarray,
                                       year: int) -> np.ndarray:
        """基于温度和 hour 合成负荷（与 download_year 中的逻辑相同）。"""
        import datetime

        # 15分钟步长 → 0.5h 粒度
        steps_per_day = 96
        hours = np.array([(i % steps_per_day) * (24.0 / steps_per_day) for i in range(n_steps)])
        hours = np.round(hours * 2) / 2  # 0.5h 粒度

        # 基础负荷（凌晨最低 ~30% 峰值）
        load_base = 0.3 + 0.2 * np.sin((hours - 6) * np.pi / 12)

        # 温度驱动空调负荷
        temp_factor = np.clip((temperature - 25.0) / 15.0, 0.0, 1.0) ** 0.8

        # 工作日因子
        day_of_week = np.array([
            datetime.datetime(year, 1, 1).weekday()
        ] * n_steps, dtype=np.float32)
        for i in range(n_steps):
            day_offset = i // steps_per_day
            day_of_week[i] = (datetime.datetime(year, 1, 1).weekday() + day_offset) % 7

        weekday_factor = np.where(day_of_week < 5, 1.0,
                                  np.where(day_of_week == 5, 0.7, 0.5))

        load_power = LOAD_PEAK_KW * load_base * (1.0 + temp_factor * 0.5) * weekday_factor
        return load_power.astype(np.float32)

    def process_local_files(self, solar_csv: str | Path,
                           year: int) -> dict[str, np.ndarray]:
        """处理本地 NASA POWER CSV 文件，生成完整数据。

        Args:
            solar_csv: 本地 solar CSV 路径（如 data/china_data/solar/solar_2023.csv）
            year: 数据年份
        """
        print(f"\n{'=' * 56}")
        print(f"  Processing local NASA POWER CSV")
        print(f"  Location: lat={self.lat}, lon={self.lon}")
        print(f"{'=' * 56}")

        # 解析 solar 数据
        solar_data = self.parse_nasa_power_csv(solar_csv)
        n_steps = len(solar_data["pv_power"])

        # 合成负荷
        print("\n  Generating synthetic load...")
        load_power = self.generate_load_from_temperature(
            n_steps, solar_data["temperature"], year
        )
        print(f"    Load range: {load_power.min():.1f} - {load_power.max():.1f} kW")

        # 保存到 data/china_data/
        solar_dir = self.data_dir / "solar"
        load_dir = self.data_dir / "load"
        solar_dir.mkdir(parents=True, exist_ok=True)
        load_dir.mkdir(parents=True, exist_ok=True)

        # 保存 solar CSV (兼容 ChinaDataLoader 格式)
        solar_fp = solar_dir / f"solar_{year}.csv"
        np.savetxt(solar_fp,
                   np.column_stack([solar_data["pv_power"],
                                   solar_data["ghi"] * 1000,  # 存回 W/m²
                                   solar_data["temperature"]]),
                   delimiter=",", fmt="%.4f",
                   header="pv_power_kW,ghi_Wm2,temperature_C",
                   comments="")
        print(f"\n  Saved: {solar_fp}")

        # 保存 load CSV
        load_fp = load_dir / f"load_{year}.csv"
        load_pu = load_power / LOAD_PEAK_KW
        np.savetxt(load_fp, load_pu, delimiter=",", fmt="%.6f",
                   header="load_per_unit",
                   comments="")
        print(f"  Saved: {load_fp}")

        return {
            **solar_data,
            "load_power": load_power,
        }


# ── 自测入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MUPC Data Loader - Unified entry point for all data sources"
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic China data (全年 15min 数据)")
    parser.add_argument("--unified", action="store_true",
                        help="Use UnifiedDataLoader (auto-detect source)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge SMART-DS + China synthetic data")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "china", "smart_ds"],
                        help="Data source: auto/china/smart_ds (default: auto)")
    parser.add_argument("--lat", type=float, default=31.23,
                        help="Latitude for China data (default: 31.23 Shanghai)")
    parser.add_argument("--lon", type=float, default=121.47,
                        help="Longitude for China data (default: 121.47 Shanghai)")
    parser.add_argument("--year", type=int, default=2023,
                        help="Year for synthetic data (default: 2023)")
    parser.add_argument("--pv-capacity", type=float, default=200.0,
                        help="PV capacity in kW (default: 200.0)")
    parser.add_argument("--peak-load", type=float, default=400.0,
                        help="Peak load in kW (default: 400.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic data (default: 42)")

    args = parser.parse_args()

    if args.generate:
        # 生成合成中国数据
        gen = ChinaSyntheticDataGenerator(
            lat=args.lat, lon=args.lon,
            pv_capacity_kw=args.pv_capacity,
            peak_load_kw=args.peak_load,
        )
        gen.save_to_csv(year=args.year, random_seed=args.seed)
        print(f"\n合成数据生成完成!")
        print(f"使用 UnifiedDataLoader 加载:")
        print(f"  from data_loader import UnifiedDataLoader")
        print(f"  loader = UnifiedDataLoader(lat={args.lat}, lon={args.lon})")
        print(f"  data = loader.load_all()")

    elif args.unified or args.source != "auto":
        # 使用统一加载器
        if args.source == "china" or (args.source == "auto" and
                (Path(__file__).parent / "data" / "china_data" / "solar").exists()):
            lat, lon = args.lat, args.lon
        else:
            lat, lon = args.lat, args.lon  # fallback to China even if auto

        loader = UnifiedDataLoader(
            lat=lat, lon=lon,
            pv_capacity_kw=args.pv_capacity,
            peak_load_kw=args.peak_load,
            auto_generate=True,
            merge_data=args.merge,
        )
        data = loader.load_all()
        train, val = loader.split(data)

        print(f"\n数据源: {loader.source_name}")
        print(f"总步数: {data['n_steps']} ({data['n_steps'] * 15 / 60 / 24:.1f} 天)")
        print(f"训练集: {train['n_steps']} 步")
        print(f"验证集: {val['n_steps']} 步")

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

    else:
        # 默认: SmartDSLoader 自测
        loader = SmartDSLoader()
        data = loader.load_all()
        train, val = loader.split(data)

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
