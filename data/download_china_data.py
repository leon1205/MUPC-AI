"""
生成中国区域训练数据集 — 离线太阳辐照模型 + 合成负荷 + 省级分时电价。

纯离线运行, 无需任何外部 API 或网络请求。
使用 Haurwitz 晴空模型 + 气候区云量模拟生成太阳辐照。

输出目录: data/china_data/
  - solar/  光伏功率估算 CSV (15分钟分辨率)
  - load/   合成负荷 per-unit CSV (5种建筑类型)
  - pricing/ 分时电价 CSV
  - summary.json  数据集元信息

用法: python data/download_china_data.py
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── 配置 ──────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "china_data"
YEARS = [2020, 2021, 2022]
PV_CAPACITY_KW = 200.0       # 光伏容量 (kW)
PERFORMANCE_RATIO = 0.80     # 系统效率
G_REF = 1000.0               # STC 标准辐照 (W/m^2)

# ── 中国代表城市 ───────────────────────────────────────────────
# (城市名, 省份, 纬度, 经度, 气候特征, 年均晴空指数, 年均温度, 温度年振幅)
CITIES = [
    ("Beijing",     "Beijing",     39.90, 116.40, "北方平原/工商业",  0.55, 12.5, 15.0),
    ("Shanghai",    "Shanghai",    31.23, 121.47, "东部沿海/工商业",  0.42, 17.0, 12.0),
    ("Guangzhou",   "Guangdong",   23.13, 113.26, "南方沿海/工商业",  0.38, 22.5,  8.0),
    ("Chengdu",     "Sichuan",     30.57, 104.07, "西南盆地/低辐照",  0.28, 16.5, 10.0),
    ("Lhasa",       "Tibet",       29.65,  91.10, "高原/高辐照",      0.72,  8.5, 10.0),
    ("Urumqi",      "Xinjiang",    43.83,  87.62, "西北干旱/高辐照",  0.60,  7.0, 18.0),
    ("Harbin",      "Heilongjiang",45.75, 126.65, "东北严寒",         0.52,  3.5, 20.0),
    ("Kunming",     "Yunnan",      25.04, 102.68, "西南高原/温和",    0.50, 16.0,  6.0),
    ("Xi'an",       "Shaanxi",     34.27, 108.93, "西北/工商业",      0.48, 14.0, 13.0),
    ("Nanjing",     "Jiangsu",     32.06, 118.80, "东部/工商业",      0.43, 16.0, 12.0),
    ("Zhengzhou",   "Henan",       34.75, 113.63, "中部平原/农业",    0.50, 14.5, 13.0),
    ("Hangzhou",    "Zhejiang",    30.29, 120.15, "东部沿海/工商业",  0.40, 17.5, 12.0),
]

BUILDING_TYPES = ["office", "commercial", "residential", "industrial", "agriculture"]


# ═══════════════════════════════════════════════════════════════
# 离线太阳辐照模型
# ═══════════════════════════════════════════════════════════════

def compute_extraterrestrial_radiation(lat: float, doy: int) -> float:
    """计算大气层外太阳常数 (W/m^2)。"""
    solar_constant = 1367.0
    lat_rad = math.radians(lat)
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + doy) / 365.0)))
    # 日地距离修正
    ecc = 1.0 + 0.033 * math.cos(math.radians(360.0 * doy / 365.0))
    # 日出时角
    tan_prod = -math.tan(lat_rad) * math.tan(decl)
    tan_prod = max(-1.0, min(1.0, tan_prod))
    omega_s = math.acos(tan_prod)
    # 日平均地外辐照
    avg_rad = (solar_constant * ecc / math.pi *
               (omega_s * math.sin(lat_rad) * math.sin(decl) +
                math.cos(lat_rad) * math.cos(decl) * math.sin(omega_s)))
    return max(0.0, avg_rad)


def haurwitz_clear_sky(sun_elev_deg: float) -> float:
    """Haurwitz 晴空模型 — 基于太阳高度角的晴空 GHI 估算 (W/m^2)。"""
    if sun_elev_deg <= 0:
        return 0.0
    elev_rad = math.radians(sun_elev_deg)
    # 简化公式: GHI_clear ≈ 1098 * sin(elev) * exp(-0.057 / sin(elev))
    ghi = 1098.0 * math.sin(elev_rad) * math.exp(-0.057 / max(math.sin(elev_rad), 0.01))
    return max(0.0, min(1200.0, ghi))


def compute_sun_position(lat: float, lon: float, dt: datetime) -> tuple[float, float]:
    """计算太阳高度角和方位角。

    Returns: (elevation_deg, azimuth_deg)
    """
    doy = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    lat_rad = math.radians(lat)
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + doy) / 365.0)))

    # 均时差 (分钟)
    b = math.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)

    # 当地太阳时
    lstm = 15.0 * round(lon / 15.0)  # 当地标准子午线
    solar_time = hour + (4.0 * (lon - lstm) + eot) / 60.0
    hour_angle = math.radians((solar_time - 12.0) * 15.0)

    # 太阳高度角
    sin_elev = (math.sin(lat_rad) * math.sin(decl) +
                math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    # 太阳方位角
    cos_az = ((math.sin(decl) - math.sin(lat_rad) * sin_elev) /
              (math.cos(lat_rad) * math.cos(math.asin(max(-1.0, min(1.0, sin_elev)))) + 1e-9))
    azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if solar_time > 12:
        azimuth = 360.0 - azimuth

    return elev, azimuth


def simulate_city_weather(city: str, lat: float, lon: float,
                          clear_sky_index: float, annual_temp: float,
                          temp_amplitude: float, years: list[int]
                          ) -> dict[str, np.ndarray]:
    """为一座城市生成多年的 15 分钟分辨率气象/光伏数据。

    完全离线 — 使用太阳几何模型 + 气候统计参数。
    """
    all_pv = []
    all_ghi = []
    all_temp = []
    all_ws = []
    all_dni = []
    all_dhi = []

    for year in years:
        days_in_year = 366 if year % 4 == 0 else 365
        n_steps = days_in_year * 96

        ghi = np.zeros(n_steps, dtype=np.float32)
        dni = np.zeros(n_steps, dtype=np.float32)
        dhi = np.zeros(n_steps, dtype=np.float32)
        temp = np.zeros(n_steps, dtype=np.float32)
        ws = np.zeros(n_steps, dtype=np.float32)

        for step in range(n_steps):
            minutes = step * 15
            dt = datetime(year, 1, 1) + timedelta(minutes=minutes)
            hour = dt.hour + dt.minute / 60.0
            doy = dt.timetuple().tm_yday

            # ── 太阳位置 ──
            elev, _ = compute_sun_position(lat, lon, dt)

            # ── 晴空 GHI ──
            ghi_clear = haurwitz_clear_sky(elev)

            # ── 云量模拟 ──
            # 晴空指数 (clear sky index) = GHI / GHI_clear
            # 基于季节 + 随机变化模拟云量
            season_phase = 2 * math.pi * (doy - 172) / 365.0  # 夏季最高
            kt_seasonal = 0.15 * math.sin(season_phase)

            # 日间随机云量 (自相关: 每 3 小时更新一次)
            cloud_block = step // 12  # 12 步 = 3 小时
            np.random.seed(hash(f"{city}_{year}_{cloud_block}") % (2**31))
            kt_random = np.random.normal(0, 0.3)

            # 日循环: 早晚云量偏多 (晨雾/暮霭)
            hour_rad = 2 * math.pi * hour / 24.0
            kt_diurnal = -0.05 * math.cos(hour_rad)

            kt = clear_sky_index + kt_seasonal + kt_random + kt_diurnal
            kt = max(0.0, min(1.0, kt))

            ghi_step = ghi_clear * kt

            # DNI/DHI 分解 (Orgill-Hollands 模型)
            if ghi_step > 10 and ghi_clear > 10:
                kt_ratio = ghi_step / ghi_clear
                if kt_ratio < 0.35:
                    dhi_step = ghi_step * (1.0 - 0.249 * kt_ratio)
                elif kt_ratio < 0.75:
                    dhi_step = ghi_step * (1.557 - 1.84 * kt_ratio)
                else:
                    dhi_step = ghi_step * 0.177
                dni_step = (ghi_step - dhi_step) / max(math.sin(math.radians(elev)), 0.01)
            else:
                dhi_step = ghi_step
                dni_step = 0.0

            ghi[step] = max(0.0, ghi_step)
            dni[step] = max(0.0, min(1200.0, dni_step))
            dhi[step] = max(0.0, min(1200.0, dhi_step))

            # ── 温度模型 ──
            day_phase = 2 * math.pi * (doy - 20) / 365.0  # 1月20日最冷
            temp_base = annual_temp + temp_amplitude * 0.5 * math.sin(day_phase)
            # 日变化: 最低在05:00, 最高在14:00
            diurnal = -3.0 * math.cos(2 * math.pi * (hour - 14.0) / 24.0)
            temp_step = temp_base + diurnal
            # 辐照加热效应
            if ghi_step > 200:
                temp_step += 2.0 * (ghi_step / 1000.0)
            temp[step] = temp_step

            # ── 风速 ──
            ws_base = 3.0 + 2.0 * math.sin(season_phase)
            np.random.seed(hash(f"ws_{city}_{year}_{step}") % (2**31))
            ws[step] = max(0.0, ws_base + np.random.normal(0, 1.0))

        # ── 光伏功率估算 ──
        pv_power = ghi * PV_CAPACITY_KW * PERFORMANCE_RATIO / G_REF
        pv_power = np.clip(pv_power, 0.0, PV_CAPACITY_KW * 1.05)

        all_pv.append(pv_power)
        all_ghi.append(ghi)
        all_dni.append(dni)
        all_dhi.append(dhi)
        all_temp.append(temp)
        all_ws.append(ws)

    return {
        "pv_power_kw": np.concatenate(all_pv),
        "ghi": np.concatenate(all_ghi),
        "dni": np.concatenate(all_dni),
        "dhi": np.concatenate(all_dhi),
        "temperature": np.concatenate(all_temp),
        "wind_speed": np.concatenate(all_ws),
        "n_steps": sum(len(a) for a in all_pv),
    }


# ═══════════════════════════════════════════════════════════════
# 中国典型负荷曲线合成
# ═══════════════════════════════════════════════════════════════

def generate_china_load_profile(n_steps: int, building_type: str,
                                city: str, seed: int = 42) -> np.ndarray:
    """生成中国典型负荷 per-unit 曲线。

    基于中国建筑类型特征:
      - 办公: 08-18 工作日高峰, 周末低
      - 商业: 09-22 全天峰, 含餐饮晚高峰
      - 住宅: 早晚双峰 (06-09, 17-23)
      - 工业: 三班倒平坦基荷 + 白班叠加
      - 农业: 灌溉季 (4-9月) 日照时段高峰
    """
    np.random.seed(seed)
    days = n_steps // 96
    profile = np.zeros(n_steps, dtype=np.float32)

    for day in range(days):
        day_of_year = day % 365
        month = (day_of_year // 30) + 1
        is_weekend = (day % 7) >= 5
        base = np.zeros(96, dtype=np.float32)

        for step in range(96):
            hour = step * 0.25

            if building_type == "office":
                if is_weekend:
                    base[step] = 0.15 + 0.05 * np.random.random()
                elif 8 <= hour < 18:
                    peak = math.sin((hour - 8) * math.pi / 10)
                    base[step] = 0.70 + 0.30 * peak
                elif 6 <= hour < 8:
                    base[step] = 0.20 + 0.50 * (hour - 6) / 2
                elif 18 <= hour < 22:
                    base[step] = 0.30 - 0.15 * (hour - 18) / 4
                else:
                    base[step] = 0.08 + 0.03 * np.random.random()

            elif building_type == "commercial":
                if 9 <= hour < 14:
                    base[step] = 0.65 + 0.20 * math.sin((hour - 9) * math.pi / 5)
                elif 14 <= hour < 17:
                    base[step] = 0.75
                elif 17 <= hour < 21:
                    base[step] = 0.85 + 0.15 * math.sin((hour - 17) * math.pi / 4)
                elif 21 <= hour < 22:
                    base[step] = 0.40
                else:
                    base[step] = 0.08 + 0.03 * np.random.random()
                if is_weekend:
                    base[step] *= 0.85

            elif building_type == "residential":
                if 6 <= hour < 9:
                    base[step] = 0.40 + 0.40 * math.sin((hour - 6) * math.pi / 3)
                elif 9 <= hour < 17:
                    base[step] = 0.15 + 0.05 * np.random.random()
                elif 17 <= hour < 23:
                    base[step] = 0.50 + 0.50 * math.sin((hour - 17) * math.pi / 6)
                else:
                    base[step] = 0.06 + 0.04 * np.random.random()
                if is_weekend:
                    base[step] *= 1.20

            elif building_type == "industrial":
                base_load = 0.70
                if 8 <= hour < 17:
                    base[step] = base_load + 0.25
                elif 17 <= hour < 24:
                    base[step] = base_load + 0.10
                else:
                    base[step] = base_load
                if is_weekend:
                    base[step] *= 0.60

            elif building_type == "agriculture":
                if month in (4, 5, 6, 7, 8, 9):
                    if 6 <= hour < 10 or 16 <= hour < 20:
                        base[step] = 0.60 + 0.30 * np.random.random()
                    elif 10 <= hour < 16:
                        base[step] = 0.30 + 0.20 * np.random.random()
                    else:
                        base[step] = 0.05
                else:
                    base[step] = 0.02 + 0.02 * np.random.random()

        # 季节性调制
        if building_type in ("office", "commercial", "residential"):
            summer = 1.0 + 0.15 * math.sin((month - 1) * math.pi / 12) ** 2
            winter = 1.0 + 0.10 * math.sin((month + 5) * math.pi / 12) ** 2
            base *= max(summer, winter)

        base += np.random.normal(0, 0.03, 96)
        profile[day * 96:(day + 1) * 96] = base

    return np.clip(profile, 0.05, 1.2).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# 中国省级分时电价
# ═══════════════════════════════════════════════════════════════

PROVINCE_TARIFFS = {
    "Beijing":     {"peak": [(10,13),(17,20)], "valley": [(23,7)],
                    "prices": {"peak":1.20,"flat":0.80,"valley":0.40}},
    "Shanghai":    {"peak": [(8,11),(13,15),(18,21)], "valley": [(22,6)],
                    "prices": {"peak":1.35,"flat":0.85,"valley":0.42}},
    "Guangdong":   {"peak": [(10,12),(14,17),(19,21)], "valley": [(0,8)],
                    "prices": {"peak":1.28,"flat":0.78,"valley":0.35}},
    "Sichuan":     {"peak": [(10,12),(15,21)], "valley": [(23,7)],
                    "prices": {"peak":1.10,"flat":0.72,"valley":0.38}},
    "Tibet":       {"peak": [(10,13),(18,21)], "valley": [(23,7)],
                    "prices": {"peak":0.90,"flat":0.60,"valley":0.30}},
    "Xinjiang":    {"peak": [(10,13),(18,21)], "valley": [(23,7)],
                    "prices": {"peak":0.95,"flat":0.65,"valley":0.32}},
    "Heilongjiang":{"peak": [(9,13),(17,21)], "valley": [(22,7)],
                    "prices": {"peak":1.05,"flat":0.70,"valley":0.35}},
    "Yunnan":      {"peak": [(9,12),(17,21)], "valley": [(23,8)],
                    "prices": {"peak":0.85,"flat":0.55,"valley":0.28}},
    "Shaanxi":     {"peak": [(8,12),(17,21)], "valley": [(23,7)],
                    "prices": {"peak":1.05,"flat":0.70,"valley":0.36}},
    "Jiangsu":     {"peak": [(8,12),(17,21)], "valley": [(23,7)],
                    "prices": {"peak":1.25,"flat":0.78,"valley":0.39}},
    "Henan":       {"peak": [(8,12),(17,21)], "valley": [(23,7)],
                    "prices": {"peak":1.10,"flat":0.73,"valley":0.37}},
    "Zhejiang":    {"peak": [(8,11),(13,17),(19,21)], "valley": [(22,6)],
                    "prices": {"peak":1.30,"flat":0.82,"valley":0.40}},
    "default":     {"peak": [(8,12),(17,21)], "valley": [(23,7)],
                    "prices": {"peak":1.15,"flat":0.75,"valley":0.38}},
}


def _in_range(hour: float, start: int, end: int) -> bool:
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def generate_china_tou(n_steps: int, province: str) -> dict[str, np.ndarray]:
    """生成省级分时电价序列。"""
    t = PROVINCE_TARIFFS.get(province, PROVINCE_TARIFFS["default"])
    prices = np.zeros(n_steps, dtype=np.float32)
    tids = np.zeros(n_steps, dtype=np.int32)

    for i in range(n_steps):
        hour = (i * 0.25) % 24
        tid, pr = 1, t["prices"]["flat"]
        for hs, he in t["valley"]:
            if _in_range(hour, hs, he):
                tid, pr = 0, t["prices"]["valley"]
                break
        for hs, he in t["peak"]:
            if _in_range(hour, hs, he):
                tid, pr = 2, t["prices"]["peak"]
                break
        prices[i] = pr
        tids[i] = tid

    next_p = np.roll(prices, -1)
    next_p[-1] = prices[-1]
    return {"current_price": prices, "next_price": next_p, "tariff_id": tids}


# ═══════════════════════════════════════════════════════════════
# CSV 保存
# ═══════════════════════════════════════════════════════════════

def save_solar_csv(data: dict, city_key: str, data_dir: Path) -> str:
    d = data_dir / "solar"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{city_key}_pv.csv"
    with open(fp, "w") as f:
        f.write("Year,Month,Day,Hour,Minute,GHI,DNI,DHI,"
                "Wind_Speed,Temperature,PoA_Irradiance,kW_Generated\n")
        for i in range(data["n_steps"]):
            dt = datetime(2022, 1, 1) + timedelta(minutes=i * 15)
            f.write(f"{dt.year},{dt.month},{dt.day},{dt.hour},{dt.minute},"
                    f"{data['ghi'][i]:.1f},{data['dni'][i]:.1f},{data['dhi'][i]:.1f},"
                    f"{data['wind_speed'][i]:.1f},{data['temperature'][i]:.1f},"
                    f"{data['ghi'][i]:.1f},{data['pv_power_kw'][i]:.1f}\n")
    return str(fp)


def save_load_csv(profile: np.ndarray, city_key: str, bt: str, data_dir: Path) -> str:
    d = data_dir / "load"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{city_key}_{bt}_pu.csv"
    with open(fp, "w") as f:
        for v in profile:
            f.write(f"{v:.8f}\n")
    return str(fp)


def save_pricing_csv(tou: dict, city_key: str, data_dir: Path) -> str:
    d = data_dir / "pricing"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{city_key}_tou.csv"
    with open(fp, "w") as f:
        f.write("current_price,next_price,tariff_id\n")
        for i in range(tou["current_price"].shape[0]):
            f.write(f"{tou['current_price'][i]:.3f},{tou['next_price'][i]:.3f},"
                    f"{tou['tariff_id'][i]}\n")
    return str(fp)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  中国区域 MUPC 训练数据集生成 (离线模式)")
    print("=" * 60)
    print(f"\n模型: Haurwitz 晴空 + 气候区云量 + 中国典型负荷 + 省级电价")
    print(f"城市: {len(CITIES)} 个 | 建筑: {len(BUILDING_TYPES)} 种 | 年份: {YEARS}")
    print(f"输出: {DATA_DIR}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"cities": {}, "stats": {}}
    np.random.seed(42)

    for city, province, lat, lon, climate, csi, ann_temp, temp_amp in CITIES:
        city_key = f"{city}_{province}"
        print(f"\n{'─' * 50}")
        print(f"  {city} ({province}) — {climate} | CSI={csi} Tavg={ann_temp}C")

        # ── 1. 气象+光伏 ──
        weather = simulate_city_weather(
            city, lat, lon, csi, ann_temp, temp_amp, YEARS,
        )
        n_total = weather["n_steps"]
        days_total = n_total * 15 / 60 / 24
        pv_mean = float(weather["pv_power_kw"].mean())
        print(f"  气象: {n_total} 步 ({days_total:.0f} 天) | PV 均值={pv_mean:.1f} kW")

        fp_solar = save_solar_csv(weather, city_key, DATA_DIR)
        print(f"  [OK] 光伏: {fp_solar}")

        # ── 2. 负荷 ──
        for bt in BUILDING_TYPES:
            seed = abs(hash(f"{city}_{bt}")) % (2**16)
            lp = generate_china_load_profile(n_total, bt, city, seed)
            # Per-unit 归一化
            pmax = np.percentile(lp, 99)
            if pmax > 1e-6:
                lp = lp / pmax
            fp_load = save_load_csv(lp, city_key, bt, DATA_DIR)
        print(f"  [OK] 负荷: {len(BUILDING_TYPES)} 种建筑类型")

        # ── 3. 电价 ──
        tou = generate_china_tou(n_total, province)
        fp_price = save_pricing_csv(tou, city_key, DATA_DIR)
        print(f"  [OK] 电价: {fp_price} ({province} 省级标准)")

        summary["cities"][city_key] = {
            "province": province, "lat": lat, "lon": lon,
            "climate": climate, "clear_sky_index": csi,
            "n_steps": n_total, "days": round(days_total),
            "pv_mean_kw": round(pv_mean, 1),
            "pv_max_kw": round(float(weather["pv_power_kw"].max()), 1),
            "ghi_mean_wm2": round(float(weather["ghi"].mean()), 1),
            "temp_mean_c": round(float(weather["temperature"].mean()), 1),
        }

    # ── 汇总 ──
    summary["stats"] = {
        "total_cities": len(summary["cities"]),
        "buildings": BUILDING_TYPES,
        "years": YEARS,
        "pv_capacity_kw": PV_CAPACITY_KW,
        "total_days_equiv": sum(c["days"] for c in summary["cities"].values()),
    }

    fp_summary = DATA_DIR / "summary.json"
    with open(fp_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  生成完成！")
    print(f"  城市: {len(summary['cities'])} 个 × {len(BUILDING_TYPES)} 种负荷")
    print(f"  数据总量: {summary['stats']['total_days_equiv']} 天等效")
    print(f"  汇总: {fp_summary}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
