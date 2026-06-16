"""
MUPC 观测构建与归一化模块 (提取自 mupc_env.py)

包含:
- EnvState: 环境状态数据载体
- update_season_time_encoding(): 季节/时段编码
- build_observation(): 构建 63/64 维观测向量
- normalize_obs(): MinMax 归一化
"""

from dataclasses import dataclass

import numpy as np

from .constants import (
    PV_ARRAY_KW,
    LOAD_PEAK_KW,
    CONTRACT_DEMAND_KW,
    MODE_ID_MAP,
    NORM_SOC,
    NORM_PV_POWER,
    NORM_LOAD_POWER,
    NORM_GRID_POWER,
    NORM_VOLTAGE,
    NORM_BATTERY_POWER,
    NORM_PRICE,
    NORM_TARIFF,
    NORM_DEMAND,
    NORM_IRRADIANCE,
    NORM_TEMPERATURE,
    NORM_DISPATCH,
)


@dataclass
class EnvState:
    """observation 模块所需的环境快照，由 core.step() / core.reset() 填充。

    命名与 mupc_env.py 中 self._* 属性一一对应。
    """

    soc: float
    pv_power: float
    load_power: float
    grid_power: float
    load_rate: float
    battery_power_prev: float
    va: float
    vb: float
    vc: float
    q_realtime_margin: float

    current_price: float
    next_price: float
    tariff_id: float

    current_demand: float
    peak_demand: float

    solar_irradiance: float
    temperature: float

    dispatch_p_set: float

    season_encoding: np.ndarray       # (6,)
    time_period_encoding: np.ndarray  # (2,)

    safety_override_active: bool
    safety_override_p_ref: float
    override_consecutive: int
    override_ratio: float

    current_mode: str
    is_multi_mode: bool               # True if training mode == "all"


# ═══════════════════════════════════════════════════════════════
# 季节/时段编码
# ═══════════════════════════════════════════════════════════════

def update_season_time_encoding(hour: float, month: float
                                ) -> tuple[np.ndarray, np.ndarray]:
    """根据当前时间步计算季节和时段 one-hot 编码。

    季节编码 (6维): [灌溉季, 炒茶季, 空调季, 常规季, 保留, 保留]
    时段编码 (2维): [白天, 夜间]

    Args:
        hour: 当前小时 (0-23)
        month: 当前月份 (1-12)

    Returns:
        (season_encoding, time_period_encoding) 两个 numpy 数组
    """
    # 季节编码 (互斥月份分组)
    season = np.zeros(6, dtype=np.float32)
    if 3 <= month < 5:          # 3-4月
        season[0] = 1.0          # 灌溉季
    elif 5 <= month < 6:         # 5月
        season[1] = 1.0          # 炒茶季
    elif 6 <= month <= 8:        # 6-8月
        season[2] = 1.0          # 空调季
    else:
        season[3] = 1.0          # 常规季

    # 时段编码: 白天=6-18, 夜间=18-6
    time_period = np.zeros(2, dtype=np.float32)
    if 6 <= hour < 18:
        time_period[0] = 1.0     # 白天
    else:
        time_period[1] = 1.0     # 夜间

    return season, time_period


# ═══════════════════════════════════════════════════════════════
# 观测构建
# ═══════════════════════════════════════════════════════════════

def build_observation(state: EnvState,
                      forecast: np.ndarray) -> np.ndarray:
    """构建 63 维观测向量 (多模式追加 mode_id 为 64 维)。

    对齐下游 MUPC AI 引擎设计文档 v2.14 to_input_vector.

    Args:
        state: 环境状态快照
        forecast: 预测器输出 (30,) = [pv_forecast(15), load_forecast(15)]

    Returns:
        归一化后的观测向量 (63,) 或 (64,) 当 is_multi_mode=True
    """
    obs_dim = 64 if state.is_multi_mode else 63
    obs = np.zeros(obs_dim, dtype=np.float32)

    # ── D1 [0..9]: 10 标量 ──
    obs[0] = state.soc
    obs[1] = state.pv_power
    obs[2] = state.load_power
    obs[3] = state.grid_power
    obs[4] = state.load_rate
    obs[5] = state.battery_power_prev
    obs[6] = state.va
    obs[7] = state.vb
    obs[8] = state.vc
    obs[9] = state.q_realtime_margin

    # ── D2 [10..24] pv_forecast (15维) ──
    obs[10:25] = forecast[:15]

    # ── D2 [25..39] load_forecast (15维) ──
    obs[25:40] = forecast[15:30]

    # ── D3 [40..42] 电价 ──
    obs[40] = state.current_price
    obs[41] = state.next_price
    obs[42] = state.tariff_id

    # ── D4 [43..45] 需量 ──
    obs[43] = state.current_demand
    obs[44] = CONTRACT_DEMAND_KW
    obs[45] = state.peak_demand

    # ── D5 [46..47] 气象 ──
    obs[46] = state.solar_irradiance
    obs[47] = state.temperature

    # ── D6 [48] 调度 ──
    obs[48] = state.dispatch_p_set if abs(state.dispatch_p_set) > 1e-6 else 0.0

    # ── D7 [49] q_realtime_margin ──
    obs[49] = state.q_realtime_margin

    # ── D7 [50..55] season_encoding (6维) ──
    obs[50:56] = state.season_encoding

    # ── D7 [56..57] time_period_encoding (2维 one-hot, v2.14对齐下游) ──
    obs[56] = state.time_period_encoding[0]  # 白天
    obs[57] = state.time_period_encoding[1]  # 夜间

    # ── D9 [58..61] 安全覆盖状态 (v2.14, 4字段对齐下游) ──
    obs[58] = 1.0 if state.safety_override_active else 0.0
    obs[59] = state.safety_override_p_ref
    obs[60] = float(state.override_consecutive)
    obs[61] = state.override_ratio

    # ── mode_id [62] (可选) ──
    if state.is_multi_mode:
        obs[62] = MODE_ID_MAP.get(state.current_mode, 0.0)

    return normalize_obs(obs)


# ═══════════════════════════════════════════════════════════════
# 归一化
# ═══════════════════════════════════════════════════════════════

def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """应用 MinMax 归一化 (v2.14: 63维)。

    归一化范围使用 constants.py 中的命名常量，
    归一化范围随物理常量修正同步更新 (例如 NORM_GRID_POWER 由 ±500kW 改为 ±200kW)。

    Args:
        obs: 原始观测向量 (63,) 或 (64,)

    Returns:
        归一化观测向量，dtype=float32
    """
    out = obs.copy()

    # D1 [0..9]
    out[0] = _minmax(obs[0], *NORM_SOC)
    out[1] = _minmax(obs[1], *NORM_PV_POWER)
    out[2] = _minmax(obs[2], *NORM_LOAD_POWER)
    out[3] = _minmax(obs[3], *NORM_GRID_POWER)
    out[4] = obs[4]                      # transformer_load: identity [0,1]
    out[5] = _minmax(obs[5], *NORM_BATTERY_POWER)
    out[6] = _minmax(obs[6], *NORM_VOLTAGE)
    out[7] = _minmax(obs[7], *NORM_VOLTAGE)
    out[8] = _minmax(obs[8], *NORM_VOLTAGE)
    out[9] = obs[9]                      # q_realtime_margin: identity [0,1]

    # D2 pv [10..24]
    out[10:25] = _minmax(obs[10:25], *NORM_PV_POWER)

    # D2 load [25..39]
    out[25:40] = _minmax(obs[25:40], *NORM_LOAD_POWER)

    # D3 [40..42] 电价
    out[40] = _minmax(obs[40], *NORM_PRICE)
    out[41] = _minmax(obs[41], *NORM_PRICE)
    out[42] = _minmax(obs[42], *NORM_TARIFF)

    # D4 [43..45] 需量
    out[43] = _minmax(obs[43], *NORM_DEMAND)
    out[44] = _minmax(obs[44], *NORM_DEMAND)
    out[45] = _minmax(obs[45], *NORM_DEMAND)

    # D5 [46..47] 气象
    out[46] = _minmax(obs[46], *NORM_IRRADIANCE)
    out[47] = _minmax(obs[47], *NORM_TEMPERATURE)

    # D6 [48] dispatch_p_set
    out[48] = _minmax(obs[48], *NORM_DISPATCH)

    # D7 [49] q_realtime_margin: identity (v2.14修正)
    out[49] = obs[49]

    # D7 [50..55] season_encoding: one-hot, identity
    # D7 [56..57] time_period: one-hot, identity
    # D9 [58..61] safety_override: identity
    # mode_id [62]: identity

    return out.astype(np.float32)


def _minmax(x, lo: float, hi: float):
    """MinMax 归一化, 支持标量和数组。"""
    clipped = np.clip(x, lo, hi)
    result = (clipped - lo) / (hi - lo + 1e-9)
    if np.isscalar(x):
        return float(result)
    return result.astype(np.float32)
