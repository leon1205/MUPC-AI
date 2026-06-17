"""
MUPC 观测构建与归一化模块 (提取自 mupc_env.py)

包含:
- EnvState: 环境状态数据载体
- update_season_time_encoding(): 季节/时段编码
- build_observation(): 构建 78/79 维观测向量 (v2.17 对齐下游 v2.14)
- normalize_obs(): MinMax 归一化

观测维度构成 (78 维单模式, 79 维多模式, 对齐下游 v2.14 §3.6/§4.4):
  D1  [0..8]    9 维 实时数据 (soc/pv/load/grid/transformer_load/battery_power/va/vb/vc)
  D2  [9..23]  15 维 pv_forecast
  D2  [24..38] 15 维 load_forecast
  D3  [39..41]  3 维 电价 (current/next/tariff_id, peak/valley_price 不入向量)
  D4  [42..44]  3 维 需量 (current/contract/peak_this_month)
  D5  [45..46]  2 维 气象 (solar_irradiance/temperature)
  D6  [47]      1 维 dispatch_p_set (dispatch_q_set 不入向量)
  D7  [48]      1 维 q_realtime_margin
  D8  [49..54]  6 维 season_encoding
  D8  [55..56]  2 维 time_period_encoding (白天/夜间)
  D9  [57..60]  4 维 safety_override (active/p_ref/consecutive/ratio)
  D10 [61..75] 15 维 load_forecast_quantiles (P3.3~P96.7)
  D10 [76]      1 维 shock_load_probability
  D10 [77]      1 维 base_load (50% 分位数)
  [78]          1 维 mode_id (仅 is_multi_mode=True)

Python ↔ Rust FusedSystemState 字段映射表 (v2.17, ONNX 数组顺序一致):
  Python (EnvState)          Rust (FusedSystemState)      观测索引
  ────────────────────────   ─────────────────────────    ────────
  load_rate                  transformer_load              [4]
  va / vb / vc               voltage_phase_a / _b / _c     [6]/[7]/[8]
  battery_power_prev         battery_power                 [5]
  current_price              current_electricity_price     [39]
  next_price                 next_period_price             [40]
  tariff_id                  price_tariff_id               [41]
  peak_demand                peak_demand_this_month        [44]
  solar_irradiance           solar_irradiance              [45]
  dispatch_p_set             dispatch_p_set                [47]
  safety_override_active     safety_override_active        [57]
  safety_override_p_ref      safety_override_p_ref         [58]
  override_consecutive       safety_override_consecutive   [59]
  override_ratio             safety_override_ratio         [60]
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
    NORM_DISPATCH_Q,
    NORM_QUANTILE_LOAD,
    NORM_SHOCK_PROBABILITY,
    NORM_BASE_LOAD,
    NORM_QUANTILES,
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
    peak_price: float                # v2.14 新增
    valley_price: float              # v2.14 新增

    current_demand: float
    peak_demand: float

    solar_irradiance: float
    temperature: float

    dispatch_p_set: float
    dispatch_q_set: float            # v2.14 新增

    season_encoding: np.ndarray       # (6,)
    time_period_encoding: np.ndarray  # (2,)

    safety_override_active: bool
    safety_override_p_ref: float
    override_consecutive: int
    override_ratio: float

    # D10 概率负荷预测 (v2.14 新增, 17 维)
    load_forecast_quantiles: np.ndarray   # (15,) 分位数负荷预测 P10/P50/P90...
    shock_load_probability: float          # 冲击负荷发生概率
    base_load: float                       # 基荷 (50% 分位数)

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
    """构建 78 维观测向量 (多模式追加 mode_id 为 79 维)。

    对齐下游 MUPC AI 引擎设计文档 v2.14 §3.6 to_input_vector().

    布局 (v2.14, 78 维单模式):
      D1  [0..8]    9 维 实时数据 (soc/pv/load/grid/transformer/battery/3相电压)
      D2  [9..23]  15 维 pv_forecast_15min
      D2  [24..38] 15 维 load_forecast_15min
      D3  [39..41]  3 维 电价 (current/next/tariff_id, 不含 peak/valley)
      D4  [42..44]  3 维 需量 (current/contract/peak_this_month)
      D5  [45..46]  2 维 气象 (irradiance/temperature)
      D6  [47]      1 维 调度 dispatch_p_set (不含 dispatch_q_set)
      D7  [48]      1 维 q_realtime_margin (实时模块无功裕度)
      D8  [49..54]  6 维 season_encoding (灌溉/炒茶/空调/常规/保留/保留)
      D8  [55..56]  2 维 time_period_encoding (白天/夜间)
      D9  [57..60]  4 维 safety_override (active/p_ref/consecutive/ratio)
      D10 [61..75] 15 维 load_forecast_quantiles (P10..P90 步进)
      D10 [76]      1 维 shock_load_probability
      D10 [77]      1 维 base_load (50% 分位数)
      [78]          1 维 mode_id (仅 is_multi_mode=True)

    注: EnvState 中保留 peak_price/valley_price/dispatch_q_set 字段供
    奖励函数和审计使用, 但**不进入** to_input_vector() (与下游 Rust 端一致).

    Args:
        state: 环境状态快照
        forecast: 预测器输出 (30,) = [pv_forecast(15), load_forecast(15)]

    Returns:
        归一化后的观测向量 (78,) 或 (79,) 当 is_multi_mode=True
    """
    obs_dim = 79 if state.is_multi_mode else 78
    obs = np.zeros(obs_dim, dtype=np.float32)

    # ── D1 [0..8]: 9 标量 (不含 q_realtime_margin, 移至 D7) ──
    obs[0] = state.soc
    obs[1] = state.pv_power
    obs[2] = state.load_power
    obs[3] = state.grid_power
    obs[4] = state.load_rate
    obs[5] = state.battery_power_prev
    obs[6] = state.va
    obs[7] = state.vb
    obs[8] = state.vc

    # ── D2 [9..23] pv_forecast (15维) ──
    obs[9:24] = forecast[:15]

    # ── D2 [24..38] load_forecast (15维) ──
    obs[24:39] = forecast[15:30]

    # ── D3 [39..41] 电价 (3 维, 不含 peak/valley) ──
    obs[39] = state.current_price
    obs[40] = state.next_price
    obs[41] = state.tariff_id

    # ── D4 [42..44] 需量 (3 维) ──
    obs[42] = state.current_demand
    obs[43] = CONTRACT_DEMAND_KW
    obs[44] = state.peak_demand

    # ── D5 [45..46] 气象 (2 维) ──
    obs[45] = state.solar_irradiance
    obs[46] = state.temperature

    # ── D6 [47] dispatch_p_set (1 维, 不含 dispatch_q_set) ──
    obs[47] = state.dispatch_p_set if abs(state.dispatch_p_set) > 1e-6 else 0.0

    # ── D7 [48] q_realtime_margin (1 维) ──
    obs[48] = state.q_realtime_margin

    # ── D8 [49..54] season_encoding (6 维 one-hot) ──
    obs[49:55] = state.season_encoding

    # ── D8 [55..56] time_period_encoding (2 维 one-hot) ──
    obs[55] = state.time_period_encoding[0]  # 白天
    obs[56] = state.time_period_encoding[1]  # 夜间

    # ── D9 [57..60] safety_override (4 维) ──
    obs[57] = 1.0 if state.safety_override_active else 0.0
    obs[58] = state.safety_override_p_ref
    obs[59] = float(state.override_consecutive)
    obs[60] = state.override_ratio

    # ── D10 [61..75] load_forecast_quantiles (15 维) ──
    obs[61:76] = state.load_forecast_quantiles[:15]

    # ── D10 [76] shock_load_probability (1 维) ──
    obs[76] = state.shock_load_probability

    # ── D10 [77] base_load (1 维, 50% 分位数) ──
    obs[77] = state.base_load

    # ── mode_id [78] (可选, 多模式训练) ──
    if state.is_multi_mode:
        obs[78] = MODE_ID_MAP.get(state.current_mode, 0.0)

    return normalize_obs(obs)


# ═══════════════════════════════════════════════════════════════
# 归一化
# ═══════════════════════════════════════════════════════════════

def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """应用 MinMax 归一化 (v2.14: 78 维单模式 / 79 维多模式)。

    归一化范围使用 constants.py 中的命名常量，
    归一化范围随物理常量修正同步更新 (例如 NORM_GRID_POWER 由 ±500kW 改为 ±200kW)。

    Args:
        obs: 原始观测向量 (78,) 或 (79,)

    Returns:
        归一化观测向量，dtype=float32
    """
    out = obs.copy()

    # D1 [0..8] 9 维 (q_realtime_margin 已移至 D7 [48])
    out[0] = _minmax(obs[0], *NORM_SOC)
    out[1] = _minmax(obs[1], *NORM_PV_POWER)
    out[2] = _minmax(obs[2], *NORM_LOAD_POWER)
    out[3] = _minmax(obs[3], *NORM_GRID_POWER)
    out[4] = obs[4]                      # transformer_load: identity [0,1]
    out[5] = _minmax(obs[5], *NORM_BATTERY_POWER)
    out[6] = _minmax(obs[6], *NORM_VOLTAGE)
    out[7] = _minmax(obs[7], *NORM_VOLTAGE)
    out[8] = _minmax(obs[8], *NORM_VOLTAGE)

    # D2 pv [9..23] 15 维
    out[9:24] = _minmax(obs[9:24], *NORM_PV_POWER)

    # D2 load [24..38] 15 维
    out[24:39] = _minmax(obs[24:39], *NORM_LOAD_POWER)

    # D3 [39..41] 3 维 (不含 peak/valley_price)
    out[39] = _minmax(obs[39], *NORM_PRICE)
    out[40] = _minmax(obs[40], *NORM_PRICE)
    out[41] = _minmax(obs[41], *NORM_TARIFF)

    # D4 [42..44] 3 维
    out[42] = _minmax(obs[42], *NORM_DEMAND)
    out[43] = _minmax(obs[43], *NORM_DEMAND)
    out[44] = _minmax(obs[44], *NORM_DEMAND)

    # D5 [45..46] 2 维
    out[45] = _minmax(obs[45], *NORM_IRRADIANCE)
    out[46] = _minmax(obs[46], *NORM_TEMPERATURE)

    # D6 [47] dispatch_p_set
    out[47] = _minmax(obs[47], *NORM_DISPATCH)

    # D7 [48] q_realtime_margin: identity
    out[48] = obs[48]

    # D8 [49..54] season_encoding: one-hot, identity
    # D8 [55..56] time_period: one-hot, identity
    # D9 [57..60] safety_override: identity

    # D10 [61..75] load_forecast_quantiles 15 维
    out[61:76] = _minmax(obs[61:76], *NORM_QUANTILE_LOAD)

    # D10 [76] shock_load_probability: identity [0,1]
    out[76] = obs[76]

    # D10 [77] base_load: scale to [0,1] by LOAD_PEAK_KW
    out[77] = _minmax(obs[77], *NORM_BASE_LOAD)

    # mode_id [78] (可选): identity

    return out.astype(np.float32)


def _minmax(x, lo: float, hi: float):
    """MinMax 归一化, 支持标量和数组。"""
    clipped = np.clip(x, lo, hi)
    result = (clipped - lo) / (hi - lo + 1e-9)
    if np.isscalar(x):
        return float(result)
    return result.astype(np.float32)
