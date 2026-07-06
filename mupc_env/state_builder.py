"""
状态构建器 — v3.1 从 mupc_env/core.py 提取 (A-4 core.py 拆分)。

将核心状态构建逻辑从 MupcEnv 解耦为纯函数, 便于独立测试和维护。
core.py 中的 _make_env_state / _make_reward_dict 保留为薄委托层。
"""

import numpy as np
from typing import Optional
from . import observation


# ═══════════════════════════════════════════════════════════════
# EnvState 构建 (原 core.py _make_env_state)
# ═══════════════════════════════════════════════════════════════

def build_env_state(*,
    # 主状态
    soc: float, step_idx: int, data: dict,
    grid_power: float, load_rate: float,
    battery_power_prev: float,
    va: float, vb: float, vc: float,
    q_realtime_margin: float,
    season_encoding: np.ndarray, time_period_encoding: np.ndarray,
    # 需量
    current_demand: float, peak_demand: float,
    # 安全
    safety_override_active: bool, safety_override_p_ref: float,
    override_consecutive: int, override_ratio: float,
    # 模式
    current_mode: str, is_multi_mode: bool,
    # D10 预测 (外部传入, 含冷启动 fallback)
    forecast: np.ndarray | None = None,
    d10_warmup_count: int = 0,
) -> observation.EnvState:
    """从显式参数构建 EnvState 快照 (纯函数, 无 self 依赖)。

    D10 概率负荷预测冷启动保护:
      - LSTM D10 头 warmup count >= threshold: 使用 LSTM 推理结果
      - 否则: fallback 到 data 合成或简单数学合成
    """
    # D10 冷启动判断 (阈值: _D10_WARMUP_THRESHOLD = 100)
    use_lstm_d10 = (
        forecast is not None
        and len(forecast) >= 47
        and d10_warmup_count >= 100
    )
    if use_lstm_d10:
        quantiles = forecast[30:45].astype(np.float32)
        shock_prob = float(np.clip(forecast[45], 0.0, 1.0))
        base_load = float(max(0.0, forecast[46]))
    elif "load_forecast_quantiles" in data:
        quantiles = data["load_forecast_quantiles"][step_idx].astype(np.float32)
        shock_prob = float(data["shock_load_probability"][step_idx]) \
            if "shock_load_probability" in data else 0.0
        base_load = float(data["base_load"][step_idx]) \
            if "base_load" in data else float(data["load_power"][step_idx])
    else:
        base = float(data["load_power"][step_idx])
        quantiles = (base * np.linspace(0.85, 1.27, 15)).astype(np.float32)
        shock_prob = 0.0
        base_load = base

    return observation.EnvState(
        soc=soc,
        pv_power=float(data["pv_power"][step_idx]),
        load_power=float(data["load_power"][step_idx]),
        grid_power=grid_power,
        load_rate=load_rate,
        battery_power_prev=battery_power_prev,
        va=va, vb=vb, vc=vc,
        q_realtime_margin=q_realtime_margin,
        current_price=float(data["current_electricity_price"][step_idx]),
        next_price=float(data["next_period_price"][step_idx]),
        tariff_id=float(data["price_tariff_id"][step_idx]),
        peak_price=_safe_get(data, "peak_price", step_idx, 1.5),
        valley_price=_safe_get(data, "valley_price", step_idx, 0.40),
        current_demand=current_demand,
        peak_demand=peak_demand,
        solar_irradiance=float(data["solar_irradiance"][step_idx]),
        temperature=float(data["temperature"][step_idx]),
        dispatch_p_set=float(data["dispatch_p_set"][step_idx]),
        dispatch_q_set=float(data["dispatch_q_set"][step_idx])
            if "dispatch_q_set" in data else 0.0,
        season_encoding=season_encoding,
        time_period_encoding=time_period_encoding,
        safety_override_active=safety_override_active,
        safety_override_p_ref=safety_override_p_ref,
        override_consecutive=override_consecutive,
        override_ratio=override_ratio,
        load_forecast_quantiles=quantiles,
        shock_load_probability=shock_prob,
        base_load=base_load,
        current_mode=current_mode,
        is_multi_mode=is_multi_mode,
    )


# ═══════════════════════════════════════════════════════════════
# 奖励函数输入组装 (原 core.py _make_reward_dict)
# ═══════════════════════════════════════════════════════════════

def build_reward_dict(*,
    # 动作
    p_batt: float, q_batt: float, load_shed: float, pv_limit: float,
    p_pv_raw: float, p_load_raw: float, p_load_eff: float,
    grid_power: float, load_rate: float,
    load_rate_unclamped: float | None = None,
    k_droop: float = 0.0, prev_k_droop: float = 0.0,
    # SOC
    soc: float, soc_new: float, soc_clipped: bool = False,
    # 电压
    va: float, vb: float, vc: float,
    prev_p_batt: float, prev_v_avg: float, prev_v_dev: float,
    # 安全
    voltage_violation_count: int,
    safety_override_active: bool, safety_override_p_ref: float,
    override_consecutive: int, override_ratio: float,
    # 辅助
    q_realtime_margin: float,
    current_demand: float,
    prev_p_batt_raw: float,
    step_idx: int, data: dict,
) -> dict:
    """组装奖励函数所需的 r dict (纯函数, 无 self 依赖)。"""
    lr_unc = load_rate_unclamped if load_rate_unclamped is not None else load_rate
    return {
        "p_batt": p_batt,
        "q_batt": q_batt,
        "load_shed": load_shed,
        "pv_limit": pv_limit,
        "p_pv_raw": p_pv_raw,
        "p_load_raw": p_load_raw,
        "p_load_eff": p_load_eff,
        "grid_power": grid_power,
        "load_rate": load_rate,
        "load_rate_unclamped": lr_unc,
        "soc": soc,
        "soc_new": soc_new,
        "soc_clipped": soc_clipped,
        "va": va, "vb": vb, "vc": vc,
        "prev_p_batt": prev_p_batt,
        "prev_v_avg": prev_v_avg,
        "prev_v_dev": prev_v_dev,
        "k_droop": k_droop,
        "prev_k_droop": prev_k_droop,
        "voltage_violation_count": voltage_violation_count,
        "safety_override_active": safety_override_active,
        "safety_override_p_ref": safety_override_p_ref,
        "override_consecutive": override_consecutive,
        "override_ratio": override_ratio,
        "q_realtime_margin": q_realtime_margin,
        "current_demand": current_demand,
        "prev_p_batt_raw": prev_p_batt_raw,
        "dispatch_p_set": float(data["dispatch_p_set"][step_idx]),
        "current_price": float(data["current_electricity_price"][step_idx]),
        "base_load": float(data["base_load"][step_idx])
            if "base_load" in data else float(data["load_power"][step_idx]),
        "load_forecast_quantiles": data["load_forecast_quantiles"][step_idx].astype(np.float32)
            if "load_forecast_quantiles" in data
            else (float(data["load_power"][step_idx]) * np.linspace(0.85, 1.27, 15)).astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _safe_get(data: dict, key: str, idx: int, default: float) -> float:
    """从 data[key] 安全取值, key 不存在时返回默认值."""
    if key in data:
        return float(data[key][idx])
    return default
