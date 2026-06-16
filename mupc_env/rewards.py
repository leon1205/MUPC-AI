"""
MUPC 奖励函数模块 (提取自 mupc_env.py)

包含:
- compute_reward(): 奖励调度器
- _reward_agri(): MODE-01 农网灌溉 (13 个子奖励函数)
- _reward_arbitrage(): MODE-02 自主套利
- _reward_demand(): MODE-03 需量控制
- _reward_vpp(): MODE-04 虚拟电厂
- _reward_green(): MODE-05 极致绿色

设计原则: 纯函数，不访问 self 状态。所有状态通过 r dict 传入。
Welford 原始奖励通过 info["welford_raw"] 回传给 core.py。
"""

import math

import numpy as np

from .constants import (
    BATTERY_CAPACITY_KWH,
    P_BATT_MAX_KW,
    LOAD_SHED_MAX_KW,
    VOLTAGE_DEADBAND,
    Q_MARGIN_THRESHOLD,
    VOLTAGE_HIGH_LIMIT,
    SOC_CRITICAL,
    CONTRACT_DEMAND_KW,
    GRID_EMISSION_FACTOR,
    DT_HOURS,
)


# ═══════════════════════════════════════════════════════════════
# 调度器
# ═══════════════════════════════════════════════════════════════

def compute_reward(mode: str, weights: dict, r: dict) -> tuple[float, dict]:
    """根据当前场景选择奖励函数并计算。

    Args:
        mode: 场景模式字符串 (如 "MODE-01")
        weights: 完整权重字典 {mode: [w1, w2, ...]}
        r: 奖励计算所需状态 dict

    Returns:
        (total_reward, info_dict) 其中 info 包含调试信息和
        info["welford_raw"]: float | None 供 core 更新 Welford 统计量
    """
    mode_weights = DEFAULT_WEIGHTS_MAP.get(mode, [])
    w = weights.get(mode, mode_weights)

    if mode == "MODE-01":
        return _reward_agri(r, w)
    elif mode == "MODE-02":
        return _reward_arbitrage(r, w)
    elif mode == "MODE-03":
        return _reward_demand(r, w)
    elif mode == "MODE-04":
        return _reward_vpp(r, w)
    elif mode == "MODE-05":
        return _reward_green(r, w)
    else:
        return 0.0, {}


# 模块内默认权重 (与 constants.py 保持一致，用作调度器 fallback)
DEFAULT_WEIGHTS_MAP = {
    "MODE-01": [1.0, 0.5, 2.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    "MODE-02": [1.0, 1.0, 2.0],
    "MODE-03": [1.0, 0.5],
    "MODE-04": [1.0, 2.0, 1.0],
    "MODE-05": [1.0, 1.0],
}


# ═══════════════════════════════════════════════════════════════
# MODE-01 子奖励函数 (v2.13 精细化奖励)
# ═══════════════════════════════════════════════════════════════

def _compute_pv_consumption(p_pv_raw: float, p_load_raw: float,
                            p_batt: float, v_avg: float) -> float:
    """光伏消纳率 (v2.8 差异化弃光奖励)。"""
    pv_total = max(p_pv_raw, 1e-6)
    pv_self = min(p_pv_raw, p_load_raw) + max(0.0, -p_batt)
    r_pv = min(pv_self / pv_total, 1.0)
    if v_avg >= VOLTAGE_HIGH_LIMIT:
        if p_batt < 0:
            r_pv = min(r_pv, 1.0)  # 充电消纳光伏 — 正确行为
        else:
            r_pv = -20.0           # 反而在放电 — 严厉惩罚
    return r_pv


def _compute_alpha(soc_new: float, q_margin: float,
                   voltage_violation_count: int) -> float:
    """自适应损耗系数 α(s)。"""
    if soc_new < SOC_CRITICAL:
        return 3.0   # SOC 极低保护
    elif q_margin <= Q_MARGIN_THRESHOLD and voltage_violation_count >= 2:
        return 0.2   # 电压支撑模式
    else:
        return 1.0   # 常规调度


def _compute_overload(lr_unc: float) -> float:
    """过载惩罚 (R-04 v2.12 分段惩罚)。"""
    if lr_unc < 0.75:
        return 0.0
    elif lr_unc < 0.90:
        return -(lr_unc - 0.75) / 0.15 * 10.0
    elif lr_unc < 1.00:
        return -(10.0 + (lr_unc - 0.90) / 0.10 * 40.0)
    else:
        return -100.0


def _compute_battery_degradation(p_batt: float) -> float:
    """电池衰减: C-rate²。"""
    return (abs(p_batt) / BATTERY_CAPACITY_KWH) ** 2


def _compute_pq_coordination(dev: float, q_margin: float, p_ref: float,
                             v_low: bool, v_high: bool,
                             safety_override_active: bool) -> float:
    """P-Q 协同度奖励 (v2.13 Sigmoid平滑化, v2.14 P-Q互斥)。"""
    if dev <= VOLTAGE_DEADBAND or safety_override_active:
        return 0.0

    SIGMOID_K = 50.0
    Q_THRESHOLD = 0.10
    P_THRESHOLD = 5.0

    w_save = 1.0 / (1.0 + math.exp(-SIGMOID_K * (q_margin - Q_THRESHOLD)))
    w_support = 1.0 - w_save

    if abs(p_ref) < P_THRESHOLD:
        r_lazy = +50.0
    else:
        r_lazy = -5.0

    if v_low and p_ref < 0:
        r_correct = +50.0
    elif v_high and p_ref > 0:
        r_correct = +50.0
    elif v_low and p_ref >= 0:
        r_correct = -30.0
    elif v_high and p_ref <= 0:
        r_correct = -30.0
    else:
        r_correct = 0.0

    return w_save * r_lazy + w_support * r_correct


def _compute_ramp_penalty(p_ref: float, prev_p_batt: float) -> float:
    """功率变化率惩罚。"""
    return abs(p_ref - prev_p_batt) / BATTERY_CAPACITY_KWH


def _compute_voltage_slope(v_avg: float, prev_v_avg: float,
                           base_w6: float) -> tuple[float, float]:
    """电压变化斜率惩罚 (R-05 v2.12 动态权重)。"""
    r_slope = abs(v_avg - prev_v_avg) if prev_v_avg is not None else 0.0
    k_w6 = 2.0
    w6_dynamic = base_w6 * (1.0 + k_w6 * r_slope)
    return w6_dynamic * r_slope, r_slope


def _compute_droop_smoothness(k_droop: float, prev_k_droop: float) -> float:
    """下垂系数平滑惩罚 (v2.8)。"""
    if k_droop == 0.0 and prev_k_droop == 0.0:
        return 0.0
    K_MAX = 50.0
    lambda_smooth = 1.0
    delta_k = abs(k_droop - prev_k_droop)
    return -(delta_k + lambda_smooth * max(0.0, abs(k_droop) - K_MAX))


def _compute_safety_override(active: bool, consecutive: int,
                             ratio: float) -> float:
    """安全覆盖惩罚 (v2.10 新增, v2.14 分层精细化)。"""
    if not active:
        return 0.0
    if consecutive < 10:
        return -1.0 / 15.0
    else:
        consecutive_norm = min(consecutive / 50.0, 1.0)
        return -ratio * consecutive_norm


def _compute_overload_warning(lr_unc: float) -> float:
    """过载预警 (R-02 v2.12)。"""
    if lr_unc > 0.85:
        return -(lr_unc - 0.85) / 0.15
    return 0.0


def _compute_soc_warning(soc_new: float) -> float:
    """SOC 预警 (R-02 v2.12)。"""
    soc_margin_low = soc_new - SOC_CRITICAL
    soc_margin_high = 0.90 - soc_new
    soc_margin = min(soc_margin_low, soc_margin_high)
    if soc_margin < 0.10:
        return -(0.10 - soc_margin) / 0.10
    return 0.0


def _compute_soc_balance(soc_new: float) -> float:
    """SOC 均衡奖励 (R-03 v2.12): 鼓励 SOC 保持在 50% 附近。"""
    return -abs(soc_new - 0.5)


def _compute_shock_readiness(lr_unc: float, load_shed: float,
                             soc_new: float, p_ref: float) -> float:
    """冲击负荷响应奖励 (R-06 v2.12 + v2.13 重构)。"""
    load_rate_delta = abs(lr_unc - 0.5) * 2
    threshold_shock = 10.0
    w_shock = 20.0
    lambda_shock = 5.0
    max_response_time = 15.0

    r_shock = 0.0
    if load_rate_delta > threshold_shock / 100.0:
        r_shock = (w_shock * load_shed / max(LOAD_SHED_MAX_KW, 1e-6)
                   - lambda_shock * 1.0 / max_response_time)

    # v2.13: 预备度奖励
    soc_reserve_target = 0.3
    p_ref_reserve_target = 10.0  # kW
    r_readiness = ((soc_reserve_target - soc_new)
                   + 0.5 * max(0.0, p_ref_reserve_target - abs(p_ref)))
    return r_shock + r_readiness


def _compute_state_improve(dev: float, prev_v_dev: float, p_ref: float) -> float:
    """状态改善率奖励 (R-14.4 v2.13)。

    仅在电压偏差改善且动作方向与改善方向一致时奖励:
    p_ref > 0 (放电→抬电压) 应配合电压从过高回落,
    p_ref < 0 (充电→降电压) 应配合电压从过低回升。
    """
    if dev <= VOLTAGE_DEADBAND or prev_v_dev <= dev:
        return 0.0
    improvement = prev_v_dev - dev
    if p_ref * improvement <= 0:
        return 0.0
    return improvement * (1.0 if p_ref >= 0 else -1.0)


# ═══════════════════════════════════════════════════════════════
# MODE-01: 农网灌溉 (主调度函数)
# ═══════════════════════════════════════════════════════════════

def _reward_agri(r: dict, w: list[float]) -> tuple[float, dict]:
    """SCENE-01 奖励函数 (v2.13 精细化奖励函数设计).

    R = w1*R_pv - α(s)*w2*P_batt_deg - w3*P_overload + w4*R_PQ
        - w5*R_ramp - w6*R_voltage_slope - w7*R_smooth - w8*R_safety_override
        + w9*R_overload_warning + w10*R_soc_warning + w11*R_soc_balance
        + w12*R_shock_response + w13*R_state_improve

    纯函数: 所有状态从 r dict 读取，不访问 self。
    Welford 原始奖励通过 info["welford_raw"] 回传。
    """
    # 提取状态
    v_avg = (r["va"] + r["vb"] + r["vc"]) / 3.0
    dev = abs(v_avg - 1.0)
    v_low = v_avg < (1.0 - VOLTAGE_DEADBAND)
    v_high = v_avg > (1.0 + VOLTAGE_DEADBAND)
    p_ref = r["p_batt"]
    soc_new = r.get("soc_new", r.get("soc", 0.5))
    q_margin = r.get("q_realtime_margin", 1.0)
    safety_override_active = r.get("safety_override_active", False)

    # 子奖励计算
    r_pv = _compute_pv_consumption(r["p_pv_raw"], r["p_load_raw"], p_ref, v_avg)
    alpha = _compute_alpha(soc_new, q_margin, r.get("voltage_violation_count", 0))
    p_overload = _compute_overload(r.get("load_rate_unclamped", r["load_rate"]))
    p_batt_deg = _compute_battery_degradation(p_ref)
    r_pq = _compute_pq_coordination(dev, q_margin, p_ref, v_low, v_high,
                                    safety_override_active)

    load_rate_unc = r.get("load_rate_unclamped", r["load_rate"])
    p_ramp = _compute_ramp_penalty(p_ref, r.get("prev_p_batt", 0.0))
    p_voltage, r_voltage_slope = _compute_voltage_slope(
        v_avg, r.get("prev_v_avg", 1.0), w[5] if len(w) > 5 else 0.5)
    r_smooth = _compute_droop_smoothness(r.get("k_droop", 0.0),
                                         r.get("prev_k_droop", 0.0))
    r_safety = _compute_safety_override(
        safety_override_active,
        r.get("override_consecutive", 0),
        r.get("override_ratio", 0.0))
    r_ov_warn = _compute_overload_warning(load_rate_unc)
    r_soc_warn = _compute_soc_warning(soc_new)
    r_soc_bal = _compute_soc_balance(soc_new)
    r_readiness = _compute_shock_readiness(load_rate_unc, r.get("load_shed", 0.0),
                                           soc_new, p_ref)
    r_improve = _compute_state_improve(dev, r.get("prev_v_dev", 0.0), p_ref)

    # 权重提取
    w4 = w[3] if len(w) > 3 else 0.0
    w5 = w[4] if len(w) > 4 else 0.0
    w7 = w[6] if len(w) > 6 else 0.0
    w8_val = w[7] if len(w) > 7 else 0.0
    w9_val = w[8] if len(w) > 8 else 0.0
    w10_val = w[9] if len(w) > 9 else 0.0
    w11 = w[10] if len(w) > 10 else 0.0
    w12_val = w[11] if len(w) > 11 else 0.0
    w13_val = w[12] if len(w) > 12 else 0.0

    # Welford 原始奖励 (前 4 个分量: r_pv, p_batt_deg, p_overload, r_pq)
    raw_reward = w[0] * r_pv - alpha * w[1] * p_batt_deg - w[2] * p_overload + w4 * r_pq

    # 子项归一化到 [-1, 1]
    r_pv_norm = float(np.clip(r_pv, 0.0, 1.0))
    p_batt_deg_norm = float(np.clip(-p_batt_deg / 0.25, -1.0, 0.0)) if p_batt_deg > 0 else 0.0
    p_overload_norm = float(np.clip(p_overload / 100.0, -1.0, 0.0)) if p_overload < 0 else 0.0
    r_pq_norm = float(np.clip(r_pq / 50.0, -1.0, 1.0))
    p_ramp_norm = float(np.clip(-p_ramp / 0.05, -1.0, 0.0)) if p_ramp > 0 else 0.0

    max_w6 = w[5] * (1.0 + 2.0 * 0.4) if len(w) > 5 else 0.5 * (1.0 + 2.0 * 0.4)
    p_vs_norm = float(np.clip(-p_voltage / max_w6 / 0.4, -1.0, 0.0)) if p_voltage > 0 and max_w6 > 0 else 0.0

    r_smooth_norm = float(np.clip(r_smooth / 150.0, -1.0, 0.0)) if r_smooth < 0 else 0.0
    r_safety_norm = float(np.clip(r_safety / 100.0, -1.0, 0.0)) if r_safety < 0 else 0.0
    r_ov_warn_norm = float(np.clip(r_ov_warn, -1.0, 0.0)) if r_ov_warn < 0 else 0.0
    r_soc_warn_norm = float(np.clip(r_soc_warn, -1.0, 0.0)) if r_soc_warn < 0 else 0.0
    r_soc_bal_norm = float(np.clip(r_soc_bal / 0.5, -1.0, 1.0)) if r_soc_bal != 0 else 0.0
    r_readiness_norm = float(np.clip(r_readiness / 0.5, -1.0, 1.0)) if r_readiness != 0 else 0.0
    r_improve_norm = float(np.clip(r_improve / 0.1, -1.0, 1.0)) if r_improve != 0 else 0.0

    # 加权求和
    total = (w[0] * r_pv_norm
             + w[1] * p_batt_deg_norm
             + w[2] * p_overload_norm
             + w4 * r_pq_norm
             + w5 * p_ramp_norm
             + w[5] * p_vs_norm
             + w7 * r_smooth_norm
             + w8_val * r_safety_norm
             + w9_val * r_ov_warn_norm
             + w10_val * r_soc_warn_norm
             + w11 * r_soc_bal_norm
             + w12_val * r_readiness_norm
             + w13_val * r_improve_norm)

    info = {
        "r_pv_consumption": float(r_pv),
        "p_battery_degradation": float(-p_batt_deg),
        "p_transformer_overload": float(-p_overload),
        "r_pq_coordination": float(r_pq),
        "p_ramp_penalty": float(-p_ramp),
        "r_voltage_slope": float(r_voltage_slope),
        "r_smooth": float(r_smooth),
        "r_safety_override": float(r_safety),
        "r_overload_warning": float(r_ov_warn),
        "r_soc_warning": float(r_soc_warn),
        "r_soc_balance": float(r_soc_bal),
        "r_readiness": float(r_readiness),
        "r_state_improve": float(r_improve),
        "v_avg": float(v_avg),
        "alpha": float(alpha),
        "q_realtime_margin": float(q_margin),
        # Welford 回传: core 用此值更新 Welford 统计量
        "welford_raw": float(raw_reward),
    }
    return float(total), info


# ═══════════════════════════════════════════════════════════════
# MODE-02: 自主套利
# ═══════════════════════════════════════════════════════════════

def _reward_arbitrage(r: dict, w: list[float]) -> tuple[float, dict]:
    """R = w1*R_spread - w2*P_batt_deg - w3*P_overload"""
    price = r.get("current_price", 0.8)
    avg_price = 0.8
    r_spread = r["p_batt"] * (price - avg_price) / (P_BATT_MAX_KW * 0.4)
    r_spread = float(np.clip(r_spread, -1.0, 1.0))

    c_rate = abs(r["p_batt"]) / BATTERY_CAPACITY_KWH
    p_batt_deg = c_rate ** 2

    lr_unc = r.get("load_rate_unclamped", r["load_rate"])
    overload_t = max(0.0, (lr_unc - 0.75) / 0.25)
    p_overload = -0.3095 * overload_t ** 2 + 0.026 * overload_t
    w3 = w[2] if len(w) > 2 else 0.0

    total = w[0] * r_spread - w[1] * p_batt_deg - w3 * p_overload
    info = {
        "r_price_spread": float(r_spread),
        "p_battery_degradation": float(-p_batt_deg),
        "p_transformer_overload": float(-p_overload) if w3 > 0 else 0.0,
        "welford_raw": None,
    }
    return float(total), info


# ═══════════════════════════════════════════════════════════════
# MODE-03: 需量控制
# ═══════════════════════════════════════════════════════════════

def _reward_demand(r: dict, w: list[float]) -> tuple[float, dict]:
    """R = w1*R_demand_avoid - w2*P_comfort_loss"""
    d_actual = r.get("current_demand", 0.0)
    d_baseline = CONTRACT_DEMAND_KW
    r_demand_avoid = max(0.0, d_baseline - d_actual) / d_baseline

    p_comfort = r["load_shed"] / LOAD_SHED_MAX_KW

    total = w[0] * r_demand_avoid - w[1] * p_comfort
    info = {
        "r_demand_penalty_avoidance": float(r_demand_avoid),
        "p_comfort_loss": float(-p_comfort),
        "welford_raw": None,
    }
    return float(total), info


# ═══════════════════════════════════════════════════════════════
# MODE-04: 虚拟电厂
# ═══════════════════════════════════════════════════════════════

def _reward_vpp(r: dict, w: list[float]) -> tuple[float, dict]:
    """R = w1*R_ancillary + w2*R_accuracy - w3*P_deadline"""
    dp = r.get("dispatch_p_set", 0.0)
    prev_p_batt = r.get("prev_p_batt", 0.0)

    # 辅助服务收益 (容量 + 里程)
    r_ancillary = (abs(dp) * 0.1 + abs(r["p_batt"] - prev_p_batt) * 0.05) / 50.0
    r_ancillary = float(np.clip(r_ancillary, 0.0, 1.0))

    # 响应精度
    if abs(dp) > 1e-6:
        r_accuracy = max(0.0, 1.0 - abs(r["p_batt"] - dp) / abs(dp))
    else:
        r_accuracy = 1.0

    # 延迟惩罚
    p_deadline = abs(r["p_batt"] - prev_p_batt) / (50.0 * DT_HOURS)
    p_deadline = float(min(p_deadline, 1.0))

    total = w[0] * r_ancillary + w[1] * r_accuracy - w[2] * p_deadline
    info = {
        "r_ancillary_service": float(r_ancillary),
        "r_response_accuracy": float(r_accuracy),
        "p_deadline_deviation": float(-p_deadline),
        "welford_raw": None,
    }
    return float(total), info


# ═══════════════════════════════════════════════════════════════
# MODE-05: 极致绿色
# ═══════════════════════════════════════════════════════════════

def _reward_green(r: dict, w: list[float]) -> tuple[float, dict]:
    """R = w1*R_green + w2*R_carbon"""
    p_pv = r["p_pv_raw"]
    p_load_eff = r["p_load_eff"]
    e_total = max(p_load_eff, 1e-6)
    e_green = min(p_pv, p_load_eff) + max(0.0, -r["p_batt"])
    r_green = min(e_green / e_total, 1.0)

    grid_buy = max(0.0, r["grid_power"])
    c_baseline = p_load_eff * GRID_EMISSION_FACTOR * DT_HOURS
    c_actual = grid_buy * GRID_EMISSION_FACTOR * DT_HOURS
    r_carbon = max(0.0, (c_baseline - c_actual) / max(c_baseline, 1e-6))

    total = w[0] * r_green + w[1] * r_carbon
    info = {
        "r_green_consumption": float(r_green),
        "r_carbon_reduction": float(r_carbon),
        "welford_raw": None,
    }
    return float(total), info
