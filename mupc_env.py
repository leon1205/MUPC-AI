"""
MUPC 全状态强化学习环境 — 48/49 维观测, 4 维动作, 5 种场景奖励。

对齐 MUPC AI 引擎 PRD v2.2:
  - 状态空间: 21 字段序列化为 48/49 维向量
  - 动作空间: [p_batt, q_batt, load_shedding, pv_limit] (4 维)
  - 5 种预设场景 MODE-01~05 各独立奖励函数
  - 三相电压简化仿真 (Q-V 耦合)
  - 5 条动作约束规则 (ActionValidator)
"""

import math
import sys
from typing import Any, Optional

import numpy as np

# ── Gymnasium 降级检测 ────────────────────────────────────────

_GYM_AVAILABLE = False
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    _GYM_AVAILABLE = True
except ImportError:
    from _gym_stub import Env as _GymStubEnv, Box

from action_validator import ActionValidator


# ═══════════════════════════════════════════════════════════════
# 物理常量 (SAFETY: 以下常量涉及硬件安全边界, 修改前请评审)
# ═══════════════════════════════════════════════════════════════

TRANSFORMER_KVA = 500.0         # 变压器额定容量 (kVA)
BATTERY_CAPACITY_KWH = 200.0   # 电池容量 (kWh)
P_BATT_MAX_KW = 500.0           # 最大充放电功率 (kW)
Q_BATT_MAX_KVAR = 300.0         # 最大无功输出 (kVar)
LOAD_SHED_MAX_KW = 500.0        # 最大切负荷 (kW)
PV_ARRAY_KW = 200.0             # 光伏容量 (kW)
LOAD_PEAK_KW = 400.0            # 负荷峰值 (kW)

SOC_MIN = 0.10                  # SAFETY: SOC 下限硬约束
SOC_MAX = 0.90                  # SAFETY: SOC 上限硬约束
OVERLOAD_THRESHOLD = 0.85       # 过载阈值
DT_HOURS = 0.25                 # 时间步长 (15 分钟)
LOAD_PF = 0.90                  # 负荷功率因数 cosφ

CONTRACT_DEMAND_KW = 300.0      # 合同需量 (kW)
GRID_EMISSION_FACTOR = 0.581   # kg CO2/kWh
EPISODE_LENGTH = 96             # 1 天 = 96 步 × 15 分钟


# ═══════════════════════════════════════════════════════════════
# 奖励权重映射
# ═══════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS: dict[str, list[float]] = {
    "MODE-01": [1.0, 0.5, 2.0],     # w1(光伏消纳), w2(电池), w3(过载)
    "MODE-02": [1.0, 1.0, 2.0],      # w1(价差), w2(电池), w3(过载)
    "MODE-03": [1.0, 0.5],           # w1(需量减免), w2(舒适度)
    "MODE-04": [1.0, 2.0, 1.0],      # w1(辅助收益), w2(响应精度), w3(延迟)
    "MODE-05": [1.0, 1.0],           # w1(绿电), w2(碳减排)
}

MODE_ID_MAP: dict[str, float] = {
    "MODE-01": 0.0, "MODE-02": 0.25, "MODE-03": 0.5,
    "MODE-04": 0.75, "MODE-05": 1.0,
}

ALL_MODES = ["MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"]


# ═══════════════════════════════════════════════════════════════
# 电压仿真
# ═══════════════════════════════════════════════════════════════

class VoltageSimulator:
    """三相电压简化线路模型 (Q-V 耦合)。"""

    K_P = 0.05            # 有功灵敏度 (p.u. / 500kW)
    K_Q = 0.03            # 无功灵敏度 (p.u. / 300kVar)
    S_BASE = 500.0        # kVA
    V_MIN = 0.85
    V_MAX = 1.15
    NOISE_STD = 0.005     # 测量噪声
    IMBALANCE = 0.003     # 三相不平衡度

    def step(self, p_net: float, q_batt: float,
             prev_va: float, prev_vb: float, prev_vc: float
             ) -> tuple[float, float, float]:
        """一步电压更新。

        Args:
            p_net: 净有功 = P_pv_eff - P_load_eff + P_batt (kW)
            q_batt: 无功功率 (kVar)
            prev_v*: 上一周期三相电压 (p.u.)

        Returns:
            (va, vb, vc) 三相电压 (p.u.)
        """
        dv = (self.K_P * p_net + self.K_Q * q_batt) / self.S_BASE
        va = prev_va + dv + np.random.normal(0, self.NOISE_STD)
        vb = prev_vb + dv + np.random.normal(0, self.NOISE_STD) + self.IMBALANCE
        vc = prev_vc + dv + np.random.normal(0, self.NOISE_STD) - self.IMBALANCE
        return (
            float(np.clip(va, self.V_MIN, self.V_MAX)),
            float(np.clip(vb, self.V_MIN, self.V_MAX)),
            float(np.clip(vc, self.V_MIN, self.V_MAX)),
        )


# ═══════════════════════════════════════════════════════════════
# 环境
# ═══════════════════════════════════════════════════════════════

class MupcEnv(gym.Env if _GYM_AVAILABLE else _GymStubEnv):
    """MUPC 全状态 RL 环境。

    Anys:
        - 观测空间: Box(48,) 或 Box(49,) (多模式)
        - 动作空间: Box(4,) ∈ [-1,1] 前2维, [0,1] 后2维
    """

    metadata = {"render_modes": []}

    # ── 初始化 ────────────────────────────────────────────

    def __init__(self, data: dict, mode: str = "all",
                 lstm_predictor: Any = None,
                 reward_weights: dict[str, list[float]] | None = None):
        """
        Args:
            data: SmartDSLoader 返回的 data dict
            mode: "all" (多模式) 或 "MODE-01"~"MODE-05" (单模式)
            lstm_predictor: LSTM 模型 (有 predict(step_idx)→(30,) 接口) 或 None→Oracle
            reward_weights: 自定义权重, e.g. {"MODE-01": [1.5, 0.3, 3.0]}
        """
        self._data = data
        self._mode = mode
        self._data_len = data["n_steps"]
        self._weights = {**DEFAULT_WEIGHTS, **(reward_weights or {})}

        # LSTM 预测器 / Oracle
        if lstm_predictor is not None:
            self._predictor = lstm_predictor
        else:
            from lstm_model import OraclePredictor
            self._predictor = OraclePredictor(data)

        # 动作校验器
        self._validator = ActionValidator()

        # 电压仿真器
        self._voltage_sim = VoltageSimulator()

        # 观测/动作空间
        obs_dim = 48 if mode != "all" else 49
        low_obs = np.full(obs_dim, -10.0, dtype=np.float32)
        high_obs = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = Box(low_obs, high_obs, dtype=np.float32)

        # 动作: [p_batt_norm, q_batt_norm, load_shed_norm, pv_limit_norm]
        low_act = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high_act = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = Box(low_act, high_act, dtype=np.float32)

        # 内部状态
        self._soc: float = 0.5
        self._va: float = 1.0
        self._vb: float = 1.0
        self._vc: float = 1.0
        self._battery_power_prev: float = 0.0
        self._grid_power: float = 0.0
        self._load_rate: float = 0.5
        self._current_demand: float = 200.0
        self._peak_demand: float = 200.0
        self._step_idx: int = 0
        self._episode_start: int = 0
        self._current_mode: str = "MODE-01"
        self._prev_p_batt: float = 0.0
        self._prev_q_batt: float = 0.0

    # ── 模式管理 ────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """运行时切换运行场景。"""
        if mode in ALL_MODES:
            self._current_mode = mode
        else:
            raise ValueError(f"无效模式: {mode}, 有效值: {ALL_MODES}")

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def mode(self) -> str:
        return self._mode

    # ── Gymnasium 接口 ────────────────────────────────────

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 随机 SOC (避开边界)
        self._soc = float(np.random.uniform(0.3, 0.7))
        self._va = 1.0
        self._vb = 1.0
        self._vc = 1.0
        self._battery_power_prev = 0.0
        self._grid_power = 0.0
        self._load_rate = 0.5
        self._current_demand = 200.0
        self._peak_demand = 200.0
        self._prev_p_batt = 0.0
        self._prev_q_batt = 0.0

        # 随机起始索引 (保证至少还有 EPISODE_LENGTH 步)
        max_start = self._data_len - EPISODE_LENGTH - 16  # 16 为预测缓冲区
        self._episode_start = np.random.randint(0, max(1, max_start))
        self._step_idx = self._episode_start

        # 设置当前场景
        if self._mode == "all":
            self._current_mode = np.random.choice(ALL_MODES)
        else:
            self._current_mode = self._mode  # 单模式: 固定为指定场景

        # 重置校验器
        self._validator.reset()

        obs = self._build_observation()
        info = {"mode": self._current_mode}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真。

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # 1. 动作约束校验
        dispatch_p = self._data["dispatch_p_set"][self._step_idx]
        if abs(dispatch_p) < 1e-6:
            dispatch_p_use = None
        else:
            dispatch_p_use = float(dispatch_p)
        clamped, violated, violations = self._validator.validate(action, dispatch_p_use)

        # 2. 反归一化动作到物理值
        p_batt = clamped[0] * P_BATT_MAX_KW
        q_batt = clamped[1] * Q_BATT_MAX_KVAR
        load_shed = clamped[2] * LOAD_SHED_MAX_KW
        pv_limit = clamped[3]            # 已在 [0,1]

        # 3. 有效负荷与光伏
        p_load_raw = float(self._data["load_power"][self._step_idx])
        p_load_eff = max(0.0, p_load_raw - load_shed)
        p_pv_raw = float(self._data["pv_power"][self._step_idx])
        p_pv_eff = p_pv_raw * pv_limit

        # 4. SOC 更新 (SAFETY: hard clamp)
        soc_raw = self._soc + (-p_batt * DT_HOURS) / BATTERY_CAPACITY_KWH
        soc_new = float(np.clip(soc_raw, SOC_MIN, SOC_MAX))
        soc_clipped = abs(soc_raw - soc_new) > 1e-9

        # 5. 电网交换功率
        grid_power = p_load_eff - p_pv_eff + p_batt

        # 6. 变压器负载率 + 过载硬约束 (SAFETY)
        q_load = p_load_eff * math.tan(math.acos(LOAD_PF))
        s_transformer = math.sqrt(grid_power ** 2 + (q_load - q_batt) ** 2)
        load_rate = s_transformer / TRANSFORMER_KVA

        # SAFETY: 变压器过载硬约束, 类似 SOC clamp
        # 当 load_rate > 1.0 时, 自动限制 p_batt 使负载率不超过 100%
        if load_rate > 1.0:
            q_net = q_load - q_batt
            s_max = TRANSFORMER_KVA
            # 计算目标有功功率 (保持符号不变)
            p_target = math.sqrt(max(0, s_max**2 - q_net**2))
            if grid_power < 0:
                p_target = -p_target
            p_batt_clamped = p_target - (p_load_eff - p_pv_eff)
            p_batt = float(np.clip(p_batt_clamped, -P_BATT_MAX_KW, P_BATT_MAX_KW))
            grid_power = p_load_eff - p_pv_eff + p_batt
            s_transformer = math.sqrt(grid_power**2 + q_net**2)
            load_rate = s_transformer / TRANSFORMER_KVA

        # 7. 电压更新
        p_net = p_pv_eff - p_load_eff + p_batt
        va, vb, vc = self._voltage_sim.step(
            p_net, q_batt, self._va, self._vb, self._vc,
        )

        # 8. 需量更新 (1 小时滑动窗口)
        window = 4
        demand_start = max(0, self._step_idx - window + 1)
        demand_slice = self._data["load_power"][demand_start:self._step_idx + 1]
        current_demand = max(float(np.mean(demand_slice)), CONTRACT_DEMAND_KW * 0.3)
        peak_demand = max(self._peak_demand, current_demand)

        # 9. 更新内部状态
        self._soc = soc_new
        self._va = va
        self._vb = vb
        self._vc = vc
        self._battery_power_prev = p_batt
        self._grid_power = grid_power
        self._load_rate = load_rate
        self._current_demand = current_demand
        self._peak_demand = peak_demand
        self._prev_p_batt = p_batt
        self._prev_q_batt = q_batt

        # 10. 奖励计算
        reward, reward_info = self._compute_reward(
            p_batt=p_batt, q_batt=q_batt, load_shed=load_shed,
            pv_limit=pv_limit, p_pv_raw=p_pv_raw, p_load_raw=p_load_raw,
            p_load_eff=p_load_eff, grid_power=grid_power,
            load_rate=load_rate, soc_new=soc_new, soc_clipped=soc_clipped,
        )

        # 11. 推进时间
        self._step_idx += 1
        terminated = (self._step_idx - self._episode_start) >= EPISODE_LENGTH
        truncated = self._step_idx >= self._data_len - 16

        # 12. 构建 info
        info = {
            "mode": self._current_mode,
            "soc": self._soc,
            "load_rate": self._load_rate,
            "p_batt": p_batt,
            "q_batt": q_batt,
            "load_shedding": load_shed,
            "pv_limit": pv_limit,
            "grid_power": grid_power,
            "va": va, "vb": vb, "vc": vc,
            "current_demand": current_demand,
            "peak_demand": peak_demand,
            "soc_clipped": soc_clipped,
            "constraint_violated": violated,
            "violations": str(violations) if violations else "",
            **reward_info,
        }
        if terminated or truncated:
            info["terminal_observation"] = self._build_observation()

        obs = self._build_observation()
        return obs, float(reward), terminated, truncated, info

    # ── 观测构建 ────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """构建 48 维观测向量 (多模式追加 mode_id 为 49 维)。

        布局 (对齐 MUPC AI 引擎设计文档 to_input_vector):
          [0..9]   D1: 9 标量 (battery_soc, pv, load, grid, trans_load,
                              battery_pwr, va, vb, vc)
          [9..24]  D2: pv_forecast (15 维)
          [24..39] D2: load_forecast (15 维)
          [39..42] D3: current_price, next_price, tariff_id
          [42..45] D4: current_demand, contract_demand, peak_demand
          [45..47] D5: solar_irradiance, temperature
          [47]     D6: dispatch_p_set (None → 0.0)
          [48]     (可选) mode_id
        """
        obs_dim = 48 if self._mode != "all" else 49
        obs = np.zeros(obs_dim, dtype=np.float32)
        params = self._data.get("norm_params", {})

        # ── D1 [0..9] ──
        obs[0] = self._soc
        obs[1] = self._data["pv_power"][self._step_idx]
        obs[2] = self._data["load_power"][self._step_idx]
        obs[3] = self._grid_power
        obs[4] = self._load_rate
        obs[5] = self._battery_power_prev
        obs[6] = self._va
        obs[7] = self._vb
        obs[8] = self._vc

        # ── D2 [9..39] 预测 ──
        forecast = self._predictor.predict(self._step_idx)  # (30,)
        obs[9:24] = forecast[:15]      # pv_forecast
        obs[24:39] = forecast[15:30]   # load_forecast

        # ── D3 [39..42] 电价 ──
        obs[39] = self._data["current_electricity_price"][self._step_idx]
        obs[40] = self._data["next_period_price"][self._step_idx]
        obs[41] = self._data["price_tariff_id"][self._step_idx]

        # ── D4 [42..45] 需量 ──
        obs[42] = self._current_demand
        obs[43] = CONTRACT_DEMAND_KW
        obs[44] = self._peak_demand

        # ── D5 [45..47] 气象 ──
        obs[45] = self._data["solar_irradiance"][self._step_idx]
        obs[46] = self._data["temperature"][self._step_idx]

        # ── D6 [47] 调度 ──
        dp = self._data["dispatch_p_set"][self._step_idx]
        obs[47] = dp if abs(dp) > 1e-6 else 0.0

        # ── mode_id [48] (可选) ──
        if self._mode == "all":
            obs[48] = MODE_ID_MAP[self._current_mode]

        # ── 归一化 ──
        obs = self._normalize_obs(obs, params)
        return obs.astype(np.float32)

    def _normalize_obs(self, obs: np.ndarray, params: dict) -> np.ndarray:
        """应用 MinMax 归一化。"""
        out = obs.copy()
        # D1
        out[0] = obs[0]  # SOC: identity
        out[1] = self._minmax(obs[1], 0.0, PV_ARRAY_KW)
        out[2] = self._minmax(obs[2], 0.0, LOAD_PEAK_KW)
        out[3] = self._minmax(obs[3], -500.0, 500.0)
        out[4] = obs[4]  # transformer_load: identity
        out[5] = self._minmax(obs[5], -500.0, 500.0)
        out[6] = self._minmax(obs[6], 0.85, 1.15)
        out[7] = self._minmax(obs[7], 0.85, 1.15)
        out[8] = self._minmax(obs[8], 0.85, 1.15)
        # D2: forecast
        out[9:24] = self._minmax(obs[9:24], 0.0, PV_ARRAY_KW)
        out[24:39] = self._minmax(obs[24:39], 0.0, LOAD_PEAK_KW)
        # D3
        out[39] = self._minmax(obs[39], 0.0, 1.5)
        out[40] = self._minmax(obs[40], 0.0, 1.5)
        out[41] = self._minmax(obs[41], 0.0, 3.0)
        # D4
        out[42] = self._minmax(obs[42], 0.0, 500.0)
        out[43] = self._minmax(obs[43], 0.0, 500.0)
        out[44] = self._minmax(obs[44], 0.0, 500.0)
        # D5
        out[45] = self._minmax(obs[45], 0.0, 1500.0)
        out[46] = self._minmax(obs[46], -20.0, 60.0)
        # D6
        out[47] = self._minmax(obs[47], -500.0, 500.0)
        # mode_id: identity
        return out

    @staticmethod
    def _minmax(x, lo, hi):
        """MinMax 归一化, 支持标量和数组。"""
        clipped = np.clip(x, lo, hi)
        result = (clipped - lo) / (hi - lo + 1e-9)
        if np.isscalar(x):
            return float(result)
        return result.astype(np.float32)

    @staticmethod
    def _minmax_scalar(x, lo, hi) -> float:
        """标量 MinMax 归一化。"""
        return float((np.clip(float(x), lo, hi) - lo) / (hi - lo + 1e-9))

    # ── 奖励函数 ────────────────────────────────────────

    def _compute_reward(self, **r) -> tuple[float, dict]:
        """根据当前场景选择奖励函数并计算。"""
        mode = self._current_mode
        w = self._weights.get(mode, DEFAULT_WEIGHTS[mode])
        info = {}

        if mode == "MODE-01":
            total, info = self._reward_agri(r, w)
        elif mode == "MODE-02":
            total, info = self._reward_arbitrage(r, w)
        elif mode == "MODE-03":
            total, info = self._reward_demand(r, w)
        elif mode == "MODE-04":
            total, info = self._reward_vpp(r, w)
        elif mode == "MODE-05":
            total, info = self._reward_green(r, w)
        else:
            total = 0.0

        return total, info

    # ── MODE-01: 农网灌溉 ──────────────────────────────

    def _reward_agri(self, r: dict, w: list[float]) -> tuple[float, dict]:
        """R = w1*R_pv - w2*P_batt_deg - w3*P_overload"""
        # 光伏消纳率
        pv_total = max(r["p_pv_raw"], 1e-6)
        pv_self = min(r["p_pv_raw"], r["p_load_raw"]) + max(0.0, -r["p_batt"])
        r_pv = min(pv_self / pv_total, 1.0)

        # 电池衰减: |ΔSOC|
        delta_soc = abs(self._soc - r["soc_new"])
        p_batt_deg = delta_soc

        # 过载惩罚
        p_overload = max(0.0, r["load_rate"] - 1.0)

        total = w[0] * r_pv - w[1] * p_batt_deg - w[2] * p_overload
        info = {
            "r_pv_consumption": float(r_pv),
            "p_battery_degradation": float(-p_batt_deg),
            "p_transformer_overload": float(-p_overload),
        }
        return float(total), info

    # ── MODE-02: 自主套利 ──────────────────────────────

    def _reward_arbitrage(self, r: dict, w: list[float]) -> tuple[float, dict]:
        """R = w1*R_spread - w2*P_batt_deg - w3*P_overload"""
        price = float(self._data["current_electricity_price"][self._step_idx])
        avg_price = 0.8
        r_spread = r["p_batt"] * (price - avg_price) / (P_BATT_MAX_KW * 0.4)
        r_spread = float(np.clip(r_spread, -1.0, 1.0))

        delta_soc = abs(self._soc - r["soc_new"])
        p_batt_deg = delta_soc

        p_overload = max(0.0, r["load_rate"] - 1.0)  # w3: 过载惩罚
        w3 = w[2] if len(w) > 2 else 0.0

        total = w[0] * r_spread - w[1] * p_batt_deg - w3 * p_overload
        info = {
            "r_price_spread": float(r_spread),
            "p_battery_degradation": float(-p_batt_deg),
            "p_transformer_overload": float(-p_overload) if w3 > 0 else 0.0,
        }
        return float(total), info

    # ── MODE-03: 需量控制 ──────────────────────────────

    def _reward_demand(self, r: dict, w: list[float]) -> tuple[float, dict]:
        """R = w1*R_demand_avoid - w2*P_comfort_loss"""
        d_actual = self._current_demand
        d_baseline = CONTRACT_DEMAND_KW
        r_demand_avoid = max(0.0, d_baseline - d_actual) / d_baseline

        p_comfort = r["load_shed"] / LOAD_SHED_MAX_KW

        total = w[0] * r_demand_avoid - w[1] * p_comfort
        info = {
            "r_demand_penalty_avoidance": float(r_demand_avoid),
            "p_comfort_loss": float(-p_comfort),
        }
        return float(total), info

    # ── MODE-04: 虚拟电厂 ──────────────────────────────

    def _reward_vpp(self, r: dict, w: list[float]) -> tuple[float, dict]:
        """R = w1*R_ancillary + w2*R_accuracy - w3*P_deadline"""
        dp = self._data["dispatch_p_set"][self._step_idx]

        # 辅助服务收益 (容量 + 里程)
        r_ancillary = (abs(dp) * 0.1 + abs(r["p_batt"] - self._prev_p_batt) * 0.05) / 50.0
        r_ancillary = float(np.clip(r_ancillary, 0.0, 1.0))

        # 响应精度
        if abs(dp) > 1e-6:
            r_accuracy = max(0.0, 1.0 - abs(r["p_batt"] - dp) / abs(dp))
        else:
            r_accuracy = 1.0

        # 延迟惩罚 (功率变化率代理)
        p_deadline = abs(r["p_batt"] - self._prev_p_batt) / (50.0 * DT_HOURS)
        p_deadline = float(min(p_deadline, 1.0))

        total = w[0] * r_ancillary + w[1] * r_accuracy - w[2] * p_deadline
        info = {
            "r_ancillary_service": float(r_ancillary),
            "r_response_accuracy": float(r_accuracy),
            "p_deadline_deviation": float(-p_deadline),
        }
        return float(total), info

    # ── MODE-05: 极致绿色 ──────────────────────────────

    def _reward_green(self, r: dict, w: list[float]) -> tuple[float, dict]:
        """R = w1*R_green + w2*R_carbon"""
        p_pv = r["p_pv_raw"]
        p_load_eff = r["p_load_eff"]
        e_total = max(p_load_eff, 1e-6)
        e_green = min(p_pv, p_load_eff) + max(0.0, -r["p_batt"])
        r_green = min(e_green / e_total, 1.0)

        # 碳排放: 仅电网买电产生
        grid_buy = max(0.0, r["grid_power"])
        c_baseline = p_load_eff * GRID_EMISSION_FACTOR * DT_HOURS
        c_actual = grid_buy * GRID_EMISSION_FACTOR * DT_HOURS
        r_carbon = max(0.0, (c_baseline - c_actual) / max(c_baseline, 1e-6))

        total = w[0] * r_green + w[1] * r_carbon
        info = {
            "r_green_consumption": float(r_green),
            "r_carbon_reduction": float(r_carbon),
        }
        return float(total), info

    # ── VPP 调度合成 ────────────────────────────────────

    def _generate_vpp_dispatch(self) -> float:
        """VPP 模式: 随机生成调度指令 (每 96 步 20% 概率触发)。"""
        if self._step_idx % 96 == 0:
            if np.random.random() < 0.2:
                sign = 1 if np.random.random() > 0.5 else -1
                mag = np.random.uniform(50, 200)
                return float(sign * mag)
        return float(self._data["dispatch_p_set"][self._step_idx])


# ═══════════════════════════════════════════════════════════════
# 自测入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_loader import SmartDSLoader

    print("=" * 56)
    print("  MUPC 环境自测")
    print("=" * 56)

    loader = SmartDSLoader()
    data = loader.load_all()
    train, val = loader.split(data)

    # 测试单模式
    print("\n── 单模式测试 (MODE-01) ──")
    env = MupcEnv(train, mode="MODE-01")
    obs, info = env.reset()
    print(f"  观测形状: {obs.shape}")
    print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  动作空间: {env.action_space}")
    print(f"  初始模式: {info['mode']}")
    print(f"  初始 SOC: {env._soc:.3f}")

    total_reward = 0.0
    overload_count = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["load_rate"] > OVERLOAD_THRESHOLD:
            overload_count += 1
        if terminated or truncated:
            break

    print(f"  随机动作 20 步: 累计奖励={total_reward:.3f}, "
          f"过载次数={overload_count}, 最终SOC={info['soc']:.3f}")

    # 测试多模式
    print("\n── 多模式测试 (all) ──")
    env2 = MupcEnv(train, mode="all")
    obs2, info2 = env2.reset()
    print(f"  观测形状: {obs2.shape}  (应为 49 维)")
    print(f"  初始模式: {info2['mode']}")

    modes_seen = set()
    for i in range(500):
        action = env2.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env2.step(action)
        modes_seen.add(info2["mode"])
        if terminated or truncated:
            env2.reset()
    print(f"  500 步覆盖模式: {sorted(modes_seen)}")

    print(f"\n[PASS] mupc_env.py 自测通过")
