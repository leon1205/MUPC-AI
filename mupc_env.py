"""
MUPC 全状态强化学习环境 — 58/59 维观测, 3 维动作, 5 种场景奖励。

对齐 MUPC AI 引擎 PRD v2.6:
  - 状态空间: 21 字段序列化为 58/59 维向量
  - 动作空间: [p_batt, load_shedding, pv_limit] (3 维) — Q 控制交由实时电压调节器闭环
  - 5 种预设场景 MODE-01~05 各独立奖励函数
  - 三相电压简化仿真 (Q-V 耦合，Q 由实时模块根据电压闭环给出)
  - 电压死区 (±5%) + 功率变化率/电压斜率惩罚，保护电池免受高频微循环损耗
  - 4 条动作约束规则 (ActionValidator): ACT-01(ΔP≤50kW), ACT-03(功率圆),
    ACT-04(pv_limit), ACT-05(调度约束) — Q 相关约束由实时控制处理
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


def _cfg():
    """懒加载配置（支持无 --config 时的硬编码回退）。"""
    from config.config_manager import get_config
    return get_config()


# ═══════════════════════════════════════════════════════════════
# 物理常量 (SAFETY: 以下常量涉及硬件安全边界, 修改前请评审)
# 优先从配置文件读取，未配置则使用默认值（向后兼容）
# ═══════════════════════════════════════════════════════════════

_c = _cfg()
TRANSFORMER_KVA = _c.physical.transformer_kva
BATTERY_CAPACITY_KWH = _c.physical.battery_capacity_kwh
P_BATT_MAX_KW = _c.physical.p_batt_max_kw
Q_BATT_MAX_KVAR = _c.physical.q_batt_max_kvar
LOAD_SHED_MAX_KW = _c.physical.load_shed_max_kw
PV_ARRAY_KW = _c.physical.pv_array_kw
LOAD_PEAK_KW = _c.physical.load_peak_kw

SOC_MIN = _c.safety.soc_min
SOC_MAX = _c.safety.soc_max
OVERLOAD_THRESHOLD = _c.safety.overload_threshold
DT_HOURS = _c.time.dt_hours
LOAD_PF = _c.time.load_pf

CONTRACT_DEMAND_KW = _c.contract.contract_demand_kw
GRID_EMISSION_FACTOR = _c.contract.grid_emission_factor
EPISODE_LENGTH = _c.time.episode_length

# ── v2.5 奖励阈值配置 ─────────────────────────────────────────
# 对齐 MUPC AI 引擎 PRD v2.5 RewardThresholdConfig

VOLTAGE_DEADBAND = _c.reward_thresholds.voltage_deadband
Q_MARGIN_THRESHOLD = _c.reward_thresholds.q_margin_threshold
VOLTAGE_HIGH_LIMIT = _c.reward_thresholds.voltage_high_limit
SOC_CRITICAL = _c.reward_thresholds.soc_critical
VOLTAGE_PENALTY_HIGH = _c.reward_thresholds.voltage_penalty_high
VOLTAGE_PENALTY_LOW = _c.reward_thresholds.voltage_penalty_low

# ── Q_batt 电压环控制增益 (v2.4 分层控制) ────────────────────────
# Q_batt 由实时电压调节器闭环控制，基于电压偏差计算
# K_Q: 无功-电压灵敏度系数 (kVar/p.u.)
K_Q = _c.q_control.k_q


# ═══════════════════════════════════════════════════════════════
# 奖励权重映射
# ═══════════════════════════════════════════════════════════════

def _default_weights() -> dict[str, list[float]]:
    """动态获取默认奖励权重（始终读取当前全局配置）。"""
    c = _cfg()
    return {
        "MODE-01": c.reward_weights.MODE_01,
        "MODE-02": c.reward_weights.MODE_02,
        "MODE-03": c.reward_weights.MODE_03,
        "MODE-04": c.reward_weights.MODE_04,
        "MODE-05": c.reward_weights.MODE_05,
    }


DEFAULT_WEIGHTS: dict[str, list[float]] = _default_weights()

MODE_ID_MAP: dict[str, float] = {
    "MODE-01": 0.0, "MODE-02": 0.25, "MODE-03": 0.5,
    "MODE-04": 0.75, "MODE-05": 1.0,
}

ALL_MODES = ["MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"]


# ═══════════════════════════════════════════════════════════════
# 电压仿真
# ═══════════════════════════════════════════════════════════════

class VoltageSimulator:
    """三相电压简化线路模型 (Q-V 耦合)，参数由配置文件决定。"""

    def __init__(self, k_p: float = 0.05, k_q: float = 0.03, s_base: float = 500.0,
                 v_min: float = 0.85, v_max: float = 1.15,
                 noise_std: float = 0.005, imbalance: float = 0.003):
        self.K_P = k_p
        self.K_Q = k_q
        self.S_BASE = s_base
        self.V_MIN = v_min
        self.V_MAX = v_max
        self.NOISE_STD = noise_std
        self.IMBALANCE = imbalance

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
    """MUPC 全状态 RL 环境 (v2.6 分层控制架构).

    动作空间 3 维: [p_batt, load_shedding, pv_limit]
    Q_batt 由实时电压调节器闭环给出，不经过 RL 动作输出。
    观测空间: Box(58,) 或 Box(59,) (多模式)
    """

    metadata = {"render_modes": []}

    # ── 初始化 ────────────────────────────────────────────

    def __init__(self, data: dict, mode: str = "all",
                 lstm_predictor: Any = None,
                 reward_weights: dict[str, list[float]] | None = None,
                 use_grid2op: bool = True,
                 grid2op_backend: str = "lightsim",
                 config: "MupcConfig | None" = None):
        """
        Args:
            data: SmartDSLoader 返回的 data dict
            mode: "all" (多模式) 或 "MODE-01"~"MODE-05" (单模式)
            lstm_predictor: LSTM 模型 (有 predict(step_idx)→(30,) 接口) 或 None→Oracle
            reward_weights: 自定义权重, e.g. {"MODE-01": [1.5, 0.3, 3.0]}
            use_grid2op: True=使用 Grid2Op 电压仿真, False=降级到 VoltageSimulator
            grid2op_backend: "lightsim" (C++ 加速) 或 "pandapower" (Python)
            config: 配置对象，未指定时从全局配置读取
        """
        from config.config_manager import get_config
        self._cfg = config or get_config()

        self._data = data
        self._mode = mode
        self._data_len = data["n_steps"]

        # 奖励权重：命令行自定义 > 配置文件 > 默认值
        cfg_weights = {
            "MODE-01": self._cfg.reward_weights.MODE_01,
            "MODE-02": self._cfg.reward_weights.MODE_02,
            "MODE-03": self._cfg.reward_weights.MODE_03,
            "MODE-04": self._cfg.reward_weights.MODE_04,
            "MODE-05": self._cfg.reward_weights.MODE_05,
        }
        self._weights = {**cfg_weights, **(reward_weights or {})}

        # LSTM 预测器 / Oracle
        if lstm_predictor is not None:
            self._predictor = lstm_predictor
        else:
            from lstm_model import OraclePredictor
            self._predictor = OraclePredictor(data)

        # 动作校验器（传入配置的动作约束参数）
        vs = self._cfg.action_constraints
        self._validator = ActionValidator(
            p_batt_max=vs.p_batt_max,
            s_max=vs.s_max,
            load_shed_max=vs.load_shed_max,
            delta_p_max=vs.delta_p_max,
        )

        # 电压仿真器（Grid2Op 不可用时降级使用）
        vc = self._cfg.voltage_simulator
        self._voltage_sim = VoltageSimulator(
            k_p=vc.k_p, k_q=vc.k_q, s_base=vc.s_base,
            v_min=vc.v_min, v_max=vc.v_max,
            noise_std=vc.noise_std, imbalance=vc.imbalance,
        )

        # ── Grid2Op 电压仿真（可开关切换）───────────────────────
        self._use_grid2op = use_grid2op
        self._grid2op_backend = grid2op_backend
        self._grid2op_power_flow: "Grid2OpPowerFlow | None" = None
        self._grid2op_init_failed = False

        # 观测/动作空间 (v2.5: 58维单模式, 59维多模式)
        obs_dim = 58 if self._mode != "all" else 59
        low_obs = np.full(obs_dim, -10.0, dtype=np.float32)
        high_obs = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = Box(low_obs, high_obs, dtype=np.float32)

        # 动作: [p_batt_norm, load_shed_norm, pv_limit_norm] (3维)
        # Q_batt 由实时电压调节器闭环控制，不经过 RL
        # pv_limit: 光伏有功限值比例 (0.0=全部切除, 1.0=全部出力)
        low_act = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        high_act = np.array([1.0, 1.0, 1.0], dtype=np.float32)
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
        # v2.5 新增字段
        self._q_realtime_margin: float = 0.5  # 实时模块剩余无功容量比例 [0.0, 1.0]
        self._season_encoding: np.ndarray = np.zeros(6, dtype=np.float32)  # 季节 one-hot
        self._time_period_encoding: np.ndarray = np.zeros(2, dtype=np.float32)  # 时段 one-hot [白天, 夜间]
        # v2.6 新增: 电压斜率追踪（用于 |ΔV| 惩罚）
        self._prev_v_avg: float = 1.0

    def _init_grid2op(self) -> None:
        """延迟初始化 Grid2Op（首次 reset 前不创建）。

        在 use_grid2op=True 时调用，尝试创建 Grid2OpPowerFlow 实例。
        如果 Grid2Op 不可用，标记为失败并降级到 VoltageSimulator。
        """
        if not self._use_grid2op or self._grid2op_init_failed:
            return

        try:
            from grid2op_env import Grid2OpPowerFlow, NumpyChronics, create_mupc_network

            net = create_mupc_network()
            chronics = NumpyChronics(self._data, force_china_data=False)
            self._grid2op_power_flow = Grid2OpPowerFlow(
                net, chronics, storage_soc_init=self._soc
            )
        except Exception as e:
            # Grid2Op 不可用：降级到 VoltageSimulator
            print(f"[WARN] Grid2Op 初始化失败，降级到 VoltageSimulator: {e}")
            self._grid2op_init_failed = True
            self._use_grid2op = False
            self._grid2op_power_flow = None

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
        self._prev_v_avg = 1.0  # v2.6: 电压斜率追踪

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

        # ── Grid2Op 初始化/重置 ────────────────────────────────
        if self._use_grid2op:
            if self._grid2op_power_flow is None:
                self._init_grid2op()
            if self._grid2op_power_flow is not None:
                self._grid2op_power_flow.reset(initial_storage_soc=self._soc)

        # 电压越限计数器（用于死区触发）
        self._voltage_violation_count: int = 0

        # v2.5 新增: 计算季节和时段编码
        self._update_season_time_encoding()

        obs = self._build_observation()
        info = {"mode": self._current_mode}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真 (v2.4 分层控制: Q_batt 由实时电压环给出).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # ── Grid2Op SOC 同步（step 入口：Grid2Op → MupcEnv）─────
        if self._use_grid2op and self._grid2op_power_flow is not None:
            grid_soc = self._grid2op_power_flow.get_storage_soc()
            self._soc = float(grid_soc)

        # 1. 计算 Q_batt (由实时电压环给出，基于前一步电压)
        v_prev = (self._va + self._vb + self._vc) / 3.0
        v_error = v_prev - 1.0
        q_batt = float(np.clip(-K_Q * v_error, -Q_BATT_MAX_KVAR, Q_BATT_MAX_KVAR))

        # v2.5: 计算 q_realtime_margin = 1 - |q_batt| / Q_BATT_MAX
        # 0=打满(无裕度), 1=空闲(最大裕度)
        self._q_realtime_margin = 1.0 - (abs(q_batt) / Q_BATT_MAX_KVAR)

        # v2.5: 更新季节时段编码 (每小时更新一次即可，这里每步更新)
        self._update_season_time_encoding()

        # 2. 动作约束校验 (ACT-01/03/05)
        dispatch_p = self._data["dispatch_p_set"][self._step_idx]
        if abs(dispatch_p) < 1e-6:
            dispatch_p_use = None
        else:
            dispatch_p_use = float(dispatch_p)
        clamped, violated, violations = self._validator.validate(
            action, dispatch_p_use, q_batt_real=q_batt)

        # 3. 反归一化动作到物理值 (3维: P_batt + Load_shedding + Pv_limit)
        p_batt = clamped[0] * P_BATT_MAX_KW
        load_shed = clamped[1] * LOAD_SHED_MAX_KW
        pv_limit = clamped[2] * 1.0  # [0.0, 1.0] 无量纲
        # Q_batt 已在上方计算，由实时电压调节器闭环给出

        # 4. 有效负荷与光伏 (pv_limit 可主动弃光)
        p_load_raw = float(self._data["load_power"][self._step_idx])
        p_load_eff = max(0.0, p_load_raw - load_shed)
        p_pv_raw = float(self._data["pv_power"][self._step_idx])
        p_pv_eff = p_pv_raw * pv_limit  # v2.6: 主动弃光

        # 4. SOC 更新 (SAFETY: hard clamp)
        soc_raw = self._soc + (-p_batt * DT_HOURS) / BATTERY_CAPACITY_KWH
        soc_new = float(np.clip(soc_raw, SOC_MIN, SOC_MAX))
        soc_clipped = abs(soc_raw - soc_new) > 1e-9

        # 5. 电网交换功率
        grid_power = p_load_eff - p_pv_eff + p_batt

        # 6. 变压器负载率
        q_load = p_load_eff * math.tan(math.acos(LOAD_PF))
        s_transformer = math.sqrt(grid_power ** 2 + (q_load - q_batt) ** 2)
        load_rate = s_transformer / TRANSFORMER_KVA
        load_rate_unclamped = load_rate  # 用于奖励计算

        # 7. 电压更新 (Grid2Op 潮流计算 或 VoltageSimulator 降级)
        has_illegal = False
        if self._use_grid2op and self._grid2op_power_flow is not None:
            # 转换为 Grid2Op 单位: kW → MW
            storage_p_mw = p_batt / 1000.0
            storage_q_mvar = q_batt / 1000.0  # kVar → MVar

            va, vb, vc, has_illegal = self._grid2op_power_flow.step(
                storage_p_mw, storage_q_mvar)

            # 同步 SOC 到 Grid2Op（step 出口：MupcEnv → Grid2Op）
            self._grid2op_power_flow.set_storage_soc(soc_new)
        else:
            # 降级到原 VoltageSimulator
            p_net = p_pv_eff - p_load_eff + p_batt
            va, vb, vc = self._voltage_sim.step(
                p_net, float(q_batt), self._va, self._vb, self._vc)
        v_avg = (va + vb + vc) / 3.0

        # v2.6: 电压斜率追踪（用于 |ΔV| 惩罚）
        delta_v = abs(v_avg - self._prev_v_avg)

        # 8. 电压越限计数器 (死区: ±5%, [0.95, 1.05])
        V_DEAD = 0.05
        if abs(v_avg - 1.0) > V_DEAD:
            self._voltage_violation_count += 1
        else:
            self._voltage_violation_count = 0  # 恢复正常则重置

        # 9. 需量更新 (1 小时滑动窗口)
        window = 4
        demand_start = max(0, self._step_idx - window + 1)
        demand_slice = self._data["load_power"][demand_start:self._step_idx + 1]
        current_demand = max(float(np.mean(demand_slice)), CONTRACT_DEMAND_KW * 0.3)
        peak_demand = max(self._peak_demand, current_demand)

        # 10. 更新内部状态
        self._soc = soc_new
        self._va = va
        self._vb = vb
        self._vc = vc
        self._battery_power_prev = p_batt
        self._grid_power = grid_power
        self._load_rate = load_rate
        self._current_demand = current_demand
        self._peak_demand = peak_demand
        prev_p_batt_for_reward = self._prev_p_batt
        self._prev_p_batt = p_batt
        self._prev_q_batt = float(q_batt)
        self._prev_v_avg = v_avg  # v2.6: 电压斜率追踪

        # 11. 奖励计算
        reward, reward_info = self._compute_reward(
            p_batt=p_batt, q_batt=float(q_batt), load_shed=load_shed,
            pv_limit=pv_limit, p_pv_raw=p_pv_raw, p_load_raw=p_load_raw,
            p_load_eff=p_load_eff, grid_power=grid_power,
            load_rate=load_rate, load_rate_unclamped=load_rate_unclamped,
            soc_new=soc_new, soc_clipped=soc_clipped,
            va=va, vb=vb, vc=vc,
            prev_p_batt=prev_p_batt_for_reward,
            voltage_violation_count=self._voltage_violation_count,
            delta_v=delta_v,
        )

        # 12. 推进时间
        self._step_idx += 1
        terminated = (self._step_idx - self._episode_start) >= EPISODE_LENGTH
        truncated = self._step_idx >= self._data_len - 16

        # 13. 构建 info
        info = {
            "mode": self._current_mode,
            "soc": self._soc,
            "load_rate": self._load_rate,
            "p_batt": p_batt,
            "q_batt": float(q_batt),
            "load_shedding": load_shed,
            "pv_limit": float(pv_limit), # v2.6: 主动弃光
            "grid_power": grid_power,
            "va": va, "vb": vb, "vc": vc,
            "v_avg": float(v_avg),
            "current_demand": current_demand,
            "peak_demand": peak_demand,
            "soc_clipped": soc_clipped,
            "constraint_violated": violated,
            "violations": str(violations) if violations else "",
            "voltage_violation_count": self._voltage_violation_count,
            "has_illegal": has_illegal,  # Grid2Op 潮流不收敛标记
            **reward_info,
        }
        if terminated or truncated:
            info["terminal_observation"] = self._build_observation()

        obs = self._build_observation()
        return obs, float(reward), terminated, truncated, info

    # ── v2.5 季节时段编码 ────────────────────────────────

    def _update_season_time_encoding(self) -> None:
        """根据当前时间步计算季节和时段 one-hot 编码。

        季节编码 (6维): [灌溉季, 炒茶季, 空调季, 常规季, 保留, 保留]
        时段编码 (2维): [白天, 夜间]
        """
        hour = float(self._data["hours"][self._step_idx])
        month = float(self._data.get("months", np.ones(self._data_len) * 7)[self._step_idx])

        # 季节编码 (互斥月份分组，防止 one-hot 冲突)
        # 灌溉: 3-4月; 炒茶: 5月; 空调: 6-8月; 常规: 其余
        season = np.zeros(6, dtype=np.float32)
        if 3 <= month < 5:  # 3-4月
            season[0] = 1.0  # 灌溉季
        elif 5 <= month < 6:  # 5月
            season[1] = 1.0  # 炒茶季
        elif 6 <= month <= 8:  # 6-8月
            season[2] = 1.0  # 空调季
        else:
            season[3] = 1.0  # 常规季
        self._season_encoding = season

        # 时段编码: 白天=6-18, 夜间=18-6
        time_period = np.zeros(2, dtype=np.float32)
        if 6 <= hour < 18:
            time_period[0] = 1.0  # 白天
        else:
            time_period[1] = 1.0  # 夜间
        self._time_period_encoding = time_period

    # ── 观测构建 ────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """构建 58 维观测向量 (多模式追加 mode_id 为 59 维)。

        对齐下游 MUPC AI 引擎设计文档 v2.5 to_input_vector:

        索引   内容                      字段数
        [0..9]  D1: 10 标量              (含 q_realtime_margin)
        [10..24] D2 pv_forecast          15维
        [25..39] D2 load_forecast         15维
        [41..43] D3 电价                  3字段
        [44..46] D4 需量                  3字段
        [47..48] D5 气象                  2字段
        [49]     D6 dispatch_p_set        1字段
        [50]     D7 q_realtime_margin     1字段
        [51..56] D7 season_encoding       6字段
        [57]     D7 time_period_encoding  1字段
        [58]     (可选) mode_id           1字段

        总维度: 10+15+15+3+3+2+1+1+6+1 = 58 (单模式)
        """
        obs_dim = 58 if self._mode != "all" else 59
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
        obs[9] = self._q_realtime_margin  # v2.5 新增: q 实时裕度

        # ── D2 [10..24] pv_forecast (15维) ──
        forecast = self._predictor.predict(self._step_idx)  # (30,)
        obs[10:25] = forecast[:15]

        # ── D2 [25..39] load_forecast (15维) ──
        obs[25:40] = forecast[15:30]

        # ── D3 [41..43] 电价 ──
        obs[41] = self._data["current_electricity_price"][self._step_idx]
        obs[42] = self._data["next_period_price"][self._step_idx]
        obs[43] = self._data["price_tariff_id"][self._step_idx]

        # ── D4 [44..46] 需量 ──
        obs[44] = self._current_demand
        obs[45] = CONTRACT_DEMAND_KW
        obs[46] = self._peak_demand

        # ── D5 [47..48] 气象 ──
        obs[47] = self._data["solar_irradiance"][self._step_idx]
        obs[48] = self._data["temperature"][self._step_idx]

        # ── D6 [49] 调度 ──
        dp = self._data["dispatch_p_set"][self._step_idx]
        obs[49] = dp if abs(dp) > 1e-6 else 0.0

        # ── D7 [50] q_realtime_margin ──
        obs[50] = self._q_realtime_margin

        # ── D7 [51..56] season_encoding (6维) ──
        obs[51:57] = self._season_encoding

        # ── D7 [57] time_period_encoding (1维) ──
        obs[57] = self._time_period_encoding[0]  # 0=夜间, 1=白天 (二进制编码)

        # ── mode_id [58] (可选) ──
        if self._mode == "all":
            obs[58] = MODE_ID_MAP[self._current_mode]

        # ── 归一化 ──
        obs = self._normalize_obs(obs, params)
        return obs.astype(np.float32)

    def _normalize_obs(self, obs: np.ndarray, params: dict) -> np.ndarray:
        """应用 MinMax 归一化 (v2.5: 56维)。"""
        out = obs.copy()
        # D1 [0..9]
        out[0] = obs[0]  # SOC: identity
        out[1] = self._minmax(obs[1], 0.0, PV_ARRAY_KW)
        out[2] = self._minmax(obs[2], 0.0, LOAD_PEAK_KW)
        out[3] = self._minmax(obs[3], -500.0, 500.0)
        out[4] = obs[4]  # transformer_load: identity
        out[5] = self._minmax(obs[5], -500.0, 500.0)
        out[6] = self._minmax(obs[6], 0.85, 1.15)
        out[7] = self._minmax(obs[7], 0.85, 1.15)
        out[8] = self._minmax(obs[8], 0.85, 1.15)
        out[9] = obs[9]  # q_realtime_margin: identity [0,1]
        # D2 pv [10..24] (15维，索引 10-24，out[10:25] 包含 10-24)
        out[10:25] = self._minmax(obs[10:25], 0.0, PV_ARRAY_KW)
        # D2 load [25..39] (15维，索引 25-39) — 修正：原为 [26:41] 错误偏移了1位
        out[25:40] = self._minmax(obs[25:40], 0.0, LOAD_PEAK_KW)
        # D3 [41..43]
        out[41] = self._minmax(obs[41], 0.0, 1.5)
        out[42] = self._minmax(obs[42], 0.0, 1.5)
        out[43] = self._minmax(obs[43], 0.0, 3.0)
        # D4 [44..46]
        out[44] = self._minmax(obs[44], 0.0, 500.0)
        out[45] = self._minmax(obs[45], 0.0, 500.0)
        out[46] = self._minmax(obs[46], 0.0, 500.0)
        # D5 [47..48]
        out[47] = self._minmax(obs[47], 0.0, 1500.0)
        out[48] = self._minmax(obs[48], -20.0, 60.0)
        # D6 [49]
        out[49] = self._minmax(obs[49], -500.0, 500.0)
        # D7 [50] q_realtime_margin: identity
        out[50] = obs[50]
        # D7 [51..56] season_encoding: one-hot, identity
        # D7 [57] time_period: binary, identity
        # mode_id [58]: identity
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
        """R = w1*R_pv - α(s)*w2*P_batt_deg - w3*P_overload - w4*P_voltage - w5*R_ramp

        v2.5 变更:
          - 弃光奖励电压前置条件: v_avg >= VOLTAGE_HIGH_LIMIT (1.05) → R_pv = 0
          - 自适应损耗系数 α(s) ∈ {3.0, 0.2, 1.0}: SOC极低保护 / 电压支撑 / 常规
          - 电压惩罚条件触发: q_realtime_margin <= Q_MARGIN_THRESHOLD (10%) 且越限 >= 2 步
        """
        v_avg = (r["va"] + r["vb"] + r["vc"]) / 3.0

        # ── 光伏消纳率 (含电压前置条件 + v2.6 主动弃光) ──
        pv_total = max(r["p_pv_raw"], 1e-6)  # 原始光伏总出力
        pv_limit = r.get("pv_limit", 1.0)    # v2.6: 主动弃光比例
        pv_eff = r["p_pv_raw"] * pv_limit    # 限后光伏出力
        pv_self = min(pv_eff, r["p_load_raw"]) + max(0.0, -r["p_batt"])
        r_pv = min(pv_self / pv_total, 1.0)
        # v2.5: 电压偏高时弃光奖励不计入
        if v_avg >= VOLTAGE_HIGH_LIMIT:
            r_pv = 0.0

        # ── 自适应损耗系数 α(s) ──
        soc_new = r.get("soc_new", self._soc)
        if soc_new < SOC_CRITICAL:
            alpha = 3.0  # SOC 极低保护
        elif self._q_realtime_margin <= Q_MARGIN_THRESHOLD and r.get("voltage_violation_count", 0) >= 2:
            alpha = 0.2  # 电压支撑模式
        else:
            alpha = 1.0  # 常规调度

        # ── 过载惩罚: 梯度从 75% 开始（Quadratic + Linear）──
        lr_unc = r.get("load_rate_unclamped", r["load_rate"])
        overload_t = max(0.0, (lr_unc - 0.75) / 0.25)
        p_overload = -0.3095 * overload_t ** 2 + 0.026 * overload_t

        # ── 电池衰减: C-rate² × α(s)（v2.5 自适应系数）──
        c_rate = abs(r["p_batt"]) / BATTERY_CAPACITY_KWH
        p_batt_deg = alpha * (c_rate ** 2)

        # ── 电压质量惩罚 (v2.5: 条件触发式) ──
        p_voltage = 0.0
        violation_count = r.get("voltage_violation_count", 0)
        dev = abs(v_avg - 1.0)

        if dev > VOLTAGE_DEADBAND and violation_count >= 2 and self._q_realtime_margin <= Q_MARGIN_THRESHOLD:
            dev_excess = dev - VOLTAGE_DEADBAND
            if v_avg < 1.0:
                p_voltage = VOLTAGE_PENALTY_LOW * (dev_excess / 0.10) ** 2
            else:
                p_voltage = VOLTAGE_PENALTY_HIGH * (dev_excess / 0.10) ** 2

        w4 = w[3] if len(w) > 3 else 0.0

        # ── 功率变化率惩罚 ──
        prev_p = r.get("prev_p_batt", 0.0)
        delta_p = abs(r["p_batt"] - prev_p)
        w5 = w[4] if len(w) > 4 else 0.0
        p_ramp_penalty = w5 * delta_p / BATTERY_CAPACITY_KWH

        # ── v2.6: 电压变化斜率惩罚（阻抗感知，迫使 AI 平滑调节）──
        delta_v = r.get("delta_v", 0.0)
        w6 = w[5] if len(w) > 5 else 0.0
        p_voltage_slope = w6 * delta_v

        total = (w[0] * r_pv
                 - w[1] * p_batt_deg
                 - w[2] * p_overload
                 - w4 * p_voltage
                 - p_ramp_penalty
                 - p_voltage_slope)

        info = {
            "r_pv_consumption": float(r_pv),
            "p_battery_degradation": float(-p_batt_deg),
            "p_transformer_overload": float(-p_overload),
            "p_voltage_deviation": float(-p_voltage),
            "p_ramp_penalty": float(-p_ramp_penalty),
            "p_voltage_slope": float(-p_voltage_slope),  # v2.6
            "v_avg": float(v_avg),
            "alpha": float(alpha),  # v2.5
            "q_realtime_margin": float(self._q_realtime_margin),  # v2.5
        }
        return float(total), info

    # ── MODE-02: 自主套利 ──────────────────────────────

    def _reward_arbitrage(self, r: dict, w: list[float]) -> tuple[float, dict]:
        """R = w1*R_spread - w2*P_batt_deg - w3*P_overload"""
        price = float(self._data["current_electricity_price"][self._step_idx])
        avg_price = 0.8
        r_spread = r["p_batt"] * (price - avg_price) / (P_BATT_MAX_KW * 0.4)
        r_spread = float(np.clip(r_spread, -1.0, 1.0))

        # 电池衰减: C-rate²（非线性）
        c_rate = abs(r["p_batt"]) / BATTERY_CAPACITY_KWH
        p_batt_deg = c_rate ** 2

        # 过载惩罚: 梯度从 75% 开始（Quadratic + Linear，匹配设计文档）
        lr_unc = r.get("load_rate_unclamped", r["load_rate"])
        overload_t = max(0.0, (lr_unc - 0.75) / 0.25)
        p_overload = -0.3095 * overload_t ** 2 + 0.026 * overload_t
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
    print(f"  观测形状: {obs2.shape}  (应为 59 维)")
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
