"""
MUPC 全状态 RL 环境主类 (v2.14 分层控制架构).

使用模块化架构:
- constants.py: 物理常数 + 归一化边界 + 权重映射
- voltage_sim.py: 三相电压简化线路模型
- observation.py: 观测构建 + 归一化 + 季节编码
- rewards.py: 奖励调度器 + 5 场景奖励函数

动作空间 3 维: [p_batt, load_shedding, pv_limit]
Q_batt 由实时电压调节器闭环给出，不经过 RL 动作输出。
观测空间: Box(63,) 单模式 或 Box(64,) 多模式
"""

import math
from typing import Any, Optional

import numpy as np

# Gymnasium 降级处理 (保持与原始 mupc_env.py 一致)
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    _GYM_AVAILABLE = True
except ImportError:
    from _gym_stub import Env as _GymStubEnv
    from _gym_stub import Box
    _GYM_AVAILABLE = False

from action_validator import ActionValidator

from . import constants
from . import observation
from . import rewards
from .voltage_sim import VoltageSimulator


class MupcEnv(gym.Env if _GYM_AVAILABLE else _GymStubEnv):
    """MUPC 全状态 RL 环境 (v2.14 分层控制架构).

    动作空间 3 维: [p_batt, load_shedding, pv_limit]
    Q_batt 由实时电压调节器闭环给出，不经过 RL 动作输出。
    观测空间: Box(63,) 或 Box(64,) (多模式)
    """

    metadata = {"render_modes": []}

    # ── 初始化 ────────────────────────────────────────────

    def __init__(self, data: dict, mode: str = "all",
                 lstm_predictor: Any = None,
                 reward_weights: Optional[dict[str, list[float]]] = None,
                 config: Any = None):
        """
        Args:
            data: SmartDSLoader 返回的 data dict
            mode: "all" (多模式) 或 "MODE-01"~"MODE-05" (单模式)
            lstm_predictor: LSTM 模型 (有 predict(step_idx)→(30,) 接口) 或 None→Oracle
            reward_weights: 自定义权重, e.g. {"MODE-01": [1.5, 0.3, 3.0]}
            config: MupcConfig 配置对象，None 则使用硬编码默认值
        """
        self._data = data
        self._mode = mode
        self._data_len = data["n_steps"]
        self._weights = {**constants.DEFAULT_WEIGHTS, **(reward_weights or {})}

        # v2.7 配置支持
        self._cfg = config

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

        # 观测/动作空间 (v2.14: 63维单模式, 64维多模式)
        obs_dim = 63 if mode != "all" else 64
        low_obs = np.full(obs_dim, -10.0, dtype=np.float32)
        high_obs = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = Box(low_obs, high_obs, dtype=np.float32)

        # v2.7 下垂模式检测
        self._dual_mode = False
        self._dual_validator: Optional[Any] = None
        if config is not None and getattr(config.dual_control, 'enabled', False):
            self._dual_mode = True
            from action_validator import DualActionValidator
            self._dual_validator = DualActionValidator(
                p_batt_max=constants.P_BATT_MAX_KW,
                k_droop_min=self._cfg.dual_control.k_droop_min,
                k_droop_max=self._cfg.dual_control.k_droop_max,
                p_ref_ramp_limit_kw=self._cfg.dual_control.p_ref_ramp_limit_kw,
                load_shed_max=constants.LOAD_SHED_MAX_KW,
                pv_limit_min=self._cfg.dual_control.pv_limit_min,
            )
            low_act = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
            high_act = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            low_act = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            high_act = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = Box(low_act, high_act, dtype=np.float32)

        # ── 内部状态 ──
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
        self._prev_k_droop: float = 0.0
        self._prev_v_avg: float = 1.0

        # v2.10: D9 安全覆盖状态
        self._safety_override_active: bool = False
        self._safety_override_p_ref: float = 0.0

        # v2.13: Welford 动态归一化
        self._welford_mean: float = 0.0
        self._welford_m2: float = 1.0
        self._welford_count: int = 0
        self._prev_v_dev: float = 0.0

        # v2.14: SafetyOverride 精细化跟踪
        self._override_count: int = 0
        self._override_window: int = 0
        self._override_ratio: float = 0.0
        self._override_consecutive: int = 0

        # v2.5: 季节/时段编码
        self._q_realtime_margin: float = 0.5
        self._season_encoding: np.ndarray = np.zeros(6, dtype=np.float32)
        self._time_period_encoding: np.ndarray = np.zeros(2, dtype=np.float32)

        # 电压越限
        self._voltage_violation_count: int = 0

    # ── 模式管理 ────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """运行时切换运行场景。"""
        if mode in constants.ALL_MODES:
            self._current_mode = mode
        else:
            raise ValueError(f"无效模式: {mode}, 有效值: {constants.ALL_MODES}")

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def mode(self) -> str:
        return self._mode

    # ── 辅助状态构建 ────────────────────────────────────

    def _make_env_state(self) -> observation.EnvState:
        """从当前 self._* 状态构建 EnvState 快照 (用于观测构建)。"""
        return observation.EnvState(
            soc=self._soc,
            pv_power=float(self._data["pv_power"][self._step_idx]),
            load_power=float(self._data["load_power"][self._step_idx]),
            grid_power=self._grid_power,
            load_rate=self._load_rate,
            battery_power_prev=self._battery_power_prev,
            va=self._va, vb=self._vb, vc=self._vc,
            q_realtime_margin=self._q_realtime_margin,
            current_price=float(self._data["current_electricity_price"][self._step_idx]),
            next_price=float(self._data["next_period_price"][self._step_idx]),
            tariff_id=float(self._data["price_tariff_id"][self._step_idx]),
            current_demand=self._current_demand,
            peak_demand=self._peak_demand,
            solar_irradiance=float(self._data["solar_irradiance"][self._step_idx]),
            temperature=float(self._data["temperature"][self._step_idx]),
            dispatch_p_set=float(self._data["dispatch_p_set"][self._step_idx]),
            season_encoding=self._season_encoding,
            time_period_encoding=self._time_period_encoding,
            safety_override_active=self._safety_override_active,
            safety_override_p_ref=self._safety_override_p_ref,
            override_consecutive=self._override_consecutive,
            override_ratio=self._override_ratio,
            current_mode=self._current_mode,
            is_multi_mode=(self._mode == "all"),
        )

    def _make_reward_dict(self, **extra) -> dict:
        """组装奖励函数所需的 r dict (纯数据，不引用 self)。"""
        r = {
            "p_batt": extra["p_batt"],
            "q_batt": extra["q_batt"],
            "load_shed": extra["load_shed"],
            "pv_limit": extra["pv_limit"],
            "p_pv_raw": extra["p_pv_raw"],
            "p_load_raw": extra["p_load_raw"],
            "p_load_eff": extra["p_load_eff"],
            "grid_power": extra["grid_power"],
            "load_rate": extra["load_rate"],
            "load_rate_unclamped": extra.get("load_rate_unclamped", extra["load_rate"]),
            "soc": self._soc,
            "soc_new": extra["soc_new"],
            "soc_clipped": extra.get("soc_clipped", False),
            "va": extra["va"], "vb": extra["vb"], "vc": extra["vc"],
            "prev_p_batt": extra["prev_p_batt"],
            "prev_v_avg": extra["prev_v_avg"],
            "prev_v_dev": extra["prev_v_dev"],
            "k_droop": extra.get("k_droop", 0.0),
            "prev_k_droop": extra.get("prev_k_droop", 0.0),
            "voltage_violation_count": self._voltage_violation_count,
            "safety_override_active": self._safety_override_active,
            "safety_override_p_ref": self._safety_override_p_ref,
            "override_consecutive": self._override_consecutive,
            "override_ratio": self._override_ratio,
            # 奖励函数中隐式依赖的附加字段
            "q_realtime_margin": self._q_realtime_margin,
            "current_demand": self._current_demand,
            "prev_p_batt_raw": self._prev_p_batt,
            "dispatch_p_set": float(self._data["dispatch_p_set"][self._step_idx]),
            "current_price": float(self._data["current_electricity_price"][self._step_idx]),
        }
        return r

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
        self._prev_k_droop = 0.0
        self._prev_v_avg = 1.0
        self._safety_override_active = False
        self._safety_override_p_ref = 0.0
        self._welford_mean = 0.0
        self._welford_m2 = 1.0
        self._welford_count = 0
        self._prev_v_dev = 0.0
        self._override_count = 0
        self._override_window = 0
        self._override_ratio = 0.0
        self._override_consecutive = 0
        self._voltage_violation_count = 0

        # 随机起始索引
        max_start = self._data_len - constants.EPISODE_LENGTH - 16
        self._episode_start = np.random.randint(0, max(1, max_start))
        self._step_idx = self._episode_start

        # 设置当前场景
        if self._mode == "all":
            self._current_mode = np.random.choice(constants.ALL_MODES)
        else:
            self._current_mode = self._mode

        # 重置校验器
        self._validator.reset()
        if self._dual_validator is not None:
            self._dual_validator.reset()

        # 计算季节时段编码
        self._update_season_time_encoding()

        state = self._make_env_state()
        forecast = self._predictor.predict(self._step_idx)
        obs = observation.build_observation(state, forecast)
        info = {"mode": self._current_mode}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真 (v2.4 分层控制: Q_batt 由实时电压环给出)."""
        action = np.asarray(action, dtype=np.float32)

        # 1. 计算 Q_batt (由实时电压环给出)
        v_prev = (self._va + self._vb + self._vc) / 3.0
        v_error = v_prev - 1.0
        K_Q_V = 200.0
        q_batt = float(np.clip(-K_Q_V * v_error,
                               -constants.Q_BATT_MAX_KVAR,
                               constants.Q_BATT_MAX_KVAR))

        # v2.5: q_realtime_margin
        self._q_realtime_margin = 1.0 - (abs(q_batt) / constants.Q_BATT_MAX_KVAR)

        # 季节时段编码
        self._update_season_time_encoding()

        # 2. 动作约束校验
        dispatch_p = self._data["dispatch_p_set"][self._step_idx]
        dispatch_p_use = float(dispatch_p) if abs(dispatch_p) >= 1e-6 else None

        if self._dual_mode:
            clamped_dual, violated, violations = self._dual_validator.validate(
                action, dispatch_p_use, is_anti_reverse=False)
            p_ref = clamped_dual[0] * constants.P_BATT_MAX_KW
            k_droop = clamped_dual[1] * (self._cfg.dual_control.k_droop_max -
                                          self._cfg.dual_control.k_droop_min) / 2.0 + \
                      (self._cfg.dual_control.k_droop_max +
                       self._cfg.dual_control.k_droop_min) / 2.0
            load_shed = clamped_dual[2] * constants.LOAD_SHED_MAX_KW
            pv_limit = clamped_dual[3] * 1.0
            p_batt = p_ref
        else:
            clamped, violated, violations = self._validator.validate(
                action, dispatch_p_use, q_batt_real=q_batt)
            p_batt = clamped[0] * constants.P_BATT_MAX_KW
            load_shed = clamped[1] * constants.LOAD_SHED_MAX_KW
            pv_limit = clamped[2] * 1.0 if len(clamped) > 2 else 1.0
            k_droop = 0.0

        # 3. 有效负荷与光伏
        p_load_raw = float(self._data["load_power"][self._step_idx])
        p_load_eff = max(0.0, p_load_raw - load_shed)
        p_pv_raw = float(self._data["pv_power"][self._step_idx])
        p_pv_eff = p_pv_raw * pv_limit

        # 4. SOC 更新 (SAFETY: hard clamp)
        soc_raw = self._soc + (-p_batt * constants.DT_HOURS) / constants.BATTERY_CAPACITY_KWH
        soc_new = float(np.clip(soc_raw, constants.SOC_MIN, constants.SOC_MAX))
        soc_clipped = abs(soc_raw - soc_new) > 1e-9

        # 5. 电网交换功率
        grid_power = p_load_eff - p_pv_eff + p_batt

        # 6. 变压器负载率
        q_load = p_load_eff * math.tan(math.acos(constants.LOAD_PF))
        s_transformer = math.sqrt(grid_power ** 2 + (q_load - q_batt) ** 2)
        load_rate = s_transformer / constants.TRANSFORMER_KVA
        load_rate_unclamped = load_rate

        # 7. 电压更新
        p_net = p_pv_eff - p_load_eff + p_batt
        va, vb, vc = self._voltage_sim.step(
            p_net, float(q_batt), self._va, self._vb, self._vc)
        v_avg = (va + vb + vc) / 3.0

        # 8. 电压越限计数器
        V_DEAD = 0.05
        if abs(v_avg - 1.0) > V_DEAD:
            self._voltage_violation_count += 1
        else:
            self._voltage_violation_count = 0

        # 9. 需量更新 (1 小时滑动窗口)
        window = 4
        demand_start = max(0, self._step_idx - window + 1)
        demand_slice = self._data["load_power"][demand_start:self._step_idx + 1]
        current_demand = max(float(np.mean(demand_slice)), constants.CONTRACT_DEMAND_KW * 0.3)
        peak_demand = max(self._peak_demand, current_demand)

        # 10. 更新内部状态 (保存 prev_* 用于奖励)
        prev_p_batt_for_reward = self._prev_p_batt
        prev_k_droop_for_reward = self._prev_k_droop
        prev_v_avg_for_reward = self._prev_v_avg
        prev_v_dev_for_reward = self._prev_v_dev

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
        self._prev_q_batt = float(q_batt)
        self._prev_k_droop = k_droop
        self._prev_v_avg = v_avg
        self._prev_v_dev = abs(v_avg - 1.0)

        # 11. 奖励计算 (使用 rewards 模块)
        r = self._make_reward_dict(
            p_batt=p_batt, q_batt=float(q_batt), load_shed=load_shed,
            pv_limit=pv_limit, p_pv_raw=p_pv_raw, p_load_raw=p_load_raw,
            p_load_eff=p_load_eff, grid_power=grid_power,
            load_rate=load_rate, load_rate_unclamped=load_rate_unclamped,
            soc_new=soc_new, soc_clipped=soc_clipped,
            va=va, vb=vb, vc=vc,
            prev_p_batt=prev_p_batt_for_reward,
            k_droop=k_droop,
            prev_k_droop=prev_k_droop_for_reward,
            prev_v_avg=prev_v_avg_for_reward,
            prev_v_dev=prev_v_dev_for_reward,
        )
        reward, reward_info = rewards.compute_reward(
            self._current_mode, self._weights, r)

        # Welford 更新 (从 rewards 返回值中读取)
        welford_raw = reward_info.pop("welford_raw", None)
        if welford_raw is not None:
            delta = welford_raw - self._welford_mean
            self._welford_count += 1
            self._welford_mean += delta / self._welford_count
            delta2 = welford_raw - self._welford_mean
            self._welford_m2 += delta * delta2

        # 12. 推进时间
        self._step_idx += 1
        terminated = (self._step_idx - self._episode_start) >= constants.EPISODE_LENGTH
        truncated = self._step_idx >= self._data_len - 16

        # 13. 构建 info
        info = {
            "mode": self._current_mode,
            "soc": self._soc,
            "load_rate": self._load_rate,
            "p_batt": p_batt,
            "q_batt": float(q_batt),
            "load_shedding": load_shed,
            "pv_limit": float(pv_limit),
            "grid_power": grid_power,
            "va": va, "vb": vb, "vc": vc,
            "v_avg": float(v_avg),
            "current_demand": current_demand,
            "peak_demand": peak_demand,
            "soc_clipped": soc_clipped,
            "constraint_violated": violated,
            "violations": str(violations) if violations else "",
            "voltage_violation_count": self._voltage_violation_count,
            **reward_info,
            "k_droop": float(k_droop) if self._dual_mode else 0.0,
            "p_ref": float(p_batt) if self._dual_mode else p_batt,
        }
        if terminated or truncated:
            state = self._make_env_state()
            forecast = self._predictor.predict(self._step_idx)
            info["terminal_observation"] = observation.build_observation(state, forecast)

        state = self._make_env_state()
        forecast = self._predictor.predict(self._step_idx)
        obs = observation.build_observation(state, forecast)
        return obs, float(reward), terminated, truncated, info

    # ── 季节时段编码 ────────────────────────────────────

    def _update_season_time_encoding(self) -> None:
        """根据当前时间步计算季节和时段 one-hot 编码。"""
        hour = float(self._data["hours"][self._step_idx])
        month = float(self._data.get("months",
                      np.ones(self._data_len, dtype=np.float32) * 7)[self._step_idx])
        self._season_encoding, self._time_period_encoding = \
            observation.update_season_time_encoding(hour, month)

    # ── VPP 调度合成 ────────────────────────────────────

    def _generate_vpp_dispatch(self) -> float:
        """VPP 模式: 随机生成调度指令 (每 96 步 20% 概率触发)。"""
        if self._step_idx % 96 == 0:
            if np.random.random() < 0.2:
                return float(np.random.uniform(-50.0, 50.0))
        return 0.0


# ═══════════════════════════════════════════════════════════════
# 自测入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_loader import SmartDSLoader

    print("=" * 56)
    print("  MUPC 环境自测 (模块化架构: mupc_env/)")
    print("=" * 56)

    loader = SmartDSLoader()
    loaded_data = loader.load_all()
    train, _val = loader.split(loaded_data)

    # 单模式测试
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
        if info["load_rate"] > constants.OVERLOAD_THRESHOLD:
            overload_count += 1
        if terminated or truncated:
            break

    print(f"  随机动作 20 步: 累计奖励={total_reward:.3f}, "
          f"过载次数={overload_count}, 最终SOC={info['soc']:.3f}")

    # 多模式测试
    print("\n── 多模式测试 (all) ──")
    env2 = MupcEnv(train, mode="all")
    obs2, info2 = env2.reset()
    print(f"  观测形状: {obs2.shape}  (应为 64 维)")
    print(f"  初始模式: {info2['mode']}")

    modes_seen = set()
    for i in range(500):
        action = env2.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env2.step(action)
        modes_seen.add(info2["mode"])
        if terminated or truncated:
            env2.reset()
    print(f"  500 步覆盖模式: {sorted(modes_seen)}")

    print(f"\n[PASS] mupc_env/core.py 自测通过")
