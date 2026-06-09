"""NumpyChronics — 将 data dict 单相标量转换为 Grid2Op 三相格式。

将 SmartDSLoader / UnifiedDataLoader 返回的 data dict 转换为 Grid2Op期望的
三相时序格式（load_p_mw / load_q_mvar / sgen_p_mw / sgen_q_mvar），
驱动 Grid2Op 环境仿真。

单相标量 → 三相展开逻辑：
- pv_power (n_steps,) → sgen_p_mw (3 相相同)
- load_power (n_steps,) → load_p_mw (3 相分解 + 轻微不平衡度)
- solar_irradiance / temperature → 环境变量（用于光伏出力计算）
"""

from __future__ import annotations

import numpy as np
from typing import Any


class NumpyChronics:
    """将 SmartDSLoader 的 data dict 转换为 Grid2Op 时序格式。

    data dict（单相标量） → Grid2Op 三相格式
    ├── pv_power (n_steps,) → sgen_p_mw (3 相相同)
    ├── load_power (n_steps,) → load_p_mw (3 相分解)
    ├── solar_irradiance → 环境变量
    └── temperature → 环境变量

    Attributes:
        _data: SmartDSLoader 返回的 data dict
        _n_steps: 总时间步数
        _current_idx: 当前时间步索引
        _n_loads: 负荷元件数量（固定为 2：居民负荷 + 农业冲击负荷）
        _n_sgens: 光伏元件数量（固定为 1：屋顶光伏）
    """

    # 三相不平衡度 (p.u.)
    PHASE_IMBALANCE: float = 0.003

    def __init__(self, data: dict) -> None:
        """初始化时序数据注入器。

        Args:
            data: SmartDSLoader.load_all() 返回的 data dict
               必需字段：pv_power, load_power, solar_irradiance, temperature
                可选字段：months, hours
        """
        self._data = data
        self._n_steps = data["n_steps"]
        self._current_idx: int = 0

        # 网络元件数量（与 create_mupc_network 一致）
        # 2 个负荷（Residential_Load, Agri_Shock_Load）
        # 1 个光伏（Rooftop_PV）
        self._n_loads: int = 2
        self._n_sgens: int = 1

        # 功率因数（用于计算无功负荷）
        self._load_pf: float = 0.90

    # ── 三相展开 ────────────────────────────────────────────────

    def _expand_single_to_three_phase(
        self, scalar_value: float, phase_imbalance: float = PHASE_IMBALANCE
    ) -> np.ndarray:
        """将单相标量展开为三相数组，叠加轻微不平衡度。

        Args:
            scalar_value: 单相标量值（任意单位，与返回值相同）
            phase_imbalance: 三相不平衡度 (p.u., default 0.003)
                每相在 base 值上叠加随机偏移：base * (1 ± uniform(0, imbalance))

        Returns:
            三相数组 (3,)，单位同输入
        """
        if scalar_value <= 0.0:
            return np.zeros(3, dtype=np.float64)

        base = scalar_value / 3.0 # 平均分配到三相
        phase_a = base * (1.0 + np.random.uniform(-phase_imbalance, phase_imbalance))
        phase_b = base * (1.0 + np.random.uniform(-phase_imbalance, phase_imbalance))
        phase_c = base * (1.0 + np.random.uniform(-phase_imbalance, phase_imbalance))
        return np.array([phase_a, phase_b, phase_c], dtype=np.float64)

    # ── Grid2Op Chronics 接口 ──────────────────────────────────

    def initialize(self, initial_storage_soc: float) -> None:
        """初始化 Grid2Op 环境第一帧。

        Args:
            initial_storage_soc: 初始 SOC (0.0~1.0)，同步到 Grid2Op storage 元件
        """
        self._current_idx = 0
        # CRITICAL-3:记录当前 SOC（由 Grid2OpPowerFlow 同步更新）
        self._current_soc: float = initial_storage_soc

    def load_next(self) -> dict[str, np.ndarray]:
        """返回下一帧数据（Grid2Op 格式）。

        Returns:
            dict: {
                "load_p_mw":   np.ndarray (n_loads, 3)  # 三相有功负荷 MW
                "load_q_mvar": np.ndarray (n_loads, 3)  # 三相无功负荷 MVar
                "sgen_p_mw":   np.ndarray (n_sgens,)   # 光伏有功 MW
                "sgen_q_mvar": np.ndarray (n_sgens,)   # 光伏无功 MVar
            }
        """
        idx = self._current_idx
        self._current_idx += 1

        # ── 负荷有功 (单相标量 → 三相) ──
        load_p_scalar = float(self._data["load_power"][idx])  # kW
        load_p_3ph = self._expand_single_to_three_phase(load_p_scalar)  # kW (3,)
        load_p_mw = load_p_3ph / 1000.0  # → MW (n_loads=2, 3) 按比例分配

        # 两个负荷按固定比例分配总负荷：
        # Residential_Load: 40%, Agri_Shock_Load: 60%
        res_p = load_p_mw * 0.4
        agri_p = load_p_mw * 0.6

        # 构建 (n_loads, 3) 数组
        load_p_mw_arr = np.vstack([res_p, agri_p])  # (2, 3) MW

        # ── 负荷无功 (基于功率因数计算) ──
        # Q = P * tan(acos(PF))，假设相同 PF
        load_q_scalar = load_p_scalar * np.tan(np.arccos(self._load_pf))
        load_q_3ph = self._expand_single_to_three_phase(load_q_scalar)
        load_q_mvar = load_q_3ph / 1000.0  # → MVar

        res_q = load_q_mvar * 0.4
        agri_q = load_q_mvar * 0.6
        load_q_mvar_arr = np.vstack([res_q, agri_q])  # (2, 3) MVar

        # ── 光伏有功 (单相标量 → 三相相同) ──
        pv_p_scalar = float(self._data["pv_power"][idx])  # kW
        sgen_p_mw = np.array([pv_p_scalar / 1000.0], dtype=np.float64)  # (1,) MW

        # ── 光伏无功 (设为 0，由潮流计算) ──
        sgen_q_mvar = np.zeros(1, dtype=np.float64)  # (1,) MVar

        return {
            "load_p_mw": load_p_mw_arr.astype(np.float64),
            "load_q_mvar": load_q_mvar_arr.astype(np.float64),
            "sgen_p_mw": sgen_p_mw.astype(np.float64),
            "sgen_q_mvar": sgen_q_mvar.astype(np.float64),
            # CRITICAL-3: 设计文档 4.1 节要求 storage_soc 字段
            "storage_soc": float(self._current_soc),
        }

    def reset(self, initial_storage_soc: float) -> None:
        """重置时序索引到 0（每个 episode 开始时调用）。

        Args:
            initial_storage_soc: 初始 SOC (0.0~1.0)
        """
        self._current_idx = 0
        self.initialize(initial_storage_soc)

    @property
    def current_timestep(self) -> int:
        """返回当前时间步索引。"""
        return self._current_idx

    def set_time_step(self, timestep: int) -> None:
        """设置当前时间步（用于从指定位置恢复）。

        Args:
            timestep: 目标时间步索引
        """
        self._current_idx = max(0, min(timestep, self._n_steps - 1))

    def set_storage_soc(self, soc: float) -> None:
        """更新当前 SOC（由 Grid2OpPowerFlow 每步同步调用）。

        Args:
            soc: 当前 SOC (0.0~1.0)
        """
        self._current_soc = float(np.clip(soc, 0.0, 1.0))

    @property
    def n_steps(self) -> int:
        """返回总时间步数。"""
        return self._n_steps