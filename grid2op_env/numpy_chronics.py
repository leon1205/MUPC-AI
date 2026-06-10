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

    def __init__(self, data: dict,
                 residential_ratio: float = 0.4,
                 agri_ratio: float = 0.6,
                 force_china_data: bool = False) -> None:
        """初始化时序数据注入器。

        Args:
            data: SmartDSLoader.load_all() 返回的 data dict
               必需字段：pv_power, load_power, solar_irradiance, temperature
                可选字段：months, hours
            residential_ratio: 居民负荷分配比例 (default 0.4)
            agri_ratio: 农业冲击负荷分配比例 (default 0.6)
            force_china_data: 强制使用中国数据模式，忽略 SMART-DS 地理不匹配警告
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

        # 负荷分配比例（可配置）
        self._residential_ratio = residential_ratio
        self._agri_ratio = agri_ratio

        # ── 数据源检测 ──────────────────────────────────────────
        # 检测数据是否具有季节性（中国合成数据特征）
        # SMART-DS 数据（美国加州）与中国农网场景不匹配
        months = data.get("months")
        if months is not None:
            unique_months = len(set(months))
            if unique_months <= 1 and not force_china_data:
                print("[WARN] 检测到数据可能来自 SMART-DS（美国加州），"
                      "与中国农网台区场景不匹配！")
                print("       建议使用 --data-source china 或 --data-source unified")
                print("       或设置 force_china_data=True 忽略此警告")

        # ── 农业冲击负荷状态 ────────────────────────────────────────
        # 农业冲击负荷（灌溉泵/炒茶设备）模拟：
        # 每 96 步（约1天）有概率触发冲击负荷，持续 4~16 步（1~4 小时）
        # 季节性：6-9月高概率高强度，其他月份低概率低强度
        self._shock_active: bool = False
        self._shock_magnitude: float = 0.0  # kW
        self._shock_countdown: int = 0        # 剩余持续步数

        # 冲击负荷参数（季节性）
        # 灌溉季（6-9月）：高概率 50%，高强度 100~200kW
        self._shock_trigger_prob_summer: float = 0.5   # 6-9月触发概率 50%
        self._shock_min_magnitude_summer: float = 100.0  # 最小冲击量 kW
        self._shock_max_magnitude_summer: float = 200.0  # 最大冲击量 kW
        self._shock_min_duration_summer: int = 8         # 最短持续步数
        self._shock_max_duration_summer: int = 16          # 最长持续步数
        # 非灌溉季（其他月份）：低概率 20%，低强度 30~80kW
        self._shock_trigger_prob_offseason: float = 0.2  # 其他月份触发概率 20%
        self._shock_min_magnitude_offseason: float = 30.0  # 最小冲击量 kW
        self._shock_max_magnitude_offseason: float = 80.0  # 最大冲击量 kW
        self._shock_min_duration_offseason: int = 4         # 最短持续步数
        self._shock_max_duration_offseason: int = 8         # 最长持续步数

        # ── 居民负荷随机噪声参数 ────────────────────────────────
        # 模拟家用电器启停产生的负荷波动（±5~15%）
        self._res_noise_std: float = 0.10  # 噪声标准差 10%（相对于基础负荷）

    # ── 农业冲击负荷模拟 ─────────────────────────────────────────

    def _apply_agri_shock(self, base_load_kw: float, step_idx: int, month: int) -> float:
        """在基础负荷上叠加农业冲击负荷随机性。

        农业冲击负荷（灌溉泵、炒茶设备等）特性：
        - 阶跃式变化：从正常运行突然跳到高功率
        - 持续时间：1~4 小时（4~16 步 @ 15min）
        - 触发规律：每 96 步（约1天）有概率出现冲击
        - 季节性：6-9月（灌溉季）高概率高强度，其他月份低概率低强度

        Args:
            base_load_kw: 基础负荷 kW
            step_idx: 当前时间步索引
            month: 当前月份 (1-12)
        Returns:
            叠加冲击后的负荷 kW（农业冲击仅影响 agri 负荷）
        """
        # 判断当前是否为灌溉季（6-9月）
        is_summer = 6 <= month <= 9

        # 选择季节性参数
        if is_summer:
            trigger_prob = self._shock_trigger_prob_summer
            min_mag = self._shock_min_magnitude_summer
            max_mag = self._shock_max_magnitude_summer
            min_dur = self._shock_min_duration_summer
            max_dur = self._shock_max_duration_summer
        else:
            trigger_prob = self._shock_trigger_prob_offseason
            min_mag = self._shock_min_magnitude_offseason
            max_mag = self._shock_max_magnitude_offseason
            min_dur = self._shock_min_duration_offseason
            max_dur = self._shock_max_duration_offseason

        # 每 96 步判断是否触发新的冲击（在步骤 0 触发）
        if step_idx > 0 and step_idx % 96 == 0:
            if np.random.random() < trigger_prob:
                self._shock_active = True
                self._shock_magnitude = np.random.uniform(min_mag, max_mag)
                self._shock_countdown = np.random.randint(min_dur, max_dur + 1)

        # 冲击激活中：叠加阶跃负荷
        if self._shock_active:
            self._shock_countdown -= 1
            if self._shock_countdown <= 0:
                self._shock_active = False
            return base_load_kw + self._shock_magnitude

        return base_load_kw

    def _apply_residential_noise(self, base_load_kw: float) -> float:
        """在居民负荷上叠加随机噪声模拟家用电器启停波动。

        居民负荷波动特性：
        - 小幅高频噪声：±5~15% 的随机波动
        - 由空调、冰箱、照明等启停产生
        - 每步独立采样

        Args:
            base_load_kw: 基础居民负荷 kW
        Returns:
            叠加噪声后的居民负荷 kW
        """
        if base_load_kw <= 0.0:
            return base_load_kw
        noise = np.random.normal(0.0, self._res_noise_std * base_load_kw)
        return max(0.0, base_load_kw + noise)

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

        # 获取当前月份（用于季节性冲击负荷）
        months = self._data.get("months")
        month = int(months[idx]) if months is not None else 6  # 默认6月

        # 农业冲击负荷仅影响 agri 负荷，residential 保持基础负荷
        agri_load_kw = load_p_scalar * self._agri_ratio
        agri_load_kw = self._apply_agri_shock(agri_load_kw, idx, month)  # 叠加冲击随机性（季节性）

        load_p_scalar_for_res = load_p_scalar * self._residential_ratio
        load_p_scalar_for_res = self._apply_residential_noise(load_p_scalar_for_res)  # 叠加居民负荷随机波动

        # residential 负荷三相展开
        res_p_3ph = self._expand_single_to_three_phase(load_p_scalar_for_res)  # kW (3,)
        res_p_mw = res_p_3ph / 1000.0  # → MW (3,)

        # agricultural 负荷三相展开（已叠加冲击）
        agri_p_3ph = self._expand_single_to_three_phase(agri_load_kw)  # kW (3,)
        agri_p_mw = agri_p_3ph / 1000.0  # → MW (3,)

        # 构建 (n_loads, 3) 数组
        load_p_mw_arr = np.vstack([res_p_mw, agri_p_mw])  # (2, 3) MW

        # ── 负荷无功 (基于功率因数计算) ──
        # Q = P * tan(acos(PF))，假设相同 PF
        load_q_scalar = load_p_scalar * np.tan(np.arccos(self._load_pf))
        load_q_3ph = self._expand_single_to_three_phase(load_q_scalar)
        load_q_mvar = load_q_3ph / 1000.0  # → MVar

        res_q = load_q_mvar * self._residential_ratio
        agri_q = load_q_mvar * self._agri_ratio
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
        # 重置农业冲击负荷状态
        self._shock_active = False
        self._shock_magnitude = 0.0
        self._shock_countdown = 0
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