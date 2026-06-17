"""
动作约束校验器 — 4 条约束规则，对齐下游 MUPC AI 引擎 PRD v2.15。

约束规则 (来自 MUPC AI 引擎 PRD v2.15 Section 6.5):
  ACT-01: |Δp_ref| <= DeltaP_Max kW/步
  ACT-02: |Δk_droop| <= DeltaK_Max kW/V/步
  ACT-03: p_ref ∈ [p_ref_min, p_ref_max] → [-50, 50] kW
  ACT-04: k_droop ∈ [k_droop_min, k_droop_max] → [0, 30] kW/V
  ACT-07: |p_ref| <= |dispatch_p| (当调度指令有效时)

v2.15: ACT-05(load_shedding)/ACT-06(pv_limit) 下沉至 strategy-engine,
       confidence 移至 ModelOutput 元数据，不再经 RL 动作空间。
"""

import numpy as np


class ActionValidator:
    """2 维动作约束校验器 (对齐下游 v2.15)。

    动作维度: [p_ref, k_droop]
    """

    P_BATT_MAX: float = 50.0          # kW
    K_DROOP_MIN: float = 0.0          # kW/V
    K_DROOP_MAX: float = 30.0         # kW/V
    DELTA_P_MAX: float = 50.0         # kW/步, ACT-01
    DELTA_K_DROOP_MAX: float = 10.0   # kW/V/步, ACT-02

    def __init__(self, p_batt_max: float = 50.0,
                 k_droop_min: float = 0.0, k_droop_max: float = 30.0,
                 delta_p_max: float = 50.0,
                 delta_k_droop_max: float = 10.0):
        """初始化动作约束参数。

        Args:
            p_batt_max: 电池最大充放电功率 (kW)
            k_droop_min: 下垂系数下限 (kW/V)
            k_droop_max: 下垂系数上限 (kW/V)
            delta_p_max: 电池变化率保护 (kW/步), ACT-01
            delta_k_droop_max: 下垂系数变化率保护 (kW/V/步), ACT-02
        """
        self.P_BATT_MAX = p_batt_max
        self.K_DROOP_MIN = k_droop_min
        self.K_DROOP_MAX = k_droop_max
        self.DELTA_P_MAX = delta_p_max
        self.DELTA_K_DROOP_MAX = delta_k_droop_max
        self.prev_p_ref: float = 0.0
        self.prev_k_droop: float = 0.0

    # ── 反归一化 / 归一化 ────────────────────────────────────

    def _denormalize(self, action_norm: np.ndarray
                     ) -> tuple[float, float]:
        """2 维动作反归一化到物理值。

        action_norm: [p_ref, k_droop]
        p_ref ∈ [-1, 1] → [-50, 50] kW
        k_droop ∈ [-1, 1] → [0, 30] kW/V
        """
        p_ref = float(action_norm[0] * self.P_BATT_MAX)
        k_range = (self.K_DROOP_MAX - self.K_DROOP_MIN) / 2.0
        k_center = (self.K_DROOP_MAX + self.K_DROOP_MIN) / 2.0
        k_droop = float(action_norm[1] * k_range + k_center)
        return p_ref, k_droop

    def _renormalize(self, p_ref: float, k_droop: float) -> np.ndarray:
        """物理值重新归一化到 2 维动作空间。"""
        k_range = (self.K_DROOP_MAX - self.K_DROOP_MIN) / 2.0
        k_center = (self.K_DROOP_MAX + self.K_DROOP_MIN) / 2.0
        return np.array([
            float(p_ref / self.P_BATT_MAX),
            float((k_droop - k_center) / k_range),
        ], dtype=np.float32)

    # ── 主校验入口 ─────────────────────────────────────────

    def validate(self, action_norm: np.ndarray,
                 dispatch_p: float | None = None,
                 ) -> tuple[np.ndarray, bool, dict[str, bool]]:
        """执行 4 条约束校验 (ACT-01~04, ACT-07)。

        Args:
            action_norm: 归一化动作 [p_ref, k_droop] (2维)
            dispatch_p: 调度有功指令 (kW), None 表示无调度

        Returns:
            (clamped_action_norm_2d, violated: bool, violations: dict)
        """
        p_ref, k_droop = self._denormalize(action_norm)
        violations: dict[str, bool] = {}

        # ACT-01: Δp_ref <= DeltaP_Max kW/步
        if abs(p_ref - self.prev_p_ref) > self.DELTA_P_MAX:
            p_ref = self.prev_p_ref + max(
                -self.DELTA_P_MAX,
                min(self.DELTA_P_MAX, p_ref - self.prev_p_ref))
            violations["ACT-01"] = True

        # ACT-02: Δk_droop <= DeltaK_Max kW/V/步
        if abs(k_droop - self.prev_k_droop) > self.DELTA_K_DROOP_MAX:
            k_droop = self.prev_k_droop + max(
                -self.DELTA_K_DROOP_MAX,
                min(self.DELTA_K_DROOP_MAX, k_droop - self.prev_k_droop))
            violations["ACT-02"] = True

        # ACT-03: p_ref ∈ [p_ref_min, p_ref_max]
        if p_ref < -self.P_BATT_MAX:
            p_ref = -self.P_BATT_MAX
            violations["ACT-03"] = True
        elif p_ref > self.P_BATT_MAX:
            p_ref = self.P_BATT_MAX
            violations["ACT-03"] = True

        # ACT-04: k_droop ∈ [k_droop_min, k_droop_max]
        if k_droop < self.K_DROOP_MIN:
            k_droop = self.K_DROOP_MIN
            violations["ACT-04"] = True
        elif k_droop > self.K_DROOP_MAX:
            k_droop = self.K_DROOP_MAX
            violations["ACT-04"] = True

        # ACT-07: |p_ref| <= |dispatch_p| (调度权限约束)
        if dispatch_p is not None and abs(dispatch_p) > 1e-6:
            limit = abs(dispatch_p)
            if abs(p_ref) > limit:
                p_ref = max(-limit, min(limit, p_ref))
                violations["ACT-07"] = True

        # 更新历史
        self.prev_p_ref = p_ref
        self.prev_k_droop = k_droop

        clamped = self._renormalize(p_ref, k_droop)
        return clamped, bool(violations), violations

    def reset(self) -> None:
        """重置历史状态 (每个 episode 开始时调用)。"""
        self.prev_p_ref = 0.0
        self.prev_k_droop = 0.0
