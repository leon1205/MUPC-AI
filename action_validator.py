"""
动作约束校验器 — 7 条约束规则，对齐下游 MUPC AI 引擎 PRD v2.13。

约束规则 (来自 MUPC AI 引擎 PRD v2.13 Section 6.5):
  ACT-01: |Δp_ref| <= 50 kW/步
  ACT-02: |Δk_droop| <= 10 kW/V/步  (v2.13 新增)
  ACT-03: p_ref ∈ [p_ref_min, p_ref_max]      → [-50, 50] kW
  ACT-04: k_droop ∈ [k_droop_min, k_droop_max] → [0, 30] kW/V
  ACT-05: load_shedding ∈ [0.0, max_load_shedding]
  ACT-06: pv_limit ∈ [pv_limit_min, 1.0]
  ACT-07: |p_ref| <= |dispatch_p| (当调度指令有效时)

Q_batt 由实时控制核心闭环处理，不经过 RL 动作空间。
"""

import math
from typing import Any
import numpy as np


class ActionValidator:
    """统一 5 维动作约束校验器 (对齐下游 v2.13)。

    顺序执行 7 条约束规则，违反时 clamp 并记录。
    动作维度: [p_ref, k_droop, load_shedding, pv_limit, confidence]
    """

    P_BATT_MAX: float = 50.0         # kW，匹配电池最大放电功率
    K_DROOP_MIN: float = 0.0         # kW/V，对齐下游 [0, 30]
    K_DROOP_MAX: float = 30.0        # kW/V
    LOAD_SHED_MAX: float = 60.0      # kW，匹配负荷峰值
    DELTA_P_MAX: float = 50.0        # kW/步，电池变化率保护
    DELTA_K_DROOP_MAX: float = 10.0  # kW/V/步，ACT-02

    def __init__(self, p_batt_max: float = 50.0,
                 k_droop_min: float = 0.0, k_droop_max: float = 30.0,
                 load_shed_max: float = 60.0, delta_p_max: float = 50.0,
                 delta_k_droop_max: float = 10.0, pv_limit_min: float = 0.0):
        """初始化动作约束参数（支持配置文件注入）。

        Args:
            p_batt_max: 电池最大充放电功率 (kW)
            k_droop_min: 下垂系数下限 (kW/V)，下游 v2.13 = 0.0
            k_droop_max: 下垂系数上限 (kW/V)，下游 v2.13 = 30.0
            load_shed_max: 最大切负荷 (kW)
            delta_p_max: 电池变化率保护 (kW/步)
            delta_k_droop_max: 下垂系数变化率保护 (kW/V/步)，ACT-02
            pv_limit_min: 光伏限功率下限
        """
        self.P_BATT_MAX = p_batt_max
        self.K_DROOP_MIN = k_droop_min
        self.K_DROOP_MAX = k_droop_max
        self.LOAD_SHED_MAX = load_shed_max
        self.DELTA_P_MAX = delta_p_max
        self.DELTA_K_DROOP_MAX = delta_k_droop_max
        self.PV_LIMIT_MIN = pv_limit_min
        self.prev_p_ref: float = 0.0
        self.prev_k_droop: float = 0.0

    # ── 反归一化 / 归一化 ────────────────────────────────────

    def _denormalize(self, action_norm: np.ndarray
                     ) -> tuple[float, float, float, float, float]:
        """5 维动作反归一化到物理值。

        action_norm: [p_ref, k_droop, load_shedding, pv_limit, confidence]
        p_ref_norm ∈ [-1, 1] → [-50, 50] kW
        k_droop_norm ∈ [-1, 1] → [0, 30] kW/V (线性映射)
        load_shed_norm ∈ [0, 1] → [0, 60] kW
        pv_limit_norm ∈ [0, 1] → [0, 1]
        confidence_norm ∈ [0, 1] → [0, 1]
        """
        p_ref = float(action_norm[0] * self.P_BATT_MAX)
        k_range = (self.K_DROOP_MAX - self.K_DROOP_MIN) / 2.0
        k_center = (self.K_DROOP_MAX + self.K_DROOP_MIN) / 2.0
        k_droop = float(action_norm[1] * k_range + k_center)
        load_shed = float(action_norm[2] * self.LOAD_SHED_MAX)
        pv_limit = float(action_norm[3])
        confidence = float(action_norm[4])
        return p_ref, k_droop, load_shed, pv_limit, confidence

    def _renormalize(self, p_ref: float, k_droop: float,
                     load_shed: float, pv_limit: float,
                     confidence: float) -> np.ndarray:
        """物理值重新归一化到 5 维动作空间。"""
        k_range = (self.K_DROOP_MAX - self.K_DROOP_MIN) / 2.0
        k_center = (self.K_DROOP_MAX + self.K_DROOP_MIN) / 2.0
        return np.array([
            float(p_ref / self.P_BATT_MAX),
            float((k_droop - k_center) / k_range),
            float(load_shed / self.LOAD_SHED_MAX),
            float(pv_limit),
            float(confidence),
        ], dtype=np.float32)

    # ── 主校验入口 ─────────────────────────────────────────

    def validate(self, action_norm: np.ndarray,
                 dispatch_p: float | None = None,
                 ) -> tuple[np.ndarray, bool, dict[str, bool]]:
        """执行 7 条约束校验 (ACT-01~07)。

        Args:
            action_norm: 归一化动作 [p_ref, k_droop, load_shedding,
                         pv_limit, confidence] (5维)
            dispatch_p: 调度有功指令 (kW), None 表示无调度

        Returns:
            (clamped_action_norm_5d, violated: bool, violations: dict)
        """
        p_ref, k_droop, load_shed, pv_limit, confidence = \
            self._denormalize(action_norm)
        violations: dict[str, bool] = {}

        # ACT-01: Δp_ref <= 50 kW/步 (p_ref 变化率约束)
        if abs(p_ref - self.prev_p_ref) > self.DELTA_P_MAX:
            p_ref = self.prev_p_ref + max(
                -self.DELTA_P_MAX,
                min(self.DELTA_P_MAX, p_ref - self.prev_p_ref))
            violations["ACT-01"] = True

        # ACT-02: Δk_droop <= 10 kW/V/步 (k_droop 变化率约束, v2.13 新增)
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

        # ACT-05: load_shedding ∈ [0, max_load_shedding]
        if load_shed < 0.0:
            load_shed = 0.0
            violations["ACT-05"] = True
        elif load_shed > self.LOAD_SHED_MAX:
            load_shed = self.LOAD_SHED_MAX
            violations["ACT-05"] = True

        # ACT-06: pv_limit ∈ [pv_limit_min, 1.0]
        if pv_limit < self.PV_LIMIT_MIN:
            pv_limit = self.PV_LIMIT_MIN
            violations["ACT-06"] = True
        elif pv_limit > 1.0:
            pv_limit = 1.0
            violations["ACT-06"] = True

        # ACT-07: |p_ref| <= |dispatch_p| (调度指令权限约束)
        if dispatch_p is not None and abs(dispatch_p) > 1e-6:
            limit = abs(dispatch_p)
            if abs(p_ref) > limit:
                p_ref = max(-limit, min(limit, p_ref))
                violations["ACT-07"] = True

        # confidence 仅通过，不约束
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # 更新历史
        self.prev_p_ref = p_ref
        self.prev_k_droop = k_droop

        clamped = self._renormalize(
            p_ref, k_droop, load_shed, pv_limit, confidence)
        return clamped, bool(violations), violations

    def reset(self) -> None:
        """重置历史状态 (每个 episode 开始时调用)。"""
        self.prev_p_ref = 0.0
        self.prev_k_droop = 0.0
