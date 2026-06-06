"""
动作约束校验器 — 5 条约束规则，与 MUPC AI 引擎 ActionValidator 一致。

约束规则 (来自 MUPC AI 引擎 PRD 5.4 节):
  ACT-01: |Δp_batt| ≤ 50 kW/步
  ACT-02: |Δq_batt| ≤ 30 kVar/步
  ACT-03: sqrt(p_batt² + q_batt²) ≤ 500 kVA
  ACT-04: pv_limit ≥ 0.1
  ACT-05: |p_batt| ≤ |dispatch_p| (当调度指令有效时)
"""

import math
from typing import Any


class ActionValidator:
    """顺序执行 5 条约束规则, 违反时 clamp 并记录。"""

    P_BATT_MAX: float = 500.0     # kW
    Q_BATT_MAX: float = 300.0     # kVar
    LOAD_SHED_MAX: float = 500.0  # kW
    S_MAX: float = 500.0          # kVA 功率圆
    DELTA_P_MAX: float = 50.0     # kW/步
    DELTA_Q_MAX: float = 30.0     # kVar/步
    PV_LIMIT_MIN: float = 0.1

    def __init__(self):
        self.prev_p_batt: float = 0.0
        self.prev_q_batt: float = 0.0

    # ── 反归一化 / 归一化 ────────────────────────────────────

    def _denormalize(self, action_norm) -> tuple[float, float, float, float]:
        """动作反归一化到物理值。

        action_norm: [p_batt_norm, q_batt_norm, load_shed_norm, pv_limit_norm]
        p_batt_norm, q_batt_norm ∈ [-1, 1]
        load_shed_norm ∈ [0, 1]
        pv_limit_norm ∈ [0, 1]

        Returns: (p_batt_kW, q_batt_kVar, load_shed_kW, pv_limit_ratio)
        """
        p = action_norm[0] * self.P_BATT_MAX
        q = action_norm[1] * self.Q_BATT_MAX
        ls = action_norm[2] * self.LOAD_SHED_MAX
        pv = action_norm[3]               # 本身已是 [0,1]
        return p, q, ls, pv

    def _renormalize(self, p: float, q: float, ls: float, pv: float) -> tuple:
        """物理值重新归一化到 action space。"""
        return (
            float(p / self.P_BATT_MAX),
            float(q / self.Q_BATT_MAX),
            float(ls / self.LOAD_SHED_MAX),
            float(pv),
        )

    # ── 主校验入口 ─────────────────────────────────────────

    def validate(self, action_norm, dispatch_p: float | None = None
                 ) -> tuple[tuple[float, float, float, float], bool, dict[str, bool]]:
        """执行 5 条约束校验。

        Args:
            action_norm: 归一化动作 [p_norm, q_norm, ls_norm, pv_norm]
            dispatch_p: 调度有功指令 (kW), None 表示无调度

        Returns:
            (clamped_action_norm, violated: bool, violations: dict)
        """
        p, q, ls, pv = self._denormalize(action_norm)
        violations: dict[str, bool] = {}

        # ACT-01: Δp ≤ 50 kW
        if abs(p - self.prev_p_batt) > self.DELTA_P_MAX:
            p = self.prev_p_batt + max(-self.DELTA_P_MAX,
                                       min(self.DELTA_P_MAX, p - self.prev_p_batt))
            violations["ACT-01"] = True

        # ACT-02: Δq ≤ 30 kVar
        if abs(q - self.prev_q_batt) > self.DELTA_Q_MAX:
            q = self.prev_q_batt + max(-self.DELTA_Q_MAX,
                                       min(self.DELTA_Q_MAX, q - self.prev_q_batt))
            violations["ACT-02"] = True

        # ACT-03: 功率圆限制
        s = math.sqrt(p * p + q * q)
        if s > self.S_MAX:
            scale = self.S_MAX / s
            p *= scale
            q *= scale
            violations["ACT-03"] = True

        # ACT-04: pv_limit ≥ 0.1
        if pv < self.PV_LIMIT_MIN:
            pv = self.PV_LIMIT_MIN
            violations["ACT-04"] = True

        # ACT-05: 调度指令约束
        if dispatch_p is not None and abs(dispatch_p) > 1e-6:
            limit = abs(dispatch_p)
            if abs(p) > limit:
                p = max(-limit, min(limit, p))
                violations["ACT-05"] = True

        # 更新历史
        self.prev_p_batt = p
        self.prev_q_batt = q

        clamped = self._renormalize(p, q, ls, pv)
        return clamped, bool(violations), violations

    def reset(self) -> None:
        """重置历史状态 (每个 episode 开始时调用)。"""
        self.prev_p_batt = 0.0
        self.prev_q_batt = 0.0
