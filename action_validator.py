"""
动作约束校验器 — 3 条约束规则，对应 MUPC AI 引擎 PRD v2.6 分层控制架构。

约束规则 (来自 MUPC AI 引擎 PRD 5.4 节 v2.6):
  ACT-01: |Δp_batt| ≤ 50 kW/步
  ACT-03: sqrt(p² + q_real²) ≤ 500 kVA（Q 由实时控制闭环，此约束主要限制 P）
  ACT-04: |pv_limit| ≤ 1.0（光伏限功率比例，v2.6 恢复）
  ACT-05: |p_batt| ≤ |dispatch_p| (当调度指令有效时)

Q_batt 由实时控制核心闭环处理:
  - Q_batt: 由实时电压调节器闭环控制
"""

import math
from typing import Any
import numpy as np


class ActionValidator:
    """顺序执行 4 条约束规则 (ACT-01, ACT-03, ACT-04, ACT-05), 违反时 clamp 并记录。"""

    P_BATT_MAX: float = 50.0      # kW（匹配电池最大放电功率）
    S_MAX: float = 200.0          # kVA (功率圆上限，匹配变压器 200kVA)
    LOAD_SHED_MAX: float = 60.0   # kW（匹配负荷峰值）
    DELTA_P_MAX: float = 50.0     # kW/步（电池变化率保护）

    def __init__(self):
        self.prev_p_batt: float = 0.0

    # ── 反归一化 / 归一化 ────────────────────────────────────

    def _denormalize(self, action_norm) -> tuple[float, float, float]:
        """动作反归一化到物理值 (3维: P_batt + Load_shedding + Pv_limit).

        action_norm: [p_batt_norm, load_shed_norm, pv_limit_norm]
        p_batt_norm ∈ [-1, 1] → [-50, 50] kW（匹配电池最大充放电功率）
        load_shed_norm ∈ [0, 1] → [0, 60] kW（匹配负荷峰值）
        pv_limit_norm ∈ [0, 1] → [0, 1] (无量纲比例)
        """
        p = action_norm[0] * self.P_BATT_MAX
        ls = action_norm[1] * self.LOAD_SHED_MAX
        pv_limit = action_norm[2] * 1.0  # 已经是 [0, 1] 范围
        return p, ls, pv_limit

    def _renormalize(self, p: float, ls: float, pv_limit: float) -> tuple:
        """物理值重新归一化到 action space。"""
        return (
            float(p / self.P_BATT_MAX),
            float(ls / self.LOAD_SHED_MAX),
            float(pv_limit),  # 无量纲，直接返回
        )

    # ── 主校验入口 ─────────────────────────────────────────

    def validate(self, action_norm, dispatch_p: float | None = None,
                 q_batt_real: float = 0.0
                 ) -> tuple[tuple[float, float, float], bool, dict[str, bool]]:
        """执行 4 条约束校验。

        Args:
            action_norm: 归一化动作 [p_batt_norm, load_shed_norm, pv_limit_norm] (3维)
            dispatch_p: 调度有功指令 (kW), None 表示无调度
            q_batt_real: 实时控制核心闭环的 Q 值 (kVar)，用于 ACT-03 功率圆校验

        Returns:
            (clamped_action_norm_3d, violated: bool, violations: dict)
        """
        p, ls, pv_limit = self._denormalize(action_norm)
        violations: dict[str, bool] = {}

        # ACT-01: Δp ≤ 50 kW/步 (电池变化率保护)
        if abs(p - self.prev_p_batt) > self.DELTA_P_MAX:
            p = self.prev_p_batt + max(-self.DELTA_P_MAX,
                                       min(self.DELTA_P_MAX, p - self.prev_p_batt))
            violations["ACT-01"] = True

        # ACT-03: 功率圆约束 sqrt(p² + q_real²) ≤ 500 kVA
        # Q 由实时控制闭环给出，此约束确保 P 不会单独触发越限
        s = math.sqrt(p**2 + q_batt_real**2)
        if s > self.S_MAX:
            scale = self.S_MAX / s
            p *= scale
            violations["ACT-03"] = True

        # ACT-04: pv_limit ∈ [0, 1] (光伏限功率比例约束)
        pv_limit = float(np.clip(pv_limit, 0.0, 1.0))
        if pv_limit != action_norm[2]:
            violations["ACT-04"] = True

        # ACT-05: 调度指令约束
        if dispatch_p is not None and abs(dispatch_p) > 1e-6:
            limit = abs(dispatch_p)
            if abs(p) > limit:
                p = max(-limit, min(limit, p))
                violations["ACT-05"] = True

        # 更新历史
        self.prev_p_batt = p

        clamped = self._renormalize(p, ls, pv_limit)
        return clamped, bool(violations), violations

    def reset(self) -> None:
        """重置历史状态 (每个 episode 开始时调用)。"""
        self.prev_p_batt = 0.0
