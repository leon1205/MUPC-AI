"""
动作约束校验器 — 4+1 条约束规则，对齐下游 MUPC AI 引擎 v2.15。

约束规则 (v2.17 完全对齐下游 Rust action_validator.rs):
  ACT-01: |Δp_ref| <= 50 kW/步
  ACT-02: |Δk_droop| <= 10 kW/V/步
  ACT-03: √(p_ref² + k_droop²) <= 200 kVA (视在功率圆, 对齐下游 S-circle)
  ACT-04: k_droop ∈ [-100, 100] kW/V (最终值域 clamp)
  ACT-05: |p_ref| <= |dispatch_p| (调度权限约束)

v2.17 变更:
  - k_droop 范围 [0,30] → [-100,100] (对齐下游 Dual 模式默认值)
  - ACT-03 从简单 p_ref clamp 改为 S-circle (对齐下游标准模式)
  - 新增 MAX_APPARENT_POWER_KVA = 200.0

v3.1 对齐:
  - ACT-02 阈值恢复为 10 kW/V/步 (对齐下游 PRD §6.5)
"""

import numpy as np


class ActionValidator:
    """2 维动作约束校验器 (v2.17 对齐下游 v2.15)。

    动作维度: [p_ref, k_droop]
    p_ref > 0 = 放电, p_ref < 0 = 充电 (v2.15 符号约定)
    """

    P_BATT_MAX: float = 50.0              # kW
    K_DROOP_MIN: float = -100.0           # kW/V (v2.17: 对齐下游 Dual 模式)
    K_DROOP_MAX: float = 100.0            # kW/V (v2.17: 对齐下游 Dual 模式)
    DELTA_P_MAX: float = 50.0             # kW/步, ACT-01
    DELTA_K_DROOP_MAX: float = 10.0       # kW/V/步, ACT-02 对齐下游 PRD §6.5
    MAX_APPARENT_POWER_KVA: float = 200.0  # kVA, ACT-03 S-circle

    def __init__(self, p_batt_max: float = 50.0,
                 k_droop_min: float = -100.0, k_droop_max: float = 100.0,
                 delta_p_max: float = 50.0,
                 delta_k_droop_max: float = 10.0,
                 max_apparent_power_kva: float = 200.0):
        """初始化动作约束参数。

        Args:
            p_batt_max: 电池最大充放电功率 (kW)
            k_droop_min: 下垂系数下限 (kW/V), v2.17 默认 -100
            k_droop_max: 下垂系数上限 (kW/V), v2.17 默认 100
            delta_p_max: 电池变化率保护 (kW/步), ACT-01
            delta_k_droop_max: 下垂系数变化率保护 (kW/V/步), ACT-02 对齐下游 PRD §6.5
            max_apparent_power_kva: 视在功率上限 (kVA), ACT-03 S-circle
        """
        self.P_BATT_MAX = p_batt_max
        self.K_DROOP_MIN = k_droop_min
        self.K_DROOP_MAX = k_droop_max
        self.DELTA_P_MAX = delta_p_max
        self.DELTA_K_DROOP_MAX = delta_k_droop_max
        self.MAX_APPARENT_POWER_KVA = max_apparent_power_kva
        self.prev_p_ref: float = 0.0
        self.prev_k_droop: float = 0.0

    # ── 反归一化 / 归一化 ────────────────────────────────────

    def _denormalize(self, action_norm: np.ndarray
                     ) -> tuple[float, float]:
        """2 维动作反归一化到物理值。

        action_norm: [p_ref, k_droop]
        p_ref ∈ [-1, 1] → [-50, 50] kW
        k_droop ∈ [-1, 1] → [-100, 100] kW/V (v2.17)
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
        """执行 4+1 条约束校验 (v2.17 对齐下游)。

        Args:
            action_norm: 归一化动作 [p_ref, k_droop] (2维)
            dispatch_p: 调度有功指令 (kW), None 表示无调度

        Returns:
            (clamped_action_norm_2d, violated: bool, violations: dict)
        """
        p_ref, k_droop = self._denormalize(action_norm)
        violations: dict[str, bool] = {}

        # ACT-01: Δp_ref <= 50 kW/步
        if abs(p_ref - self.prev_p_ref) > self.DELTA_P_MAX:
            p_ref = self.prev_p_ref + max(
                -self.DELTA_P_MAX,
                min(self.DELTA_P_MAX, p_ref - self.prev_p_ref))
            violations["ACT-01"] = True

        # ACT-02: Δk_droop <= 10 kW/V/步
        if abs(k_droop - self.prev_k_droop) > self.DELTA_K_DROOP_MAX:
            k_droop = self.prev_k_droop + max(
                -self.DELTA_K_DROOP_MAX,
                min(self.DELTA_K_DROOP_MAX, k_droop - self.prev_k_droop))
            violations["ACT-02"] = True

        # ACT-03: √(p_ref² + k_droop²) <= 200 kVA (v2.17: S-circle)
        s = np.sqrt(p_ref ** 2 + k_droop ** 2)
        if s > self.MAX_APPARENT_POWER_KVA:
            scale = self.MAX_APPARENT_POWER_KVA / s
            p_ref *= scale
            k_droop *= scale
            violations["ACT-03"] = True

        # ACT-04: k_droop ∈ [-100, 100] kW/V (v2.17: [0,30]→[-100,100])
        if k_droop < self.K_DROOP_MIN:
            k_droop = self.K_DROOP_MIN
            violations["ACT-04"] = True
        elif k_droop > self.K_DROOP_MAX:
            k_droop = self.K_DROOP_MAX
            violations["ACT-04"] = True

        # ACT-05: |p_ref| <= |dispatch_p| (调度权限约束)
        if dispatch_p is not None and abs(dispatch_p) > 1e-6:
            limit = abs(dispatch_p)
            if abs(p_ref) > limit:
                p_ref = max(-limit, min(limit, p_ref))
                violations["ACT-05"] = True

        # 最终安全 clamp: p_ref ∈ [-50, 50] (对齐下游最终值域 clamp)
        p_ref = float(np.clip(p_ref, -self.P_BATT_MAX, self.P_BATT_MAX))

        # 更新历史
        self.prev_p_ref = p_ref
        self.prev_k_droop = k_droop

        clamped = self._renormalize(p_ref, k_droop)
        return clamped, bool(violations), violations

    def reset(self) -> None:
        """重置历史状态 (每个 episode 开始时调用)。"""
        self.prev_p_ref = 0.0
        self.prev_k_droop = 0.0
