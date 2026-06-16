"""
MUPC 三相电压简化线路模型 (Q-V 耦合)

提取自 mupc_env.py，S_BASE 引用 constants.TRANSFORMER_KVA 消除重复定义。
"""

import numpy as np

from .constants import TRANSFORMER_KVA as S_BASE


class VoltageSimulator:
    """三相电压简化线路模型 (Q-V 耦合)。"""

    K_P = 0.05            # 有功灵敏度 (p.u. / MW scaled)
    K_Q = 0.03            # 无功灵敏度 (p.u. / MVar scaled)
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
        dv = (self.K_P * p_net + self.K_Q * q_batt) / S_BASE
        va = prev_va + dv + np.random.normal(0, self.NOISE_STD)
        vb = prev_vb + dv + np.random.normal(0, self.NOISE_STD) + self.IMBALANCE
        vc = prev_vc + dv + np.random.normal(0, self.NOISE_STD) - self.IMBALANCE
        return (
            float(np.clip(va, self.V_MIN, self.V_MAX)),
            float(np.clip(vb, self.V_MIN, self.V_MAX)),
            float(np.clip(vc, self.V_MIN, self.V_MAX)),
        )
