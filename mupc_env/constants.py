"""
MUPC 环境物理常数与配置 (提取自 mupc_env.py)

物理常量对齐 CLAUDE.md 规格和 config_manager.py PhysicalConfig。
"""

# ═══════════════════════════════════════════════════════════════
# 物理常量 (SAFETY: 以下常量涉及硬件安全边界, 修改前请评审)
# ═══════════════════════════════════════════════════════════════

TRANSFORMER_KVA = 200.0          # 变压器额定容量 (kVA) — 对齐规格
BATTERY_CAPACITY_KWH = 100.0    # 电池容量 (kWh) — 对齐规格
P_BATT_MAX_KW = 50.0             # 最大充放电功率 (kW) — 对齐规格
Q_BATT_MAX_KVAR = 300.0          # 最大无功输出 (kVar)
LOAD_SHED_MAX_KW = 60.0          # 最大切负荷 (kW) — 对齐规格
PV_ARRAY_KW = 150.0              # 光伏容量 (kW) — 对齐规格
LOAD_PEAK_KW = 60.0              # 负荷峰值 (kW) — 对齐规格

# 下垂控制常量 (对齐下游 v2.13)
K_DROOP_MIN = 0.0                # 下垂系数下限 (kW/V)
K_DROOP_MAX = 30.0               # 下垂系数上限 (kW/V)
K_DROOP_RAMP_LIMIT = 10.0        # 下垂系数变化率限制 (kW/V/步)，ACT-02
P_REF_RAMP_LIMIT_KW = 50.0       # p_ref 变化率限制 (kW/步)，ACT-01

SOC_MIN = 0.10                   # SAFETY: SOC 下限硬约束
SOC_MAX = 0.90                   # SAFETY: SOC 上限硬约束
OVERLOAD_THRESHOLD = 0.85        # 过载阈值
DT_HOURS = 0.25                  # 时间步长 (15 分钟)
LOAD_PF = 0.90                   # 负荷功率因数 cosφ

CONTRACT_DEMAND_KW = 300.0       # 合同需量 (kW)
GRID_EMISSION_FACTOR = 0.581    # kg CO2/kWh
EPISODE_LENGTH = 96              # 1 天 = 96 步 x 15 分钟

# ═══════════════════════════════════════════════════════════════
# 奖励阈值配置 (v2.5) — 对齐 MUPC AI 引擎 PRD v2.5
# ═══════════════════════════════════════════════════════════════

VOLTAGE_DEADBAND = 0.05          # 电压死区 ±5%
Q_MARGIN_THRESHOLD = 0.10       # 实时模块无功耗尽阈值 (10%)
VOLTAGE_HIGH_LIMIT = 1.05        # 弃光前置电压阈值 (p.u.)
SOC_CRITICAL = 0.10              # SOC 极低保护阈值
VOLTAGE_PENALTY_HIGH = 2.0      # 高电压侧惩罚系数 (光伏超发)
VOLTAGE_PENALTY_LOW = 1.0       # 低电压侧惩罚系数 (灌溉/炒茶/空调)

# ═══════════════════════════════════════════════════════════════
# 奖励权重映射
# ═══════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS: dict[str, list[float]] = {
    # v2.13 MODE-01: w1~w13
    # w1(光伏消纳), w2(电池损耗), w3(过载), w4(P-Q协同), w5(变化率),
    # w6(电压斜率), w7(下垂平滑), w8(安全覆盖), w9(过载预警), w10(SOC预警),
    # w11(SOC均衡), w12(冲击预备度), w13(状态改善率)
    "MODE-01": [1.0, 0.5, 2.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    "MODE-02": [1.0, 1.0, 2.0],       # w1(价差), w2(电池), w3(过载)
    "MODE-03": [1.0, 0.5],            # w1(需量减免), w2(舒适度)
    "MODE-04": [1.0, 2.0, 1.0],       # w1(辅助收益), w2(响应精度), w3(延迟)
    "MODE-05": [1.0, 1.0],            # w1(绿电), w2(碳减排)
}

MODE_ID_MAP: dict[str, float] = {
    "MODE-01": 0.0, "MODE-02": 0.25, "MODE-03": 0.5,
    "MODE-04": 0.75, "MODE-05": 1.0,
}

ALL_MODES = ["MODE-01", "MODE-02", "MODE-03", "MODE-04", "MODE-05"]

# ═══════════════════════════════════════════════════════════════
# 归一化边界 (用于 observation.normalize_obs)
# ═══════════════════════════════════════════════════════════════

NORM_SOC = (0.0, 1.0)
NORM_PV_POWER = (0.0, PV_ARRAY_KW)          # 光伏功率: [0, 150] kW
NORM_LOAD_POWER = (0.0, LOAD_PEAK_KW)        # 负荷功率: [0, 60] kW
NORM_GRID_POWER = (-TRANSFORMER_KVA, TRANSFORMER_KVA)  # 电网功率: [-200, 200] kW
NORM_VOLTAGE = (0.85, 1.15)                  # 相电压标幺值
NORM_BATTERY_POWER = (-P_BATT_MAX_KW, P_BATT_MAX_KW)   # 电池功率: [-50, 50] kW
NORM_PRICE = (0.0, 1.5)                      # 电价: [0, 1.5] yuan/kWh
NORM_TARIFF = (0.0, 3.0)                     # 时段费率系数
NORM_DEMAND = (0.0, 500.0)                   # 需量值: [0, 500] kW
NORM_IRRADIANCE = (0.0, 1500.0)              # 辐照度: [0, 1500] W/m²
NORM_TEMPERATURE = (-20.0, 60.0)             # 温度: [-20, 60] °C
NORM_DISPATCH = (-TRANSFORMER_KVA, TRANSFORMER_KVA)  # 调度指令: [-200, 200] kW
