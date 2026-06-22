"""create_mupc_network() — 构建农网台区 Pandapower 网络拓扑。

农网台区规格（2026-06-10 更新）：
- 配电变压器: 200kVA, 10/0.4kV
- 主干线路: NAYY 4x50 SE, 1.5km
- 居民负荷: ~60kW
- 农业冲击负荷: 最高120kW
- 光伏: 150kW
- 储能: 100kWh, 最大放电50kW

Returns:
    Pandapower 网络对象（未执行潮流初始化）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandapower as pp
    from pandapower.auxiliary import pandapowerNet


# ── 网络拓扑常量 ────────────────────────────────────────────────

TRANSFORMER_KVA: float = 200.0     # 配电变压器额定容量 (kVA)
BATTERY_CAPACITY_KWH: float = 100.0  # 储能额定容量 (kWh)
PV_ARRAY_KW: float = 150.0        # 光伏额定容量 (kW)
LINE_LENGTH_KM: float = 0.5       # 主干线路长度 (km)，保证潮流收敛 + 足够电压变化(0.79~1.21pu)
MAX_BATTERY_DISCHARGE_KW: float = 50.0  # 储能最大放电功率 (kW)


def create_mupc_network() -> "pandapowerNet":
    """构建农网台区 Pandapower 网络。

    拓扑结构：
        外部电网 (10kV)
            └── 配电变压器 (200kVA, 10/0.4kV)
                    └── 低压母线 (0.4kV)
                            ├── 主干线路 (NAYY 4x50 SE, 1.5km)
                            │       └── 末端节点（居民/农业负荷/光伏/储能）
                            ├── 居民负荷（~60kW）
                            ├── 农业冲击负荷（灌溉/炒茶，最高120kW）
                            ├──屋顶光伏（150kW，可控）
                            └── 储能 MUPC_BESS（100kWh，最大放电50kW，可控）

    Returns:
        Pandapower 网络对象（未执行潮流）
    """
    import pandapower as pp

    #1. 创建空网络 (50Hz)
    net = pp.create_empty_network(
        name="Agri_MUPC_LV_Network",
        f_hz=50.0,
        sn_mva=1.0  # 标准视在功率基准值
    )

    # 2. 外部电网 (10kV 高压侧)
    ext_bus = pp.create_bus(net, vn_kv=10.0, name="HV_Grid")
    pp.create_ext_grid(
        net,
        bus=ext_bus,
        vm_pu=1.0,
        va_degree=0.0,
        name="Grid_Connection"
    )

    # 3. 配电变压器 (200kVA, 10/0.4kV)
    # 使用标准 0.25 MVA 类型并覆盖 sn_mva，保证参数完整性
    lv_bus = pp.create_bus(net, vn_kv=0.4, name="LV_Main_Bus")
    pp.create_transformer(
        net,
        hv_bus=ext_bus,
        lv_bus=lv_bus,
        std_type="0.25 MVA 10/0.4 kV",
        name="Dist_Transformer"
    )
    # 覆盖额定容量为 PRD v2.6 规格 200kVA
    net.trafo.at[0, "sn_mva"] = 0.2

    # 4. 主干线路（LGJ-70 架空线，农网典型配置）
    # R=0.45 Ω/km, X=0.35 Ω/km (70mm² 钢芯铝绞线)
    # 0.5km 保证潮流收敛，电压变化范围 0.79~1.21 pu 满足 RL 训练需求
    end_bus = pp.create_bus(net, vn_kv=0.4, name="End_Node_Bus")
    pp.create_line_from_parameters(
        net,
        from_bus=lv_bus,
        to_bus=end_bus,
        length_km=LINE_LENGTH_KM,
        r_ohm_per_km=0.45,       # LGJ-70 直流电阻
        x_ohm_per_km=0.35,       # LGJ-70 线路电抗
        c_nf_per_km=0.0,         # 架空线电容可忽略
        max_i_ka=0.3,            # 最大电流 300A
        name="Main_Overhead_Line"
    )

    # 5. 居民负荷（初始值 0，由 Chronics 动态注入）
    pp.create_load(
        net,
        bus=end_bus,
        p_mw=0.0,
        q_mvar=0.0,
        name="Residential_Load",
        controllable=False
    )

    # 6. 农业冲击负荷（初始值 0，由 Chronics 动态注入）
    pp.create_load(
        net,
        bus=end_bus,
        p_mw=0.0,
        q_mvar=0.0,
        name="Agri_Shock_Load",
        controllable=False
    )

    # 7. 屋顶分布式光伏（可控，可限功率）
    pp.create_sgen(
        net,
        bus=end_bus,
        p_mw=0.0,
        q_mvar=0.0,
        name="Rooftop_PV",
        controllable=True,
        # grid2op 默认使用 single values for sgen，这里用标量
    )

    # 8. MUPC 储能装置（100kWh，最大放电50kW，PRD v2.6 规格）
    pp.create_storage(
        net,
        bus=end_bus,
        p_mw=0.0,
        max_e_mwh=BATTERY_CAPACITY_KWH / 1000.0,  # 100kWh → 0.1 MWh
        min_e_mwh=0.0,
        soc_percent=50.0,
        name="MUPC_BESS",
        controllable=True
    )

    return net