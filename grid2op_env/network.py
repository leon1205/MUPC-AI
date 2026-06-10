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
LINE_LENGTH_KM: float = 1.5       # 主干线路长度 (km)
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
    lv_bus = pp.create_bus(net, vn_kv=0.4, name="LV_Main_Bus")
    pp.create_transformer(
        net,
        hv_bus=ext_bus,
        lv_bus=lv_bus,
        std_type="0.25 MVA 10/0.4 kV",
        name="Dist_Transformer"
    )

    # 4. 主干线路（农网高阻抗架空线，模拟长距离电压跌落）
    # 使用 NAYY 4x50 SE 电缆标准类型（参考 docs/MUPC/仿真环境.md）
    # 如果标准类型不存在，使用参数化方式创建线路
    end_bus = pp.create_bus(net, vn_kv=0.4, name="End_Node_Bus")
    try:
        # 尝试使用标准类型
        pp.create_line(
            net,
            from_bus=lv_bus,
            to_bus=end_bus,
            length_km=LINE_LENGTH_KM,
            std_type="NAYY 4x50 SE",
            name="Main_Overhead_Line"
        )
    except Exception:
        # 降级：使用参数化方式创建线路（r=0.534 ohm/km, x=0.08 ohm/km, c=0）
        pp.create_line_from_parameters(
            net,
            from_bus=lv_bus,
            to_bus=end_bus,
            length_km=LINE_LENGTH_KM,
            r_ohm_per_km=0.534,  # NAYY 4x50 SE 的电阻
            x_ohm_per_km=0.08,    # NAYY 4x50 SE 的电抗
            c_nf_per_km=0.0,      # 电缆电容
            max_i_ka=0.3,          # 最大电流 300A
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

    # 8. MUPC 储能装置（100kWh，最大放电50kW）
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