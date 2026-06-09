# Grid2Op 电压仿真替换 — 技术设计文档

| 版本 | 日期 | 作者 | 状态 |
|------|------|------|------|
| v1.0 | 2026-06-09 | 架构师 | `[DESIGN_APPROVED]` |

**对应 PRD:** `docs/superpowers/specs/2026-06-09-Grid2Op电压仿真替换-PRD.md` v1.0 (`[REVIEWED: PASS]`)

---

## 目录

1. [技术选型与选型理由](#1-技术选型与选型理由)
2. [核心模块设计](#2-核心模块设计)
3. [数据流图](#3-数据流图)
4. [核心类函数接口设计](#4-核心类函数接口设计)
5. [SOC 双向同步机制](#5-soc-双向同步机制)
6. [文件结构预估](#6-文件结构预估)
7. [风险与解决措施（R-01~R-08）](#7-风险与解决措施r-01r-08)

---

## 1. 技术选型与选型理由

### 1.1 选型结论

**路线 A：组合模式（Composite Pattern）**

> `MupcEnv` 内部持有 `Grid2Op Env` 实例，Grid2Op 作为内部潮流计算引擎，不继承 Gymnasium 基类。

```
MupcEnv (外部: Gymnasium 接口)
  └── Grid2OpEnv (内部: 潮流计算引擎, LightSimBackend)
        ├── PandapowerNetwork (农网台区物理拓扑)
        └── NumpyChronics (时序数据注入)
```

### 1.2 选型理由

| 方案 | 描述 | 优点 | 缺点 | 选择 |
|------|------|------|------|------|
| 路线 A：组合模式 | MupcEnv 持有 Grid2Op 实例，外部保持 Gymnasium | 接口完全兼容，改动最小 | Grid2Op 需独立管理生命周期 | **采用** |
| 路线 B：继承模式 | MupcEnv 继承 Grid2Op 环境基类 | 直接利用 Grid2Op 生态 | Grid2Op 与 Gymnasium 基类冲突，适配成本高 | 放弃 |
| 路线 C：降级方案 | 保留原 VoltageSimulator，通过开关切换 | 改动最少 | 两种仿真引擎需同时维护，长期成本高 | 放弃 |

### 1.3 关键技术决策

**决策 1：LightSimBackend 而非 PandaPowerBackend**

Grid2Op 默认 `PandaPowerBackend` 调用 Pandapower Python API，每步耗时 10~50ms。
`LightSimBackend`（C++ 实现）速度提升 20~50 倍，API 兼容，完全满足 ≤50ms/步要求。

**决策 2：NumpyChronics 自定义类**

现有 `data` dict（来自 SmartDSLoader）为单相标量格式：
- `pv_power: np.ndarray (n_steps,)` — 光伏 kW
- `load_power: np.ndarray (n_steps,)` — 负荷 kW
- `solar_irradiance: np.ndarray (n_steps,)` — W/m2

Grid2Op 期望三相格式 `load_p_mw / load_q_mvar`（每相独立）。自定义 `NumpyChronics` 将单相标量展开为 Grid2Op 三相格式，同时注入光伏、负荷、气象数据。

**决策 3：三相潮流默认开启**

当前 `VoltageSimulator` 为单相等效模型（Q-V 耦合灵敏度系数）。替换为 `runpp_3ph` 三相潮流计算后，三相电压 `va/vb/vc` 均由 Pandapower 独立计算，不平衡度自然呈现，无需手动叠加噪声。

**决策 4：SOC 双向同步**

`MupcEnv._soc` 与 Grid2Op `storage.soc` 每步同步：
- `step()` 开始时：Grid2Op → MupcEnv（同步上一轮结果）
- `step()` 结束时：MupcEnv → Grid2Op（准备下一轮）

### 1.4 依赖版本约束

```
grid2op >= 1.9.0
pandapower >= 3.8.0
lightsim2grid >= 0.5.0  # 可选加速后端
numpy >= 1.21.0
```

---

## 2. 核心模块设计

### 2.1 NumpyChronics

**职责**：将 `data` dict（单相标量）转换为 Grid2Op 时序格式，驱动 Grid2Op 仿真。

```python
class NumpyChronics:
    """将 SmartDSLoader 的 data dict 转换为 Grid2Op 时序格式。

    data dict（单相标量） → Grid2Op 三相格式
    ├── pv_power (n_steps,) → sgen_p_mw (3相相同)
    ├── load_power (n_steps,) → load_p_mw (3相分解)
    ├── solar_irradiance → 环境变量
    └── temperature → 环境变量
    """

    def __init__(self, data: dict):
        """
        Args:
            data: SmartDSLoader.load_all() 返回的 data dict
        """
        self._data = data
        self._n_steps = data["n_steps"]
        self._current_idx = 0

    def initialize(self, initial_storage_soc: float) -> None:
        """初始化第一帧数据。

        Args:
            initial_storage_soc: 初始 SOC (0.0~1.0)，同步到 Grid2Op storage 元件
        """

    def load_next(self) -> dict[str, np.ndarray]:
        """返回下一帧数据（Grid2Op 格式）。

        Returns:
            {
                "load_p_mw":   (n_element, 3)  # 三相有功负荷
                "load_q_mvar": (n_element, 3)  # 三相无功负荷
                "sgen_p_mw":   (n_sgen,)       # 光伏有功
                "sgen_q_mvar": (n_sgen,)       # 光伏无功
            }
        """

    def reset(self, initial_storage_soc: float) -> None:
        """重置时序索引到 0（每个 episode 开始时调用）。"""
```

**三相展开逻辑**：

```python
def _expand_single_to_three_phase(self, scalar_value: float,
                                   phase_imbalance: float = 0.003
                                   ) -> np.ndarray:
    """将单相标量展开为三相数组，叠加轻微不平衡度。

    Args:
        scalar_value: 单相标量值 (kW)
        phase_imbalance: 三相不平衡度 (default 0.003 p.u.)
    Returns:
        三相数组 (3,)，单位同输入
    """
    base = scalar_value / 3.0  # 平均分配到三相
    phase_a = base * (1.0 + np.random.uniform(-phase_imbalance, phase_imbalance))
    phase_b = base * (1.0 + np.random.uniform(-phase_imbalance, phase_imbalance))
    phase_c = base * (1.0 + np.random.uniform(-phase_imbalance, phase_imbalance))
    return np.array([phase_a, phase_b, phase_c])
```

**时序推进**：`current_idx` 每步递增，超过 `n_steps` 时截断或循环。

### 2.2 Grid2OpPowerFlow

**职责**：封装 Grid2Op 环境生命周期，提供同步 SOC 和获取三相电压的接口。

```python
class Grid2OpPowerFlow:
    """Grid2Op 潮流计算引擎封装。

    使用 LightSimBackend（如不可用则降级到 PandaPowerBackend）。
    不继承 Gymnasium 基类，仅作为 MupcEnv 的内部计算组件。
    """

    def __init__(self, pandapower_net: "pp.auxiliary.pandapowerNet",
                 chronics: NumpyChronics,
                 storage_soc_init: float = 0.5):
        """
        Args:
            pandapower_net: Pandapower 网络拓扑
            chronics: NumpyChronics 实例
            storage_soc_init: 初始 SOC
        """

    def reset(self, initial_storage_soc: float) -> None:
        """重置 Grid2Op 环境到初始状态。"""

    def step(self, storage_p_mw: float, storage_q_mvar: float
             ) -> tuple[float, float, float, bool]:
        """执行一步潮流计算。

        Args:
            storage_p_mw: 储能有功 (MW +充电/-放电)
            storage_q_mvar: 储能无功 (MVar)

        Returns:
            (va, vb, vc, has_illegal)
            va/vb/vc: 三相电压标幺值 (p.u.)
            has_illegal: 是否有非法状态（电压/潮流越限）
        """

    def get_storage_soc(self) -> float:
        """获取当前 SOC (0.0~1.0)。"""

    def set_storage_soc(self, soc: float) -> None:
        """设置 SOC（同步 MupcEnv._soc 到 Grid2Op）。"""

    def get_storage_power(self) -> tuple[float, float]:
        """获取当前储能有功/无功出清值。"""
```

**潮流不收敛处理**（R-03 缓解）：

```python
def step(self, storage_p_mw, storage_q_mvar):
    # 尝试三相潮流
    success = self._run_powerflow_3ph(storage_p_mw, storage_q_mvar)

    if not success or self._has_nan_voltage():
        # 回退到上一时刻电压（保持安全值）
        va, vb, vc = self._prev_va, self._prev_vb, self._prev_vc
        has_illegal = True
    else:
        va, vb, vc = self._extract_voltages()
        self._prev_va, self._prev_vb, self._prev_vc = va, vb, vc

    return va, vb, vc, has_illegal
```

### 2.3 MupcEnv 适配层

**职责**：保持 Gymnasium 接口（`reset()`/`step()`）完全兼容，内部集成 Grid2Op 进行电压计算。

关键设计点：

1. **接口不变**：`reset()`/`step()` 签名和返回值与原实现 100% 兼容
2. **内部替换**：`VoltageSimulator` → `Grid2OpPowerFlow`
3. **SOC 同步**：每步在 `step()` 入口和出口进行双向同步
4. **降级支持**：Grid2Op 不可用时自动降级到 `VoltageSimulator`

```python
class MupcEnv(gym.Env if _GYM_AVAILABLE else _GymStubEnv):
    """MUPC 全状态 RL 环境 (v2.5 + Grid2Op 电压仿真).

    内部持有 Grid2OpPowerFlow 实例，三相电压由 Pandapower 潮流计算。
    """

    def __init__(self, data: dict, mode: str = "all",
                 lstm_predictor: Any = None,
                 reward_weights: dict[str, list[float]] | None = None,
                 use_grid2op: bool = True):
        # ... 原有初始化逻辑 ...

        # Grid2Op 电压仿真（可开关切换）
        if use_grid2op:
            from grid2op_env import Grid2OpPowerFlow, NumpyChronics, create_mupc_network
            self._grid2op_powerflow: Grid2OpPowerFlow | None = None
            self._use_grid2op = True
        else:
            self._use_grid2op = False
            self._voltage_sim = VoltageSimulator()

    def _init_grid2op(self):
        """延迟初始化 Grid2Op（首次 reset 前不创建）。"""
        net = create_mupc_network()
        chronics = NumpyChronics(self._data)
        self._grid2op_power_flow = Grid2OpPowerFlow(
            net, chronics, storage_soc_init=self._soc
        )

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # ... 原有 reset 逻辑 ...

        if self._use_grid2op:
            if self._grid2op_power_flow is None:
                self._init_grid2op()
            self._grid2op_power_flow.reset(initial_storage_soc=self._soc)

        # ... 其余 reset 逻辑 ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真，三相电压由 Grid2Op/Pandapower 计算。"""

        # 1. Grid2Op SOC 同步（step 开始时：Grid2Op → MupcEnv）
        if self._use_grid2op:
            grid_soc = self._grid2op_power_flow.get_storage_soc()
            self._soc = grid_soc  # 同步 Grid2Op 的 SOC 到 MupcEnv

        # 2. Q_batt 实时电压环（与原逻辑一致）
        v_prev = (self._va + self._vb + self._vc) / 3.0
        q_batt = float(np.clip(-K_Q_V * (v_prev - 1.0),
                               -Q_BATT_MAX_KVAR, Q_BATT_MAX_KVAR))
        self._q_realtime_margin = 1.0 - (abs(q_batt) / Q_BATT_MAX_KVAR)

        # 3. 动作约束校验（ACT-01/03/05）
        clamped, violated, violations = self._validator.validate(
            action, dispatch_p_use, q_batt_real=q_batt)

        # 4. 反归一化动作
        p_batt = clamped[0] * P_BATT_MAX_KW
        load_shed = clamped[1] * LOAD_SHED_MAX_KW

        # 5. 有效负荷与光伏
        p_load_raw = float(self._data["load_power"][self._step_idx])
        p_load_eff = max(0.0, p_load_raw - load_shed)
        p_pv_raw = float(self._data["pv_power"][self._step_idx])
        p_pv_eff = p_pv_raw

        # 6. SOC 更新（MupcEnv 侧计算）
        soc_raw = self._soc + (-p_batt * DT_HOURS) / BATTERY_CAPACITY_KWH
        soc_new = float(np.clip(soc_raw, SOC_MIN, SOC_MAX))
        soc_clipped = abs(soc_raw - soc_new) > 1e-9

        # 7. Grid2Op 潮流计算（替换 VoltageSimulator）
        if self._use_grid2op:
            # 转换为 Grid2Op 单位: kW → MW
            storage_p_mw = p_batt / 1000.0
            storage_q_mvar = q_batt / 1000.0  # kVar → MVar

            va, vb, vc, has_illegal = self._grid2op_power_flow.step(
                storage_p_mw, storage_q_mvar)

            # 同步 SOC 到 Grid2Op（step 结束时：MupcEnv → Grid2Op）
            self._grid2op_power_flow.set_storage_soc(soc_new)
        else:
            # 降级到原 VoltageSimulator
            p_net = p_pv_eff - p_load_eff + p_batt
            va, vb, vc = self._voltage_sim.step(
                p_net, float(q_batt), self._va, self._vb, self._vc)
            has_illegal = False

        # 8. 后续逻辑与原实现一致（电网功率、变压器负载率、奖励计算等）
        ...
```

### 2.4 Pandapower 网络拓扑

**create_mupc_network()**：构建农网台区物理拓扑，对齐 `docs/MUPC/仿真环境.md`。

```python
def create_mupc_network() -> "pp.auxiliary.pandapowerNet":
    """构建农网台区 Pandapower 网络。

    拓扑结构:
        外部电网 (10kV)
            └── 配电变压器 (500kVA, 10/0.4kV)
                    └── 低压母线 (0.4kV)
                            ├── 主干线路 (LGJ-70, 1.5km)
                            │       └── 末端节点（负荷/光伏/储能）
                            ├── 居民负荷（晚高峰）
                            ├── 农业冲击负荷（灌溉/炒茶）
                            ├── 光伏（200kW）
                            └── 储能 MUPC_BESS（200kWh）

    Returns:
        Pandapower 网络对象
    """
    import pandapower as pp
    import pandapower.net as n

    net = pp.create_empty_network(name="Agri_MUPC_LV_Network", f_hz=50.0)

    # 外部电网
    ext_bus = pp.create_bus(net, vn_kv=10.0, name="HV_Grid")
    pp.create_ext_grid(net, bus=ext_bus, vm_pu=1.0, va_degree=0)

    # 配电变压器 (500kVA)
    lv_bus = pp.create_bus(net, vn_kv=0.4, name="LV_Main_Bus")
    pp.create_transformer(net, hv_bus=ext_bus, lv_bus=lv_bus,
                          std_type="0.4 MVA 10/0.4 kV", name="Dist_Transformer")

    # 主干线路（农网高阻抗线路，模拟长距离导致的电压跌落）
    end_bus = pp.create_bus(net, vn_kv=0.4, name="End_Node_Bus")
    pp.create_line(net, from_bus=lv_bus, to_bus=end_bus, length_km=1.5,
                   std_type="NAYY 4x50 SE", name="Main_Overhead_Line")

    # 居民负荷（可控=False，数据由 Chronics 注入）
    pp.create_load(net, bus=end_bus, p_mw=0.0, q_mvar=0.0,
                   name="Residential_Load", controllable=False)

    # 农业冲击负荷
    pp.create_load(net, bus=end_bus, p_mw=0.0, q_mvar=0.0,
                   name="Agri_Shock_Load", controllable=False)

    # 分布式光伏（可控=True，可限功率）
    pp.create_sgen(net, bus=end_bus, p_mw=0.0, q_mvar=0.0,
                   name="Rooftop_PV", controllable=True)

    # MUPC 储能装置
    pp.create_storage(net, bus=end_bus,
                      p_mw=0.0, max_e_mwh=0.2,  # 200kWh
                      soc_percent=50.0,
                      name="MUPC_BESS", controllable=True)

    return net
```

---

## 3. 数据流图

```
data dict (SmartDSLoader.load_all())
  │
  ├── pv_power:           (n_steps,) kW
  ├── load_power:         (n_steps,) kW
  ├── solar_irradiance:   (n_steps,) W/m2
  ├── temperature:        (n_steps,) °C
  ├── current_electricity_price: (n_steps,)
  ├── dispatch_p_set:     (n_steps,)
  └── ...
           │
           ▼
NumpyChronics (data dict → Grid2Op 格式)
  │  展开单相标量为三相
  │  pv_power → sgen_p_mw (3相相同)
  │  load_power → load_p_mw (3相分解 + 不平衡度)
  │
  ▼
Grid2OpEnv (LightSimBackend + PandapowerNetwork)
  │
  ├── 每步注入负荷/光伏时序数据
  ├── 接收 storage_p_mw / storage_q_mvar 动作
  ├── 执行 runpp_3ph 三相潮流计算
  └── 输出三相电压 va/vb/vc
           │
           ▼
MupcEnv.step()  ←  关键同步点
  │
  ├── 入口同步：grid_soc → self._soc
  │
  ├── 三相电压 va/vb/vc → _build_observation() [索引 6..8]
  │
  ├── Q_batt 实时电压环（闭环，基于电压）
  │
  ├── 出口同步：self._soc → grid2op_power_flow.set_storage_soc()
  │
  └── 输出 (obs, reward, terminated, truncated, info)
           │
           ▼
info dict 包含 (va, vb, vc, v_avg, soc, load_rate, ...)
```

### 3.1 一帧仿真时序

```
step(t) 执行顺序：

1. [T=0] Grid2Op SOC → MupcEnv._soc（同步上一轮结果）
2. [T=1] Q_batt 实时电压环（基于 t-1 时刻电压）
3. [T=2] 动作约束校验（ACT-01/03/05）
4. [T=3] 反归一化动作 → p_batt, load_shed
5. [T=4] SOC 更新（MupcEnv 侧）
6. [T=5] Grid2Op 潮流计算 → va/vb/vc
7. [T=6] MupcEnv._soc → Grid2Op SOC（同步本轮结果）
8. [T=7] 变压器负载率、电网功率计算
9. [T=8] 奖励函数计算
10. [T=9] 观测构建、info 打包
11. [T=10] 时间步推进 (step_idx += 1)
```

---

## 4. 核心类函数接口设计

### 4.1 NumpyChronics 接口

```python
class NumpyChronics:
    """将 data dict 转换为 Grid2Op 时序格式。"""

    def __init__(self, data: dict) -> None:
        """初始化时序数据注入器。

        Args:
            data: SmartDSLoader.load_all() 返回的 data dict
        """
        ...

    def initialize(self, initial_storage_soc: float) -> None:
        """初始化 Grid2Op 环境第一帧。

        Args:
            initial_storage_soc: 初始 SOC (0.0~1.0)
        """
        ...

    def load_next(self) -> dict[str, np.ndarray]:
        """获取下一帧时序数据。

        Returns:
            dict: {
                "load_p_mw":   np.ndarray (n_load, 3)  # 三相有功负荷 MW
                "load_q_mvar": np.ndarray (n_load, 3)  # 三相无功负荷 MVar
                "sgen_p_mw":   np.ndarray (n_sgen,)    # 光伏有功 MW
                "sgen_q_mvar": np.ndarray (n_sgen,)    # 光伏无功 MVar
                "storage_soc": float                   # 当前 SOC（由 Grid2Op 管理）
            }
        """
        ...

    def reset(self, initial_storage_soc: float) -> None:
        """重置时序索引到 0。

        Args:
            initial_storage_soc: 初始 SOC (0.0~1.0)
        """
        self._current_idx = 0
        self.initialize(initial_storage_soc)

    @property
    def current_timestep(self) -> int:
        """返回当前时间步索引。"""
        return self._current_idx

    def set_time_step(self, timestep: int) -> None:
        """设置当前时间步（用于从指定位置恢复）。"""
        self._current_idx = timestep
```

### 4.2 Grid2OpPowerFlow 接口

```python
class Grid2OpPowerFlow:
    """Grid2Op 潮流计算引擎封装。"""

    def __init__(self, pandapower_net, chronics: NumpyChronics,
                 storage_soc_init: float = 0.5) -> None:
        """初始化 Grid2Op 潮流引擎。

        Args:
            pandapower_net: Pandapower 网络拓扑
            chronics: NumpyChronics 实例
            storage_soc_init: 初始 SOC (0.0~1.0)
        """
        ...

    def reset(self, initial_storage_soc: float) -> None:
        """重置环境到初始状态。"""
        ...

    def step(self, storage_p_mw: float, storage_q_mvar: float
             ) -> tuple[float, float, float, bool]:
        """执行一步潮流计算。

        Args:
            storage_p_mw: 储能有功 MW（+充电/-放电）
            storage_q_mvar: 储能无功 MVar

        Returns:
            tuple: (va, vb, vc, has_illegal)
                va/vb/vc: 三相电压标幺值 (p.u.)
                has_illegal: 是否有非法状态
        """
        ...

    def get_storage_soc(self) -> float:
        """获取当前 SOC (0.0~1.0)。"""
        ...

    def set_storage_soc(self, soc: float) -> None:
        """设置 SOC（由 MupcEnv 侧计算值同步过来）。"""
        ...

    def get_storage_power(self) -> tuple[float, float]:
        """获取当前储能实际出清有功/无功 (kW, kVar)。"""
        ...

    def get_transformer_loading(self) -> float:
        """获取当前变压器负载率 (p.u.)。"""
        ...

    def get_grid_power(self) -> float:
        """获取当前电网交换功率 (kW)。"""
        ...

    def close(self) -> None:
        """关闭 Grid2Op 环境，释放资源。"""
        ...
```

### 4.3 MupcEnv 适配层（新增参数）

```python
class MupcEnv:
    # 新增 __init__ 参数
    def __init__(self, data: dict, mode: str = "all",
                 lstm_predictor: Any = None,
                 reward_weights: dict[str, list[float]] | None = None,
                 use_grid2op: bool = True,  # ← 新增参数
                 grid2op_backend: str = "lightsim"):  # ← 新增参数
        """初始化 MUPC 环境。

        Args:
            data: SmartDSLoader 返回的 data dict
            mode: "all" (多模式) 或 "MODE-01"~"MODE-05" (单模式)
            lstm_predictor: LSTM 模型或 None→Oracle
            reward_weights: 自定义奖励权重
            use_grid2op: True=使用 Grid2Op 电压仿真, False=降级到 VoltageSimulator
            grid2op_backend: "lightsim" (C++ 加速) 或 "pandapower" (Python)
        """
        ...
```

### 4.4 Gymnasium 接口兼容性保证

```python
# 以下接口签名与原实现 100% 相同，无任何变更

def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
    """重置环境。"""
    ...

def step(self, action: np.ndarray
         ) -> tuple[np.ndarray, float, bool, bool, dict]:
    """执行一步仿真。"""
    ...

# 观测空间维度不变
observation_space: Box(low, high, shape=(58,))   # 单模式
observation_space: Box(low, high, shape=(59,))   # 多模式

# 动作空间维度不变
action_space: Box([-1, 0], [1, 1], shape=(2,))   # [p_batt_norm, load_shed_norm]

# info 字典字段不变
info: {
    "mode", "soc", "load_rate", "p_batt", "q_batt",
    "load_shedding", "pv_limit", "grid_power",
    "va", "vb", "vc", "v_avg",
    "current_demand", "peak_demand",
    "soc_clipped", "constraint_violated", "violations",
    "voltage_violation_count",
    ...reward_info
}
```

---

## 5. SOC 双向同步机制

### 5.1 同步原理

Grid2Op 的 `storage` 元件有自己的 SOC 状态，由其内部潮流计算管理。
MupcEnv 也有独立的 `_soc` 状态，用于观测构建和奖励计算。
两者必须每步同步，以保持一致。

### 5.2 同步时序

```
┌─────────────────────────────────────────────────────────────────────┐
│                         step(t) 执行流程                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [step() 入口]                                                       │
│  grid_soc = grid2op.get_storage_soc()  ← Grid2Op → MupcEnv          │
│  self._soc = grid_soc              (同步上一轮最终 SOC)              │
│                                                                      │
│  [物理计算]                                                          │
│  SOC_raw = self._soc + (-p_batt*DT) / CAP                           │
│  soc_new = clip(SOC_raw, 10%, 90%)                                  │
│                                                                      │
│  [潮流计算]                                                          │
│  grid2op.step(storage_p_mw, storage_q_mvar)                         │
│                                                                      │
│  [step() 出口]                                                       │
│  grid2op.set_storage_soc(soc_new)    ← MupcEnv → Grid2Op           │
│  self._soc = soc_new                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 同步一致性验证

```python
def step(self, action):
    # 入口同步
    grid_soc = self._grid2op_power_flow.get_storage_soc()
    self._soc = grid_soc

    # ... 物理计算 ...

    # 出口同步
    self._grid2op_power_flow.set_storage_soc(soc_new)
    self._soc = soc_new

    # 验证：下一轮 step 入口读回的 grid_soc 应等于本轮 soc_new
    # （误差容忍 0.1%，来自浮点精度）
```

### 5.4 不收敛情况处理

当 `has_illegal=True`（潮流不收敛或电压越限）时：
1. 三相电压回退到上一时刻的安全值
2. SOC 同步照常进行（使用 MupcEnv 侧计算值）
3. `info["voltage_violation_count"]` 递增
4. 不触发 `terminated`/`truncated`（潮流不收敛不等于 episode 结束）

---

## 6. 文件结构预估

```
mupc_env.py                    # 修改: 集成 Grid2OpPowerFlow
action_validator.py           # 不变
data_loader.py                # 不变
lstm_model.py                 # 不变
train.py                      # 不变

新增文件:
├── grid2op_env/
│   ├── __init__.py
│   ├── numpy_chronics.py     # NumpyChronics 类
│   ├── power_flow.py         # Grid2OpPowerFlow 类
│   ├── network.py            # create_mupc_network() Pandapower 拓扑
│   └── backend.py            # Backend 选择 (lightsim vs pandapower)
└── docs/superpowers/plans/
    └── 2026-06-09-Grid2Op电压仿真替换-DESIGN.md  ← 本文档
```

### 6.1 新增文件清单

| 文件 | 职责 | 行数预估 |
|------|------|----------|
| `grid2op_env/__init__.py` | 包导出接口 | ~20 |
| `grid2op_env/numpy_chronics.py` | data dict → Grid2Op 格式转换 | ~200 |
| `grid2op_env/power_flow.py` | Grid2Op 引擎封装 | ~250 |
| `grid2op_env/network.py` | Pandapower 网络拓扑 | ~150 |
| `grid2op_env/backend.py` | Backend 降级策略 | ~50 |

### 6.2 修改范围

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `mupc_env.py` | 修改 | 新增 `use_grid2op` 参数，`step()` 集成 Grid2Op 调用 |

---

## 7. 风险与解决措施（R-01~R-08）

### R-01：Grid2Op 与 Gymnasium 接口冲突

**风险描述**：Grid2Op 环境基类与 Gymnasium 基类不兼容，`MupcEnv` 无法同时继承两者。

**影响等级**：高

**解决措施**：采用**组合模式**（Composite Pattern），而非继承模式。

```python
# 错误做法（会导致 MRO 冲突）
class MupcEnv(Grid2OpEnv, gym.Env):  # ← 不要这样做
    pass

# 正确做法：MupcEnv 内部持有 Grid2Op 实例
class MupcEnv(gym.Env):
    def __init__(self, ...):
        self._grid2op_power_flow: Grid2OpPowerFlow | None = None
```

Grid2Op 仅作为内部计算组件，不暴露给 RL 训练代码（train.py 感知不到）。

---

### R-02：潮流计算耗时过高

**风险描述**：Pandapower 每步 Python 调用耗时 10~50ms，导致训练速度下降 5~10 倍。

**影响等级**：高

**解决措施**：

1. **优先使用 LightSimBackend**（C++ 实现，加速 20~50 倍）

```python
try:
    from lightsim2grid import LightSimBackend
    backend = LightSimBackend()
except ImportError:
    from grid2op.Backend import PandaPowerBackend
    backend = PandaPowerBackend()
```

2. **如果仍不满足 ≤50ms/步要求**，降级为单相潮流（runpp 单相而非 runpp_3ph），牺牲三相不平衡精度换取速度。

3. **性能监控**：在 `step()` 入口记录时间，每 1000 步报告平均耗时。

---

### R-03：三相潮流收敛失败

**风险描述**：孤岛、过载导致潮流不收敛（`pp.runpp_3ph` 返回非 0），电压为 NaN。

**影响等级**：中

**解决措施**：

```python
def step(self, storage_p_mw, storage_q_mvar):
    try:
        pp.runpp_3ph(self._net, calculate_voltage_angles=False)
        has_illegal = self._check_illegal()
        va, vb, vc = self._extract_voltages()
    except pp.powerflow.PowerflowError:
        # 潮流不收敛：回退到上一时刻安全电压
        has_illegal = True
        va, vb, vc = self._prev_va, self._prev_vb, self._prev_vc

    if np.any(np.isnan([va, vb, vc])):
        # 电压为 NaN：同样回退
        has_illegal = True
        va, vb, vc = self._prev_va, self._prev_vb, self._prev_vc

    self._prev_va, self._prev_vb, self._prev_vc = va, vb, vc
    return va, vb, vc, has_illegal
```

---

### R-04：Grid2Op Chronics 数据格式与 data dict 不兼容

**风险描述**：Grid2Op 期望三相格式 `load_p_mw`，而 `data` dict 提供单相标量 `load_power`。

**影响等级**：高

**解决措施**：实现 `NumpyChronics` 自定义类，将单相标量展开为 Grid2Op 三相格式。

```python
class NumpyChronics:
    """将 SmartDSLoader data dict 转换为 Grid2Op 时序格式。"""

    def load_next(self) -> dict[str, np.ndarray]:
        idx = self._current_idx
        self._current_idx += 1

        # 单相标量 → 三相展开
        load_p = self._data["load_power"][idx]  # kW
        load_p_3ph = self._expand_single_to_three_phase(load_p)  # (3,)
        load_p_mw = load_p_3ph / 1000.0  # → MW

        pv_p = self._data["pv_power"][idx]  # kW
        sgen_p_mw = np.array([pv_p / 1000.0])  # 光伏（单元件）

        return {
            "load_p_mw": load_p_mw.reshape(1, 3),   # (n_load, 3)
            "load_q_mvar": np.zeros((1, 3)),         # 无功由 PF 计算
            "sgen_p_mw": sgen_p_mw,                  # (n_sgen,)
            "sgen_q_mvar": np.array([0.0]),
        }
```

---

### R-05：动作空间映射错误

**风险描述**：2 维 RL 动作 `[p_batt_norm, load_shed_norm]` 未正确映射到 Grid2Op `storage_p`。

**影响等级**：高

**解决措施**：

1. **单位转换**：kW ↔ MW（`p_batt / 1000.0`）

2. **动作映射验证单元测试**：

```python
def test_action_mapping():
    """验证 action → storage_p 映射一致性。"""
    env = MupcEnv(data, use_grid2op=True)
    env.reset()

    # 测试动作满量程
    action = np.array([1.0, 1.0])  # p_batt=+500kW, load_shed=500kW
    grid_p_before = env._grid2op_power_flow.get_storage_power()[0]

    # 触发 step
    obs, reward, _, _, _ = env.step(action)

    # 验证 Grid2Op 接收到的 storage_p（应在误差范围内）
    grid_p_after = env._grid2op_power_flow.get_storage_power()[0]
    expected_delta = 500.0 / 1000.0  # MW

    assert np.isclose(grid_p_after - grid_p_before, expected_delta, atol=0.01)
```

---

### R-06：SOC 状态不同步

**风险描述**：`MupcEnv._soc` 与 Grid2Op `storage.soc` 在每步结束时不一致。

**影响等级**：高

**解决措施**：双向同步机制（详见第 5 章）。

```python
def step(self, action):
    # 入口同步（Grid2Op → MupcEnv）
    self._soc = self._grid2op_power_flow.get_storage_soc()

    # ... 物理计算 ...

    # 出口同步（MupcEnv → Grid2Op）
    self._grid2op_power_flow.set_storage_soc(soc_new)
    self._soc = soc_new
```

---

### R-07：奖励函数依赖字段不可用

**风险描述**：Grid2Op Observation 不包含奖励函数所需的某些字段（如 `solar_irradiance`, `current_demand`）。

**影响等级**：中

**解决措施**：

Grid2Op Observation 提供 `load_p`/`storage_p` 等电气量，三相电压由 Pandapower 计算。
奖励函数所需的其他字段（电价、气象、调度）继续从 `data` dict 读取，不依赖 Grid2Op。

```python
def _compute_reward(self, **r) -> tuple[float, dict]:
    # 以下字段来自 data dict，不受 Grid2Op 影响
    price = self._data["current_electricity_price"][self._step_idx]
    dispatch_p = self._data["dispatch_p_set"][self._step_idx]
    solar_irradiance = self._data["solar_irradiance"][self._step_idx]

    # 以下字段来自 Grid2Op
    va, vb, vc = r["va"], r["vb"], r["vc"]
    load_rate = r["load_rate"]  # Grid2Op 提供

    # 奖励计算逻辑不变
    ...
```

---

### R-08：依赖库版本冲突

**风险描述**：`pandapower` vs `grid2op` 版本不兼容（grid2op 内置特定版本 pandapower）。

**影响等级**：低

**解决措施**：

1. **虚拟环境隔离**：使用 `venv` 或 `conda` 创建独立环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows
pip install grid2op pandapower lightsim2grid numpy
```

2. **版本兼容性矩阵测试**：

```python
# 在 CI 中测试以下组合
COMPATIBILITY_MATRIX = [
    ("grid2op==1.9.0", "pandapower==3.8.0"),
    ("grid2op==1.9.2", "pandapower==3.8.5"),
    ("grid2op==1.10.0", "pandapower==3.9.0"),
]
```

3. **自动降级**：若 `import grid2op` 失败，自动回退到 `VoltageSimulator`（`use_grid2op=False`）

```python
try:
    import grid2op
    _GRID2OP_AVAILABLE = True
except ImportError:
    _GRID2OP_AVAILABLE = False
    print("[WARN] Grid2Op 不可用，降级到 VoltageSimulator")
```

---

## 附录：关键常量对照表

| 常量名 | 值 | 来源 | 说明 |
|--------|-----|------|------|
| `TRANSFORMER_KVA` | 500.0 | mupc_env.py | 变压器额定容量 |
| `BATTERY_CAPACITY_KWH` | 200.0 | mupc_env.py | 电池容量 |
| `P_BATT_MAX_KW` | 500.0 | mupc_env.py | 最大充放电功率 |
| `Q_BATT_MAX_KVAR` | 300.0 | mupc_env.py | 最大无功 |
| `SOC_MIN/MAX` | 0.10/0.90 | mupc_env.py | SOC 硬限制（SAFETY）|
| `DT_HOURS` | 0.25 | mupc_env.py | 时间步长 15 分钟 |
| `OVERLOAD_THRESHOLD` | 0.85 | mupc_env.py | 过载阈值 |
| `PV_ARRAY_KW` | 200.0 | mupc_env.py | 光伏容量 |
| `LOAD_PEAK_KW` | 400.0 | mupc_env.py | 负荷峰值 |
| `LOAD_PF` | 0.90 | mupc_env.py | 负荷功率因数 |

---

## 修订记录

| 版本 | 日期 | 修订内容 |
|------|------|----------|
| v1.0 | 2026-06-09 | 初始版本：基于 Grid2Op + Pandapower 电压仿真替换 PRD |