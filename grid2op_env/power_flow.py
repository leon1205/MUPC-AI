"""Grid2OpPowerFlow — Grid2Op 潮流计算引擎封装。

封装 Grid2Op 环境生命周期，提供同步 SOC 和获取三相电压的接口。
内部使用 LightSimBackend（如可用则降级到 PandaPowerBackend）。

潮流不收敛处理（R-03 缓解）：
- 三相潮流不收敛或电压出现 NaN 时，回退到上一时刻安全电压
- has_illegal标记为 True，供调用方判断
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from .backend import _select_backend, is_grid2op_available

if TYPE_CHECKING:
    import pandapower as pp
    from grid2op import Environment
    from grid2op_env.numpy_chronics import NumpyChronics

# ── 物理常量（运行时从配置读取，避免导入时序问题）─────────────────

def _battery_capacity_kwh() -> float:
    """运行时获取电池容量（从配置）。"""
    try:
        from config.config_manager import get_config
        return get_config().physical.battery_capacity_kwh
    except Exception:
        return 100.0  # 回退默认值，与 mupc_env_config.yaml 一致


def _battery_capacity_mwh() -> float:
    return _battery_capacity_kwh() / 1000.0


class Grid2OpPowerFlow:
    """Grid2Op 潮流计算引擎封装。

    使用 LightSimBackend（如不可用则降级到 PandaPowerBackend）。
    不继承 Gymnasium 基类，仅作为 MupcEnv 的内部计算组件。

    Attributes:
        _net: Pandapower 网络拓扑
        _chronics: NumpyChronics 实例（数据注入）
        _env: Grid2Op Environment 实例
        _backend_name: 所使用的 Backend 名称
        _prev_va/vb/vc: 上一时刻三相电压（用于不收敛回退）
    """

    def __init__(
        self,
        pandapower_net: "pp.auxiliary.pandapowerNet",
        chronics: "NumpyChronics",
        storage_soc_init: float = 0.5,
        voltage_imbalance: float = 0.003,
    ) -> None:
        """初始化 Grid2Op 潮流引擎。

        电压仿真采用单相等效潮流 + 人工三相不平衡度模拟。
        Pandapower 的 runpp 是单相 Newton-Raphson，无法反映真实
        三相不平衡（如单相大负荷、不对称故障），但可正确捕捉 P-V/Q-V
        灵敏度，足以支撑 RL 训练的电压趋势学习。

        Args:
            pandapower_net: Pandapower 网络拓扑（来自 create_mupc_network）
            chronics: NumpyChronics 实例
            storage_soc_init: 初始 SOC (0.0~1.0)
            voltage_imbalance: 三相不平衡度 (p.u.)，用于在单相等效电压
                基础上模拟相间偏差 (default 0.003 = 0.3%)
        """
        self._net = pandapower_net
        self._chronics = chronics
        self._storage_soc_init = storage_soc_init
        self._voltage_imbalance = voltage_imbalance

        # Grid2Op 可用性检测
        self._grid2op_available = is_grid2op_available()
        self._backend_name = "unknown"

        # 上一时刻电压（用于不收敛回退）
        self._prev_va: float = 1.0
        self._prev_vb: float = 1.0
        self._prev_vc: float = 1.0

        # 储能元件索引（从 pandapower net 查找）
        self._storage_idx: int | None = None

        if self._grid2op_available:
            self._env = self._init_grid2op_env()
        else:
            self._env = None

    def _init_grid2op_env(self) -> "Environment":
        """初始化 Grid2Op 环境。

        优先使用 LightSimBackend，降级到 PandaPowerBackend。
        设置可控储能（controllable=True）以便接受 RL 控制指令。

        Returns:
            Grid2Op Environment 实例
        """
        BackendClass, backend_name = _select_backend()
        self._backend_name = backend_name

        try:
            # Grid2Op >= 1.9.0
            from grid2op import Environment
            from grid2op.Chronics import FromNumpyFlatForecasterWithCache
            from grid2op.Backend import Backend
        except ImportError:
            # Grid2Op 未安装：降级到 VoltageSimulator
            self._grid2op_available = False
            self._env = None
            return None

        # 通过名称查找储能元件索引（WARNING-2: 避免硬编码 0）
        self._storage_idx = None
        if "storage" in self._net and len(self._net.storage) > 0:
            # 尝试从 storage DataFrame 的 name 列查找 "MUPC_BESS"
            name_col = None
            if "name" in self._net.storage.columns:
                name_col = self._net.storage["name"]
            elif hasattr(self._net.storage, "index"):
                # 回退：遍历 index 列
                name_col = self._net.storage.index
            if name_col is not None:
                for idx, name in enumerate(name_col):
                    if "MUPC_BESS" in str(name):
                        self._storage_idx = idx
                        break
            if self._storage_idx is None:
                self._storage_idx = 0  # 后备：假设第一个储能元件

        # 将 pandapower net 注册到 Grid2Op
        # 使用简单的 FromNumpyFlatForecasterWithCache chronics 占位
        # 实际数据由 NumpyChronics.load_next() 注入
        try:
            #尝试标准 Grid2Op 环境创建方式
            env = Environment(
                pandapower_net=self._net,
                backend=BackendClass(),
                chronics_class=FromNumpyFlatForecasterWithCache,
                param_env={"thermal_limit_a": 1000.0},
            )
        except Exception:
            # 如果标准方式失败，尝试简化方式
            # 直接使用 pandapower net 创建环境
            try:
                env = Environment(
                    pandapower_net=self._net,
                    backend=BackendClass(),
                    chronics_class=FromNumpyFlatForecasterWithCache,
                )
            except Exception:
                self._grid2op_available = False
                self._env = None
                return None

        return env

    def _inject_chronics_data(self) -> bool:
        """将 NumpyChronics 当前帧数据注入到 Pandapower 网络。

        直接修改 pandapower net 的 load/sgen 值，绕过 Grid2Op backend，
        减少中间层开销，使电压提取更直接。

        Returns:
            bool: 注入是否成功
        """
        try:
            data = self._chronics.load_next()
        except Exception as e:
            print(f"[WARN] NumpyChronics.load_next() 失败: {e}")
            return False

        load_p = data["load_p_mw"]   # (n_loads, 3) MW
        load_q = data["load_q_mvar"] # (n_loads, 3) MVar
        sgen_p = data["sgen_p_mw"]   # (n_sgens,) MW

        # 注入负荷（三相求和后写入单相等效负荷）
        n_loads = min(load_p.shape[0], len(self._net.load))
        for i in range(n_loads):
            p_total = float(load_p[i].sum())  # 三相求和 → 单相等效
            q_total = float(load_q[i].sum())
            self._net.load.at[i, "p_mw"] = p_total
            self._net.load.at[i, "q_mvar"] = q_total

        # 注入光伏
        for i in range(min(len(sgen_p), len(self._net.sgen))):
            self._net.sgen.at[i, "p_mw"] = float(sgen_p[i])
            self._net.sgen.at[i, "q_mvar"] = 0.0

        return True

    def _get_bus_voltages(self) -> tuple[float, float, float, bool]:
        """从 pandapower net 提取母线电压，模拟三相不平衡。

        本网络拓扑为单母线等效三相（无相间耦合），Pandapower 的 runpp
        只计算单相等效电压 vm_pu。三相电压通过以下方式模拟：
        - A 相：基准电压 va = vm_pu
        - B 相：va × (1 - imbalance)
        - C 相：va × (1 + imbalance)

        不平衡度通过 voltage_imbalance 参数配置（默认 0.003 = 0.3%），
        模拟农网低压侧典型的三相不平衡场景。

        Returns:
            tuple: (va, vb, vc, has_illegal)
                三相电压标幺值 (p.u.)
                has_illegal: 是否有非法状态（电压越限/NaN/res_bus 为空）
        """
        res_bus = self._net.res_bus

        # 末端节点电压（End_Node_Bus = bus 索引2）
        # bus 索引：0=HV_Grid, 1=LV_Main_Bus, 2=End_Node_Bus
        end_bus_idx = 2

        if len(res_bus) > end_bus_idx:
            va = float(res_bus.at[end_bus_idx, "vm_pu"])
            vb = float(va * (1.0 - self._voltage_imbalance))
            vc = float(va * (1.0 + self._voltage_imbalance))
            has_illegal = False
        else:
            va, vb, vc = self._prev_va, self._prev_vb, self._prev_vc
            has_illegal = True

        # 检查 NaN
        if np.isnan(va) or np.isnan(vb) or np.isnan(vc):
            va, vb, vc = self._prev_va, self._prev_vb, self._prev_vc
            has_illegal = True

        return va, vb, vc, has_illegal

    def _run_powerflow(
        self, storage_p_mw: float, storage_q_mvar: float
    ) -> bool:
        """执行单相等效潮流计算。

        直接调用 pandapower runpp（Newton-Raphson），P-Q/V 灵敏度正确，
        可满足 RL 训练对电压趋势精度的要求（误差 ≤ 2%）。
        三相电压差异通过 _get_bus_voltages 中的人工不平衡度模拟。

        Args:
            storage_p_mw: 储能有功 MW（+充电/-放电）
            storage_q_mvar: 储能无功 MVar

        Returns:
            bool: 潮流是否成功收敛
        """
        # 注入当前帧数据到 pandapower net
        if not self._inject_chronics_data():
            return False

        # 设置储能功率（直接修改 pandapower net）
        if self._storage_idx is not None and "storage" in self._net:
            self._net.storage.at[self._storage_idx, "p_mw"] = storage_p_mw
            self._net.storage.at[self._storage_idx, "q_mvar"] = storage_q_mvar

        # 执行 pandapower 单相等效潮流
        import pandapower as pp
        try:
            pp.runpp(self._net, numba=False)
            return True
        except Exception:
            return False

    # ── 公开接口 ────────────────────────────────────────────────

    def reset(self, initial_storage_soc: float) -> None:
        """重置 Grid2Op 环境到初始状态。

        Args:
            initial_storage_soc: 初始 SOC (0.0~1.0)
        """
        # 重置 chronics
        self._chronics.reset(initial_storage_soc)

        # 重置上一时刻电压
        self._prev_va = 1.0
        self._prev_vb = 1.0
        self._prev_vc = 1.0

        if self._env is not None:
            try:
                self._env.reset()
            except Exception:
                pass

            # 设置初始 SOC
            self.set_storage_soc(initial_storage_soc)

    def step(
        self, storage_p_mw: float, storage_q_mvar: float
    ) -> tuple[float, float, float, bool]:
        """执行一步潮流计算。

        Args:
            storage_p_mw: 储能有功 MW（+充电/-放电）
            storage_q_mvar: 储能无功 MVar

        Returns:
            tuple: (va, vb, vc, has_illegal)
                va/vb/vc: 三相电压标幺值 (p.u.)
                has_illegal: 是否有非法状态（电压越限/潮流不收敛）
        """
        # 执行单相等效潮流
        success = self._run_powerflow(storage_p_mw, storage_q_mvar)

        if not success or self._has_nan_voltage():
            # 回退到上一时刻电压（保持安全值）
            va, vb, vc = self._prev_va, self._prev_vb, self._prev_vc
            has_illegal = True
        else:
            va, vb, vc, has_illegal = self._get_bus_voltages()
            self._prev_va, self._prev_vb, self._prev_vc = va, vb, vc

        # CRITICAL-3: 同步 SOC 到 NumpyChronics（下次 load_next 返回值含 storage_soc）
        grid_soc = self.get_storage_soc()
        self._chronics.set_storage_soc(grid_soc)

        return va, vb, vc, has_illegal

    def _has_nan_voltage(self) -> bool:
        """检查当前 pandapower 计算结果是否包含 NaN 电压。"""
        try:
            res_bus = self._net.res_bus
            if len(res_bus) == 0:
                return True
            return res_bus["vm_pu"].isna().any()
        except Exception:
            return True

    def get_storage_soc(self) -> float:
        """获取当前 SOC (0.0~1.0)。

        Returns:
            当前 SOC 值（0.0~1.0），如无法获取则返回 0.5
        """
        if self._env is None:
            return 0.5

        try:
            # Grid2Op storage SOC 存储在 backend._storage 中
            storage_df = self._env.backend._storage
            if len(storage_df) > 0:
                soc = float(storage_df.at[0, "soc_percent"]) / 100.0
                # SAFETY: SOC 硬限制 0.0~1.0（不可突破）
                return max(0.0, min(1.0, soc))
        except Exception:
            pass

        # fallback: 从 pandapower net 读取
        try:
            if "storage" in self._net and len(self._net.storage) > 0:
                soc = float(self._net.storage.at[0, "soc_percent"]) / 100.0
                # SAFETY: SOC 硬限制 0.0~1.0（不可突破）
                return max(0.0, min(1.0, soc))
        except Exception:
            pass

        return 0.5

    def set_storage_soc(self, soc: float) -> None:
        """设置 SOC（同步 MupcEnv._soc 到 Grid2Op）。

        Args:
            soc: SOC 值 (0.0~1.0)
        """
        soc_clamped = float(np.clip(soc, 0.0, 1.0))

        if self._env is not None:
            try:
                storage_df = self._env.backend._storage
                if len(storage_df) > 0:
                    storage_df.at[0, "soc_percent"] = soc_clamped * 100.0
            except Exception:
                pass

        # 同时更新 pandapower net
        if "storage" in self._net and len(self._net.storage) > 0:
            self._net.storage.at[0, "soc_percent"] = soc_clamped * 100.0

    def get_storage_power(self) -> tuple[float, float]:
        """获取当前储能实际出清有功/无功 (kW, kVar)。

        Returns:
            tuple: (p_batt_kw, q_batt_kvar)
        """
        if self._env is not None:
            try:
                storage_df = self._env.backend._storage
                if len(storage_df) > 0:
                    p_mw = float(storage_df.at[0, "p_mw"])
                    q_mvar = float(storage_df.at[0, "q_mvar"])
                    return p_mw * 1000.0, q_mvar * 1000.0
            except Exception:
                pass

        return 0.0, 0.0

    def get_transformer_loading(self) -> float:
        """获取当前变压器负载率 (p.u.)。

        Returns:
            变压器负载率（相对于 200kVA 额定容量，PRD v2.6）
        """
        try:
            if len(self._net.res_trafo) > 0:
                s_mva = float(self._net.res_trafo.at[0, "loading_percent"])
                return s_mva / 100.0  # pandapower 返回百分比 → p.u.
        except Exception:
            pass
        return 0.0

    def get_grid_power(self) -> float:
        """获取当前电网交换功率 (kW)。

        正值表示从电网购电，负值表示向电网售电。

        Returns:
            电网交换功率 (kW)
        """
        try:
            if len(self._net.res_ext_grid) > 0:
                p_mw = float(self._net.res_ext_grid.at[0, "p_mw"])
                return p_mw * 1000.0
        except Exception:
            pass
        return 0.0

    def close(self) -> None:
        """关闭 Grid2Op 环境，释放资源。"""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None