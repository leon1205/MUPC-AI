"""
MUPC 配置管理器 — 从 YAML 文件加载环境参数，支持命令行 --config 指定。

用法:
  python train.py --config config/mupc_env_config.yaml

配置文件格式见 config/mupc_env_config.yaml。

如未指定 --config，则使用代码中的硬编码默认值（向后兼容）。
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ═══════════════════════════════════════════════════════════════
# 配置数据结构 (DataClass)
# ═══════════════════════════════════════════════════════════════


@dataclass
class PhysicalConfig:
    transformer_kva: float = 200.0
    battery_capacity_kwh: float = 100.0
    p_batt_max_kw: float = 50.0
    q_batt_max_kvar: float = 300.0
    load_shed_max_kw: float = 60.0
    pv_array_kw: float = 150.0
    load_peak_kw: float = 60.0


@dataclass
class SafetyConfig:
    soc_min: float = 0.10
    soc_max: float = 0.90
    overload_threshold: float = 0.85   # 过载判断阈值（变压器额定容量百分比）
    overload_start_pct: float = 0.75  # 过载惩罚起始点（75% 开始惩罚，与 overload_threshold 形成惩罚区间）
    battery_charge_efficiency: float = 0.90   # 充电效率 (锂离子电池典型值 90%)
    battery_discharge_efficiency: float = 0.90  # 放电效率 (锂离子电池典型值 90%)


@dataclass
class TimeConfig:
    dt_hours: float = 0.25
    episode_length: int = 96
    load_pf: float = 0.90
    demand_window_steps: int = 4  # 需量滑动窗口步数（4步×15分钟=1小时）


@dataclass
class ContractConfig:
    contract_demand_kw: float = 300.0
    grid_emission_factor: float = 0.581


@dataclass
class VoltageSimulatorConfig:
    k_p: float = 0.05
    k_q: float = 0.03
    s_base: float = 500.0
    v_min: float = 0.85
    v_max: float = 1.15
    noise_std: float = 0.005
    imbalance: float = 0.003
    # 动态阻抗扰动（模拟线路老化/温度变化导致的 R/X 漂移）
    impedance_drift_pct: float = 0.10   # ±10% 随机扰动
    # 谐波背景（模拟农网配电侧 3/5 次谐波污染）
    harmonic_3rd_pct: float = 0.03     # 3次谐波 3% 幅值
    harmonic_5th_pct: float = 0.02     # 5次谐波 2% 幅值


@dataclass
class CommConfig:
    """通信延迟配置（模拟 RTU 轮询周期）。"""
    action_delay_steps_min: int = 1   # 最小延迟步数（1步=15分钟）
    action_delay_steps_max: int = 3    # 最大延迟步数


@dataclass
class DualControlConfig:
    """双参数下垂控制配置 (v2.15: 2 维动作空间 [p_ref, k_droop])"""
    enabled: bool = True
    k_droop_min: float = 0.0
    k_droop_max: float = 30.0
    p_ref_ramp_limit_kw: float = 50.0
    k_droop_ramp_limit: float = 10.0
    pv_limit_min: float = 0.0


@dataclass
class VppPricingConfig:
    """虚拟电厂价格配置 (SCENE-B3 辅助服务收益).

    对齐下游 AI 引擎 PRD v2.15 Section 7.5 公式:
      R_ancillary_service = P_regulation_capacity * capacity_price
                          + P_regulation_mileage * mileage_price
    本地训练侧使用此处占位值, 部署侧如需调整价格应同时更新本配置.
    """
    capacity_price: float = 0.1     # 容量价格 (元/kW)
    mileage_price: float = 0.05     # 里程价格 (元/kW)


@dataclass
class QControlConfig:
    k_q: float = 200.0


@dataclass
class RewardThresholdConfig:
    voltage_deadband: float = 0.05
    q_margin_threshold: float = 0.10
    voltage_high_limit: float = 1.05
    soc_critical: float = 0.10
    voltage_penalty_high: float = 2.0
    voltage_penalty_low: float = 1.0


@dataclass
class ActionConstraintConfig:
    p_batt_max: float = 50.0
    s_max: float = 200.0
    load_shed_max: float = 60.0
    delta_p_max: float = 50.0


@dataclass
class ActionSpaceConfig:
    p_ref_norm_min: float = -1.0
    p_ref_norm_max: float = 1.0
    k_droop_norm_min: float = -1.0
    k_droop_norm_max: float = 1.0
    load_shed_norm_min: float = 0.0
    load_shed_norm_max: float = 1.0
    pv_limit_min: float = 0.0
    pv_limit_max: float = 1.0
    confidence_min: float = 0.0
    confidence_max: float = 1.0


@dataclass
class ObsNormalizationConfig:
    d1_pv_range: list = field(default_factory=lambda: [0.0, 150.0])
    d1_load_range: list = field(default_factory=lambda: [0.0, 60.0])
    d1_power_range: list = field(default_factory=lambda: [-500.0, 500.0])
    d1_voltage_range: list = field(default_factory=lambda: [0.85, 1.15])
    d5_irradiance_range: list = field(default_factory=lambda: [0.0, 1500.0])
    d5_temp_range: list = field(default_factory=lambda: [-20.0, 60.0])
    d6_dispatch_range: list = field(default_factory=lambda: [-500.0, 500.0])


@dataclass
class RewardWeightsConfig:
    # v2.15 MODE-01: 8 项权重 (对齐下游 v2.15 PRD 7.2)
    # w1(光伏消纳), w2(电池), w3(过载), w4(P-Q协同), w5(变化率),
    # w6(电压斜率), w7(下垂平滑), w8(安全覆盖)
    MODE_01: list = field(default_factory=lambda: [1.0, 0.5, 2.0, 1.0, 0.5, 0.5, 0.5, 1.0])
    MODE_02: list = field(default_factory=lambda: [1.0, 1.0, 2.0])
    MODE_03: list = field(default_factory=lambda: [1.0, 0.5])
    MODE_04: list = field(default_factory=lambda: [1.0, 2.0, 1.0])
    MODE_05: list = field(default_factory=lambda: [1.0, 1.0])


# ═══════════════════════════════════════════════════════════════
# 主配置类
# ═══════════════════════════════════════════════════════════════


@dataclass
class MupcConfig:
    """MUPC 完整配置容器。"""
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    voltage_simulator: VoltageSimulatorConfig = field(default_factory=VoltageSimulatorConfig)
    q_control: QControlConfig = field(default_factory=QControlConfig)
    reward_thresholds: RewardThresholdConfig = field(default_factory=RewardThresholdConfig)
    action_constraints: ActionConstraintConfig = field(default_factory=ActionConstraintConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    obs_normalization: ObsNormalizationConfig = field(default_factory=ObsNormalizationConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    dual_control: DualControlConfig = field(default_factory=DualControlConfig)
    vpp_pricing: VppPricingConfig = field(default_factory=VppPricingConfig)

    # 配置文件路径（用于校验）
    _source_file: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MupcConfig":
        """从 YAML 文件加载配置。"""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        def _section(cfg_cls, section: dict):
            return cfg_cls(**{k: v for k, v in section.items() if k in cfg_cls.__dataclass_fields__})

        return cls(
            physical=_section(PhysicalConfig, data.get("physical", {})),
            safety=_section(SafetyConfig, data.get("safety", {})),
            time=_section(TimeConfig, data.get("time", {})),
            contract=_section(ContractConfig, data.get("contract", {})),
            voltage_simulator=_section(VoltageSimulatorConfig, data.get("voltage_simulator", {})),
            q_control=_section(QControlConfig, data.get("q_control", {})),
            reward_thresholds=_section(RewardThresholdConfig, data.get("reward_thresholds", {})),
            action_constraints=_section(ActionConstraintConfig, data.get("action_constraints", {})),
            action_space=_section(ActionSpaceConfig, data.get("action_space", {})),
            obs_normalization=_section(ObsNormalizationConfig, data.get("obs_normalization", {})),
            reward_weights=_section(RewardWeightsConfig, data.get("reward_weights", {})),
            comm=_section(CommConfig, data.get("comm", {})),
            dual_control=_section(DualControlConfig, data.get("dual_control", {})),
            vpp_pricing=_section(VppPricingConfig, data.get("vpp_pricing", {})),
            _source_file=str(path),
        )

    @classmethod
    def default(cls) -> "MupcConfig":
        """返回默认配置（硬编码回退）。"""
        return cls()

    def summary(self) -> str:
        """返回配置摘要字符串（用于日志输出）。"""
        src = self._source_file or "<default>"
        lines = [
            f"MupcConfig(source={src})",
            f"  physical: transformer={self.physical.transformer_kva}kVA, "
            f"battery={self.physical.battery_capacity_kwh}kWh, "
            f"p_batt={self.physical.p_batt_max_kw}kW, "
            f"pv={self.physical.pv_array_kw}kW",
            f"  safety: SOC=[{self.safety.soc_min},{self.safety.soc_max}], "
            f"overload={self.safety.overload_threshold:.0%}",
            f"  action_constraints: p_batt_max={self.action_constraints.p_batt_max}kW, "
            f"s_max={self.action_constraints.s_max}kVA, "
            f"load_shed_max={self.action_constraints.load_shed_max}kW",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 全局配置实例（训练入口可通过 load_config() 设置）
# ═══════════════════════════════════════════════════════════════

_global_config: MupcConfig = MupcConfig.default()


def get_config() -> MupcConfig:
    """获取当前全局配置实例。"""
    return _global_config


def load_config(config_path: str | Path | None = None) -> MupcConfig:
    """加载配置到全局实例并返回。

    Args:
        config_path: YAML 配置文件路径，None 则使用默认配置。

    Returns:
        MupcConfig 实例。
    """
    global _global_config
    if config_path is not None:
        cfg = MupcConfig.from_yaml(config_path)
        _global_config = cfg
        print(f"[CONFIG] 已加载配置文件: {config_path}")
        print(cfg.summary())
    else:
        cfg = MupcConfig.default()
        _global_config = cfg
        print("[CONFIG] 使用默认配置（未指定 --config）")
    return cfg


# ═══════════════════════════════════════════════════════════════
# train.py 参数解析扩展
# ═══════════════════════════════════════════════════════════════


def add_config_args(p: argparse.ArgumentParser) -> None:
    """为 train.py 添加 --config 参数。"""
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径，如 config/mupc_env_config.yaml "
             "(未指定时使用代码中的硬编码默认值)",
    )