"""
MUPC 强化学习环境包 (模块化架构, v2.16)

使用:
    from mupc_env import MupcEnv

模块:
    constants.py   — 物理常数 + 归一化边界 + 权重映射
    voltage_sim.py — 三相电压简化线路模型 (Q-V 耦合)
    observation.py — 观测构建 + 归一化 + 季节编码
    rewards.py     — 奖励调度器 + 5 场景奖励函数
    core.py        — MupcEnv 主类
"""

from .core import MupcEnv

__all__ = ["MupcEnv"]
