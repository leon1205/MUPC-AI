"""
MUPC 全状态强化学习环境 — 兼容重定向

代码已迁移至 mupc_env/ 包 (模块化架构):
  mupc_env/__init__.py     — 重导出 MupcEnv
  mupc_env/constants.py    — 物理常数 + 归一化边界 + 权重映射
  mupc_env/voltage_sim.py  — 三相电压简化线路模型 (Q-V 耦合)
  mupc_env/observation.py  — 观测构建 + 归一化 + 季节编码
  mupc_env/rewards.py      — 奖励调度器 + 5 场景奖励函数
  mupc_env/core.py         — MupcEnv 主类 + 自测入口

使用:
  from mupc_env import MupcEnv   # 与之前完全兼容

自测:
  python -m mupc_env.core        # 新方式
  python -m mupc_env             # (需要 __main__.py)
"""

from mupc_env import MupcEnv

__all__ = ["MupcEnv"]

if __name__ == "__main__":
    print(__doc__)
    print("请使用: python -m mupc_env.core  运行模块化自测")
