"""
MSSA (麻雀搜索算法) 超参优化工具包 — v3.1。

用于自动搜索 LSTM 预测模型的最优超参数组合，替代人工 grid search。

用法:
  python -m tools.mssa_optimizer --population 20 --iterations 50 --mode MODE-01

架构:
  mssa.py          — 麻雀搜索算法核心
  search_space.py  — 10 维搜索空间编解码
  objective.py     — 目标函数 (subprocess 调用 train.py)
  config.py        — MSSA 运行参数配置
  output.py        — 结果输出 JSON
"""

__version__ = "v3.1.0"
