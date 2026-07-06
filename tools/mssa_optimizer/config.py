"""
MSSA 运行参数配置 — v3.1。
"""

from dataclasses import dataclass, field


@dataclass
class MssaConfig:
    """麻雀搜索算法运行配置。"""

    # 种群参数
    population: int = 20          # 麻雀总数
    iterations: int = 50          # 最大迭代次数
    discoverer_ratio: float = 0.2  # 发现者比例
    sentinel_ratio: float = 0.2    # 警戒者比例
    safety_threshold: float = 0.8  # 安全阈值 ST

    # 训练参数
    mode: str = "MODE-01"         # 训练模式
    data_source: str = "smartds"  # 数据源
    train_steps: int = 50000      # 每次评估的训练步数
    no_lstm: bool = False         # 是否使用 Oracle (快速模式, 仅搜索 RL 超参)

    # 输出
    output_path: str = "mssa_search_result.json"
    temp_dir: str = "./mssa_temp/"  # 临时 config JSON 目录

    # 并行
    n_workers: int = 1            # 并行训练数 (需 GPU 或多核 CPU)
    cache_enabled: bool = True    # 启用 SHA256 参数缓存


DEFAULT_CONFIG = MssaConfig()
