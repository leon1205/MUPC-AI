"""
10 维超参数搜索空间定义 + 编解码 — v3.1。

将连续向量 x ∈ [0,1]^D 映射为具体的超参数值，反之亦然。
"""

import numpy as np
from typing import Any

# 搜索空间元数据: (名称, 类型, 范围, 精度)
SEARCH_SPACE = [
    # dim 0: hidden_size
    ("hidden_size", "discrete", [32, 64, 96, 128], 0),
    # dim 1: num_layers
    ("num_layers", "discrete", [1, 2, 3], 0),
    # dim 2: input_window
    ("input_window", "discrete", [12, 24, 36], 0),
    # dim 3: vmd_k
    ("vmd_k", "integer", [2, 10], 0),
    # dim 4: vmd_alpha (log scale)
    ("vmd_alpha", "continuous", [100.0, 5000.0], 1),
    # dim 5: learning_rate (log scale)
    ("learning_rate", "log_continuous", [1e-4, 1e-2], 6),
    # dim 6: batch_size
    ("batch_size", "discrete", [16, 32, 64, 128], 0),
    # dim 7: dropout
    ("dropout", "continuous", [0.0, 0.5], 3),
    # dim 8: attn_type (当前仅 additive, 预留扩展 dot_product/scaled_dot)
    # 扩展时更新枚举值列表, 编解码自动适配
    ("attn_score", "enum", ["additive"], 0),
    # dim 9: optimizer
    ("optimizer", "enum", ["Adam", "AdamW"], 0),
]

DIM = len(SEARCH_SPACE)

# 注意: KEY_MAP 权威定义在 train.py (parse_mssa_config), 此处为参考副本, 请勿以此为准


def encode(params: dict[str, Any]) -> np.ndarray:
    """将超参数字典编码为 [0,1]^D 连续向量。"""
    x = np.zeros(DIM)
    for i, (name, kind, values, _) in enumerate(SEARCH_SPACE):
        v = params[name]
        if kind == "discrete":
            idx = values.index(v)
            x[i] = idx / (len(values) - 1) if len(values) > 1 else 0.0
        elif kind == "integer":
            lo, hi = values
            x[i] = (v - lo) / (hi - lo)
        elif kind == "continuous":
            lo, hi = values
            x[i] = (v - lo) / (hi - lo)
        elif kind == "log_continuous":
            lo, hi = values
            x[i] = (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        elif kind == "enum":
            idx = values.index(v)
            x[i] = idx / (len(values) - 1) if len(values) > 1 else 0.0
    return x


def decode(x: np.ndarray) -> dict[str, Any]:
    """将 [0,1]^D 连续向量解码为超参数字典, 可直接传给 train.py --config。"""
    params = {}
    for i, (name, kind, values, precision) in enumerate(SEARCH_SPACE):
        xi = float(np.clip(x[i], 0.0, 1.0))
        if kind == "discrete":
            opts = values
            idx = int(round(xi * (len(opts) - 1)))
            params[name] = opts[idx]
        elif kind == "integer":
            lo, hi = values
            v = int(round(lo + xi * (hi - lo)))
            params[name] = max(lo, min(hi, v))
        elif kind == "continuous":
            lo, hi = values
            v = lo + xi * (hi - lo)
            if precision > 0:
                v = round(v, precision)
            params[name] = float(v)
        elif kind == "log_continuous":
            lo, hi = values
            log_v = np.log10(lo) + xi * (np.log10(hi) - np.log10(lo))
            v = round(10 ** log_v, precision)
            params[name] = float(v)
        elif kind == "enum":
            opts = values
            idx = int(round(xi * (len(opts) - 1)))
            params[name] = opts[idx]
    return params


def random_position() -> np.ndarray:
    """生成随机的 [0,1]^D 向量。"""
    return np.random.rand(DIM)


def random_population(n: int) -> np.ndarray:
    """生成 n 个随机麻雀位置。"""
    return np.random.rand(n, DIM)
