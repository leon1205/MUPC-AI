"""
Gymnasium 最小替代 — 当 gymnasium 包不可用时自动降级。

提供 Env 基类和 Box 空间，接口兼容 gymnasium 核心 API,
使 mupc_env.py 在不安装 gymnasium 的情况下也能独立运行。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class Box:
    """连续有界空间，兼容 gymnasium.spaces.Box。"""

    def __init__(self, low, high, shape=None, dtype=np.float32):
        arr_low = np.asarray(low, dtype=dtype)
        arr_high = np.asarray(high, dtype=dtype)
        target_shape = shape if shape is not None else arr_low.shape
        self.low = np.broadcast_to(arr_low, target_shape)
        self.high = np.broadcast_to(arr_high, target_shape)
        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != self.shape:
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self) -> str:
        return f"Box({self.low}, {self.high}, shape={self.shape}, dtype={self.dtype})"


class Env(ABC):
    """Minimal Env base class, compatible with gymnasium.Env core interface."""

    observation_space: Box
    action_space: Box
    metadata: dict[str, Any] = {"render_modes": []}

    @abstractmethod
    def reset(self, seed=None, options=None) -> tuple:
        ...

    @abstractmethod
    def step(self, action) -> tuple:
        ...

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def render(self):
        pass
