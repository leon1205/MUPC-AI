"""Grid2Op Backend 选择逻辑。

优先使用 LightSimBackend（C++ 加速），降级到 PandaPowerBackend。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from grid2op.Backend import Backend

# ── 全局降级标志 ────────────────────────────────────────────────

_GRID2OP_AVAILABLE: bool = False
_BACKEND_CLASS: type["Backend"] | None = None
_BACKEND_NAME: str = "unknown"


def _select_backend() -> tuple[type["Backend"], str]:
    """选择 Grid2Op Backend 类型。

    优先顺序：
    1. LightSimBackend（lightsim2grid，C++ 实现，加速 20~50 倍）
    2. PandaPowerBackend（Grid2Op 默认，Python 实现）

    Returns:
        tuple: (Backend 类, 后端名称)
    """
    global _GRID2OP_AVAILABLE, _BACKEND_CLASS, _BACKEND_NAME

    # 尝试 LightSimBackend
    try:
        from lightsim2grid import LightSimBackend

        _GRID2OP_AVAILABLE = True
        _BACKEND_CLASS = LightSimBackend
        _BACKEND_NAME = "lightsim"
        return LightSimBackend, "lightsim"
    except ImportError:
        pass

    # 降级到 PandaPowerBackend
    try:
        from grid2op.Backend import PandaPowerBackend

        _GRID2OP_AVAILABLE = True
        _BACKEND_CLASS = PandaPowerBackend
        _BACKEND_NAME = "pandapower"
        return PandaPowerBackend, "pandapower"
    except ImportError:
        _GRID2OP_AVAILABLE = False
        _BACKEND_CLASS = None
        _BACKEND_NAME = "unavailable"
        return None, "unavailable"


def get_backend_class() -> type["Backend"] | None:
    """获取已选择的 Backend 类。"""
    if _BACKEND_CLASS is None:
        _select_backend()
    return _BACKEND_CLASS


def get_backend_name() -> str:
    """获取当前 Backend名称。"""
    if _BACKEND_NAME == "unknown":
        _select_backend()
    return _BACKEND_NAME


def is_grid2op_available() -> bool:
    """检查 Grid2Op 是否可用。"""
    if _GRID2OP_AVAILABLE is None:
        _select_backend()
    return _GRID2OP_AVAILABLE