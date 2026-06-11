"""MUPC 配置包."""

from config.config_manager import (
    MupcConfig,
    get_config,
    load_config,
    add_config_args,
)

__all__ = ["MupcConfig", "get_config", "load_config", "add_config_args"]