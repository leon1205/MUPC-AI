"""
MSSA 搜索结果输出 — v3.1。

生成 mssa_search_result.json 文件，与 downstream §15.5 格式对齐。
"""

import json
import time
from typing import Any
import numpy as np


def format_result(best_params: dict[str, Any],
                  best_objective: float,
                  best_pv_mape: float,
                  best_load_mape: float,
                  convergence: list[float],
                  trajectory: list[list[float]],
                  stats: dict,
                  elapsed: float,
                  config: dict) -> dict:
    """构建 MSSA 搜索结果字典。"""

    quality_flag = "usable" if best_objective < 0.50 else "unusable"

    return {
        "search_metadata": {
            "algorithm": "MSSA (Multi-Strategy Sparrow Search Algorithm)",
            "version": "v3.1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "iterations": config.get("iterations", 0),
            "population": config.get("population", 0),
            **stats,
        },
        "best_hyperparameters": {
            "hidden_size": best_params.get("hidden_size"),
            "num_layers": best_params.get("num_layers"),
            "input_window": best_params.get("input_window"),
            "vmd_k": best_params.get("vmd_k"),
            "vmd_alpha": best_params.get("vmd_alpha"),
            "learning_rate": best_params.get("learning_rate"),
            "batch_size": best_params.get("batch_size"),
            "dropout": best_params.get("dropout"),
            "attn_score": best_params.get("attn_score"),
            "optimizer": best_params.get("optimizer"),
        },
        "best_objective": {
            "weighted_mape": round(best_objective, 6),
            "mape_pv": round(best_pv_mape, 6),
            "mape_load": round(best_load_mape, 6),
        },
        "convergence_curve": [round(v, 6) for v in convergence],
        "per_parameter_trajectory": _format_trajectory(trajectory, config.get("iterations", 0)),
        "quality_flag": quality_flag,
    }


def _format_trajectory(trajectory: list[list[float]], n_iter: int) -> list[dict]:
    """每轮最优参数轨迹 → 按迭代索引的列表。"""
    from .search_space import SEARCH_SPACE
    result = []
    for t in range(min(n_iter, len(trajectory))):
        entry = {"iteration": t}
        vec = trajectory[t]
        for i, (name, _, _, _) in enumerate(SEARCH_SPACE):
            entry[name] = round(float(vec[i]), 6) if i < len(vec) else None
        result.append(entry)
    return result


def save_result(output: dict, path: str) -> None:
    """写入 mssa_search_result.json。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nMSSA 结果已保存: {path}")
