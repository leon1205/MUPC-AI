"""
目标函数 — 通过 subprocess 调用 train.py 评估超参数组合 — v3.1。
"""

import json
import os
import re
import subprocess
import sys
import hashlib
from typing import Optional


def _hash_params(params: dict) -> str:
    """SHA256 哈希超参数 (用于缓存)。"""
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ObjectiveEvaluator:
    """评估一组超参数 = 调用一次 train.py --train-lstm。

    解析 stdout 输出 "PV_MAPE=X.XXXX LOAD_MAPE=Y.YYYY" 格式。
    """

    def __init__(self, mode: str = "MODE-01", data_source: str = "smartds",
                 train_steps: int = 50000, temp_dir: str = "./mssa_temp/",
                 cache_enabled: bool = True):
        self.mode = mode
        self.data_source = data_source
        self.train_steps = train_steps
        self.temp_dir = temp_dir
        self.cache: dict[str, float] = {}
        self.cache_enabled = cache_enabled
        self.eval_count = 0
        self.cache_hits = 0
        os.makedirs(temp_dir, exist_ok=True)

    def evaluate(self, params: dict) -> float:
        """评估超参数组合, 返回 weighted_MAPE (越小越好)。

        weighted_MAPE = 0.5 * PV_MAPE + 0.5 * LOAD_MAPE
        """
        self.eval_count += 1

        # 缓存检查
        if self.cache_enabled:
            h = _hash_params(params)
            if h in self.cache:
                self.cache_hits += 1
                return self.cache[h]

        # 生成临时 config JSON
        config_path = os.path.join(self.temp_dir, f"mssa_iter_{self.eval_count}.json")
        with open(config_path, "w") as f:
            json.dump(params, f)

        # 构建命令行
        cmd = [
            sys.executable, "train.py",
            "--train-lstm",
            "--no-error-correction",
            "--mode", self.mode,
            "--data-source", self.data_source,
            "--total-timesteps", str(self.train_steps),
            "--config", config_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=600,  # 10 分钟超时
                cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            )
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] iter={self.eval_count}, params={params}")
            return 1e9  # 大惩罚

        stdout = result.stdout + result.stderr

        # 解析 PV_MAPE / LOAD_MAPE
        pv_match = re.search(r"PV_MAPE=([\d.]+)", stdout)
        load_match = re.search(r"LOAD_MAPE=([\d.]+)", stdout)

        if not pv_match or not load_match:
            print(f"  [FAIL] iter={self.eval_count}: 未找到 MAPE 输出")
            print(f"    stdout tail: {stdout[-200:]}")
            return 1e9

        pv_mape = float(pv_match.group(1))
        load_mape = float(load_match.group(1))
        weighted = 0.5 * pv_mape + 0.5 * load_mape

        # 有效性检查
        if weighted <= 0 or weighted > 100:
            weighted = 1e9
        if pv_mape > 50 or load_mape > 50:
            weighted = 1e9  # MAPE > 50% 视为失败

        # 缓存
        if self.cache_enabled:
            h = _hash_params(params)
            self.cache[h] = weighted

        return weighted

    def get_stats(self) -> dict:
        return {
            "total_evaluations": self.eval_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.eval_count - self.cache_hits,
        }
