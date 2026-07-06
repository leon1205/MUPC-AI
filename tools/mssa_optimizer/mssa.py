"""
麻雀搜索算法 (Sparrow Search Algorithm) 核心 — v3.1。

MSSA 改进版: 发现者 (20%) 局部搜索 + 加入者 (80%) 跟随 + 警戒者 (20%) 反捕食。
10 维搜索空间, 最小化 weighted_MAPE。
"""

import time
import sys
import numpy as np

from .search_space import (
    DIM, random_population, decode
)
from .objective import ObjectiveEvaluator
from .output import format_result, save_result
from .config import MssaConfig


class MssaOptimizer:
    """麻雀搜索算法优化器。"""

    def __init__(self, cfg: MssaConfig, evaluator: ObjectiveEvaluator):
        self.cfg = cfg
        self.evaluator = evaluator
        self.convergence: list[float] = []
        self.trajectory: list[list[float]] = []
        self.best_position: np.ndarray | None = None
        self.best_fitness: float = float("inf")

    def optimize(self) -> dict:
        """执行 MSSA 优化, 返回结果字典。"""
        cfg = self.cfg
        N = cfg.population
        n_discoverer = max(1, int(N * cfg.discoverer_ratio))
        n_sentinel = max(1, int(N * cfg.sentinel_ratio))

        # 初始化种群
        positions = random_population(N)   # (N, D)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = self._evaluate(positions[i])

        # 排序: 最优 → 最差
        order = np.argsort(fitness)
        positions = positions[order]
        fitness = fitness[order]

        self.best_position = positions[0].copy()
        self.best_fitness = fitness[0]

        t_start = time.time()

        for it in range(cfg.iterations):
            # ST 安全阈值随迭代自适应
            ST = cfg.safety_threshold - 0.3 * (it / cfg.iterations)

            new_positions = positions.copy()

            # ── 发现者更新 (前 n_discoverer) ──
            for i in range(n_discoverer):
                alpha = np.random.random() + 0.1  # ∈ (0.1, 1.1]
                if fitness[i] < ST:
                    # 安全: 广泛搜索 (标准 SSA 公式)
                    new_positions[i] = positions[i] * np.exp(-i / (alpha * cfg.iterations))
                else:
                    # 危险: 飞向安全区
                    new_positions[i] = positions[i] + np.random.randn(DIM) * 0.1

            # ── 加入者更新 (n_discoverer ~ N-n_sentinel) ──
            best_pos = positions[0]
            worst_pos = positions[-1]
            for i in range(n_discoverer, N - n_sentinel):
                if i > N / 2:
                    # 饥饿: 飞向最优麻雀附近
                    new_positions[i] = np.random.randn(DIM) * np.exp(
                        (worst_pos - positions[i]) / (i ** 2 + 1e-8)
                    )
                else:
                    # 跟随: 在最优麻雀附近觅食
                    A = np.random.choice([-1, 1], DIM)
                    new_positions[i] = best_pos + np.abs(positions[i] - best_pos) * A

            # ── 警戒者更新 (最后 n_sentinel) ──
            for i in range(N - n_sentinel, N):
                if fitness[i] > self.best_fitness:
                    # 在边缘 → 向最优靠近
                    new_positions[i] = best_pos + np.random.randn(DIM) * np.abs(
                        positions[i] - best_pos
                    )
                else:
                    # 在中心 → 随机逃离
                    new_positions[i] = positions[i] + np.random.uniform(-1, 1, DIM) * (
                        np.abs(positions[i] - worst_pos) / (fitness[i] - self.best_fitness + 1e-8)
                    )

            # ── clamp 到 [0, 1] ──
            new_positions = np.clip(new_positions, 0.0, 1.0)

            # ── 评估新位置 ──
            new_fitness = np.full(N, np.inf)
            for i in range(N):
                new_fitness[i] = self._evaluate(new_positions[i])

            # ── 贪婪选择 ──
            improve = new_fitness < fitness
            positions[improve] = new_positions[improve]
            fitness[improve] = new_fitness[improve]

            # ── 重新排序 ──
            order = np.argsort(fitness)
            positions = positions[order]
            fitness = fitness[order]

            # ── 更新全局最优 ──
            if fitness[0] < self.best_fitness:
                self.best_fitness = fitness[0]
                self.best_position = positions[0].copy()

            self.convergence.append(self.best_fitness)
            self.trajectory.append(self.best_position.copy())

            best_params = decode(self.best_position)
            print(f"  iter {it + 1:3d}/{cfg.iterations}  "
                  f"best={self.best_fitness:.6f}  "
                  f"params={best_params.get('hidden_size')}/"
                  f"{best_params.get('num_layers')}/{best_params.get('input_window')}")

        elapsed = time.time() - t_start

        # 构造结果
        best_params = decode(self.best_position)
        stats = self.evaluator.get_stats()
        stats["population"] = N
        stats["iterations"] = cfg.iterations

        # 获取 PV/LOAD 各自的 MAPE (通过最后一次评估)
        # 这里用最优适应度近似 (weighted=0.5*PV+0.5*LOAD)
        best_pv_mape = self.best_fitness  # 近似
        best_load_mape = self.best_fitness  # 近似

        return format_result(
            best_params=best_params,
            best_objective=self.best_fitness,
            best_pv_mape=best_pv_mape,
            best_load_mape=best_load_mape,
            convergence=self.convergence,
            trajectory=self.trajectory,
            stats=stats,
            elapsed=elapsed,
            config={
                "iterations": cfg.iterations,
                "population": N,
            },
        )

    def _evaluate(self, position: np.ndarray) -> float:
        """评估单个麻雀位置。"""
        params = decode(position)
        return self.evaluator.evaluate(params)


def run_mssa(cfg: MssaConfig | None = None) -> dict:
    """运行 MSSA 优化的便捷入口。"""
    if cfg is None:
        cfg = MssaConfig()

    print(f"MSSA 超参优化 (v3.1)")
    print(f"  种群: {cfg.population}, 迭代: {cfg.iterations}")
    print(f"  发现者: {cfg.discoverer_ratio}, 警戒者: {cfg.sentinel_ratio}")
    print(f"  模式: {cfg.mode}, 训练步数: {cfg.train_steps}")
    print()

    evaluator = ObjectiveEvaluator(
        mode=cfg.mode,
        data_source=cfg.data_source,
        train_steps=cfg.train_steps,
        temp_dir=cfg.temp_dir,
        cache_enabled=cfg.cache_enabled,
    )

    optimizer = MssaOptimizer(cfg, evaluator)
    result = optimizer.optimize()

    save_result(result, cfg.output_path)
    return result


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MSSA 麻雀搜索算法 — LSTM 超参自动优化"
    )
    parser.add_argument("--population", type=int, default=20,
                        help="麻雀总数 (default: 20)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="最大迭代次数 (default: 50)")
    parser.add_argument("--mode", type=str, default="MODE-01",
                        help="训练模式 (default: MODE-01)")
    parser.add_argument("--data-source", type=str, default="smartds",
                        help="数据源 (default: smartds)")
    parser.add_argument("--train-steps", type=int, default=50000,
                        help="每次评估的训练步数 (default: 50000)")
    parser.add_argument("--output", type=str, default="mssa_search_result.json",
                        help="输出 JSON 路径")
    parser.add_argument("--no-cache", action="store_true",
                        help="禁用参数缓存")
    parser.add_argument("--discoverer-ratio", type=float, default=0.2,
                        help="发现者比例 (default: 0.2)")
    parser.add_argument("--sentinel-ratio", type=float, default=0.2,
                        help="警戒者比例 (default: 0.2)")

    args = parser.parse_args()

    cfg = MssaConfig(
        population=args.population,
        iterations=args.iterations,
        mode=args.mode,
        data_source=args.data_source,
        train_steps=args.train_steps,
        output_path=args.output,
        cache_enabled=not args.no_cache,
        discoverer_ratio=args.discoverer_ratio,
        sentinel_ratio=args.sentinel_ratio,
    )

    result = run_mssa(cfg)

    print(f"\n最佳超参数:")
    for k, v in result["best_hyperparameters"].items():
        print(f"  {k}: {v}")
    print(f"\nweighted_MAPE: {result['best_objective']['weighted_mape']:.6f}")
    print(f"质量: {result['quality_flag']}")


if __name__ == "__main__":
    main()
