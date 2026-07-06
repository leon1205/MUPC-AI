"""
MIC (最大信息系数) 特征筛选工具 — v3.1。

计算 SMART-DS 数据中各特征与光伏出力/负荷功率的 MIC 值，
输出 Top-K 特征 JSON 供 train.py --mic 参数使用。

用法:
  python tools/mic_analysis.py --data-dir data/smart_ds --top-k 7 --output mic_result.json

输出 JSON 格式 (对齐 train.py load_mic_features()):
  {
    "top_k": 7,
    "features": [
      {"name": "pv_power_lag96", "mic_score": 0.85, "mic_score_load": 0.12, "selected": true},
      ...
    ],
    "method": "MIC (minepy)" or "Pearson (fallback)",
    "data_summary": {"n_samples": 35040, "time_range": "2023-01-01 ~ 2023-12-31"}
  }

降级: minepy 不可用时自动降级到 Pearson 相关系数。
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
from collections import OrderedDict


def _try_import_minepy():
    """尝试导入 minepy, 不可用则返回 None。"""
    try:
        from minepy import MINE
        return MINE
    except ImportError:
        return None


def _load_smart_ds_data(data_dir: str) -> dict:
    """加载 SMART-DS 光伏/负荷/气象数据，返回 numpy 数组字典。"""
    data_path = Path(data_dir)
    data = {}

    # 尝试加载 npz 缓存
    npz_path = data_path / "smart_ds_processed.npz"
    if npz_path.exists():
        loaded = np.load(npz_path)
        return {k: loaded[k] for k in loaded.files}

    # 回退: 调用 data_loader
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from data_loader import SmartDSLoader
        loader = SmartDSLoader(data_dir=data_dir)
        raw = loader.load_all()
        data["pv_power"] = raw["pv_power"].astype(np.float32)
        data["load_power"] = raw["load_power"].astype(np.float32)
        data["solar_irradiance"] = raw["solar_irradiance"].astype(np.float32)
        data["temperature"] = raw["temperature"].astype(np.float32)
        if "hours" in raw:
            data["hours"] = raw["hours"].astype(np.float32)
        print(f"  数据加载完成: {len(data['pv_power'])} 个时间步")
    except Exception as e:
        print(f"[ERROR] SMART-DS 数据加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    return data


def _build_features(data: dict) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """构建特征矩阵 X (N, F) 和目标向量 y_pv, y_load。

    特征包括: 原始值 + 滞后值 + 衍生特征
    """
    n = len(data["pv_power"])
    pv = data["pv_power"]
    load = data["load_power"]
    ghi = data["solar_irradiance"]
    temp = data["temperature"]
    hours = data.get("hours", np.arange(n, dtype=np.float32) * 15 / 60 % 24)

    features = OrderedDict()
    features["pv_power"] = pv
    features["load_power"] = load
    features["solar_irradiance"] = ghi
    features["temperature"] = temp
    features["hour_sin"] = np.sin(hours * 2 * math.pi / 24)
    features["hour_cos"] = np.cos(hours * 2 * math.pi / 24)
    # 滞后特征 (昨日同时刻)
    features["pv_power_lag96"] = np.roll(pv, 96)
    features["load_power_lag96"] = np.roll(load, 96)
    features["ghi_lag96"] = np.roll(ghi, 96)
    features["temp_lag96"] = np.roll(temp, 96)
    # 变化率特征
    features["pv_ramp"] = np.diff(pv, prepend=pv[0])
    features["load_ramp"] = np.diff(load, prepend=load[0])
    # 交互特征
    features["pv_load_ratio"] = np.divide(pv, load + 1e-6, out=np.zeros_like(pv), where=load > 1e-6)

    # 跳过前 96 步 (滞后无效)
    start = 96
    feature_names = list(features.keys())
    X = np.column_stack([v[start:] for v in features.values()])
    y_pv = pv[start + 1:]   # 预测下一步光伏
    y_load = load[start + 1:]  # 预测下一步负荷

    # 对齐长度
    min_len = min(len(X), len(y_pv), len(y_load))
    return X[:min_len], feature_names, y_pv[:min_len], y_load[:min_len]


def _compute_mic(X: np.ndarray, y_pv: np.ndarray, y_load: np.ndarray,
                 feature_names: list[str], mine_class) -> list[dict]:
    """用 minepy 计算 MIC 值。"""
    results = []
    for i, name in enumerate(feature_names):
        x_col = X[:, i]
        try:
            mine_pv = mine_class(alpha=0.6, c=15)
            mine_pv.compute_score(x_col, y_pv)
            mic_pv = float(mine_pv.mic())

            mine_load = mine_class(alpha=0.6, c=15)
            mine_load.compute_score(x_col, y_load)
            mic_load = float(mine_load.mic())
        except Exception:
            mic_pv, mic_load = 0.0, 0.0

        results.append({
            "name": name,
            "mic_score": round(mic_pv, 4),
            "mic_score_load": round(mic_load, 4),
        })
    return results


def _compute_pearson(X: np.ndarray, y_pv: np.ndarray, y_load: np.ndarray,
                     feature_names: list[str]) -> list[dict]:
    """用 Pearson 相关系数作为 MIC 降级方案。"""
    results = []
    for i, name in enumerate(feature_names):
        x_col = X[:, i]
        # 去除 NaN/Inf
        mask = np.isfinite(x_col) & np.isfinite(y_pv)
        if mask.sum() < 2:
            corr_pv, corr_load = 0.0, 0.0
        else:
            corr_pv = abs(float(np.corrcoef(x_col[mask], y_pv[mask])[0, 1]))
            mask_load = np.isfinite(x_col) & np.isfinite(y_load)
            corr_load = abs(float(np.corrcoef(x_col[mask_load], y_load[mask_load])[0, 1]))
        results.append({
            "name": name,
            "mic_score": round(corr_pv, 4),
            "mic_score_load": round(corr_load, 4),
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MIC 特征筛选 — 输出 Top-K 特征 JSON"
    )
    parser.add_argument("--data-dir", type=str, default="data/smart_ds",
                        help="SMART-DS 数据目录 (default: data/smart_ds)")
    parser.add_argument("--top-k", type=int, default=7,
                        help="选取 Top-K 个特征 (default: 7)")
    parser.add_argument("--output", type=str, default="mic_result.json",
                        help="输出 JSON 路径 (default: mic_result.json)")
    parser.add_argument("--npz", type=str, default=None,
                        help="直接使用 .npz 缓存文件")
    args = parser.parse_args()

    print(f"MIC 特征筛选工具 (v3.1)")
    print(f"  数据目录: {args.data_dir}")
    print(f"  Top-K: {args.top_k}")

    # 检测 minepy
    MINE = _try_import_minepy()
    method = "MIC (minepy)" if MINE else "Pearson (fallback)"
    print(f"  方法: {method}")

    # 加载数据
    data = _load_smart_ds_data(args.data_dir)

    # 构建特征
    X, feature_names, y_pv, y_load = _build_features(data)
    print(f"  特征矩阵: {X.shape}, 特征数: {len(feature_names)}")

    # 计算相关性
    if MINE:
        results = _compute_mic(X, y_pv, y_load, feature_names, MINE)
    else:
        results = _compute_pearson(X, y_pv, y_load, feature_names)

    # 按 PV MIC 降序排列
    results.sort(key=lambda r: r["mic_score"], reverse=True)

    # 标记 Top-K 为 selected
    for i, r in enumerate(results):
        r["selected"] = i < args.top_k

    # 构建输出
    output = {
        "top_k": args.top_k,
        "features": results,
        "method": method,
        "data_summary": {
            "n_samples": int(len(X)),
            "n_features": len(feature_names),
        },
    }

    # 写入 JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n  Top-{args.top_k} 特征:")
    for r in results[:args.top_k]:
        print(f"    [{r['mic_score']:.4f}] {r['name']}")

    print(f"\nMIC 结果已保存: {args.output}")
    print(f"  使用方法: python train.py --mic {args.output}")


if __name__ == "__main__":
    main()
