"""
VMD (Variational Mode Decomposition) 信号分解器 (v3.0 R2).

用途: 训练阶段对输入窗口执行 VMD 分解, 将原始序列分解为 K 个子模态 (IMF).
     部署时 VMD 由 MUPC 推理端 CPU 执行, 训练管线仅训练/导出 VMD 兼容模型.

K 值: 光伏 4~6, 负荷 5~8 (由 MSSA 搜索确定).
参考试验: alpha=2000.0, tau=0.0, tol=1e-6, max_iter=500.

注意: VMD 是 R2 可选特性, 由 LSTMTrainer config["vmd_enabled"] 控制启用.
      当前为骨架实现, 训练时 VMD 通过外部库 (vmdpy) 或纯 NumPy 分解.
"""

import numpy as np
from typing import Optional


class VmdDecomposer:
    """VMD 分解包装器.

    训练阶段: 对每条样本的输入窗口执行 VMD 分解, 各 IMF 分别送入 LSTM.
    部署阶段: 不参与 (推理端 CPU 做 VMD, 训练管线仅提供 K 通道 ONNX 模型).

    K 值: 光伏 4~6, 负荷 5~8.
    """

    def __init__(self, K: int = 5, alpha: float = 2000.0, tau: float = 0.0,
                 tol: float = 1e-6, max_iter: int = 500,
                 dc: bool = False, init_mode: int = 1):
        """
        Args:
            K: 模态数 (默认 5)
            alpha: 带宽约束参数 (默认 2000)
            tau: 噪声容忍度 (默认 0 = 无噪声)
            tol: 收敛容差 (默认 1e-6)
            max_iter: 最大迭代次数 (默认 500)
            dc: 是否保留直流分量 (默认 False)
            init_mode: 初始化模式 (0=全零, 1=均匀分布, 2=随机)
        """
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.dc = dc
        self.init_mode = init_mode

    def decompose(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """将 (T,) 信号分解为 (K, T) IMF 分量.

        Args:
            signal: 输入序列, shape (T,)

        Returns:
            IMF 分量, shape (K, T), 或 None 若分解失败
        """
        try:
            return self._vmd_numpy(signal)
        except Exception:
            return None

    def _vmd_numpy(self, signal: np.ndarray) -> np.ndarray:
        """纯 NumPy VMD 实现 (骨架).

        实际使用时替换为 vmdpy 库调用:
          from vmdpy import VMD
          u, u_hat, omega = VMD(signal, self.alpha, self.tau,
                                self.K, self.dc, self.init_mode, self.tol)
          return u  # (K, T)

        当前返回等间隔正弦分解近似 (仅用于接口验证, 非实际 VMD).
        """
        T = len(signal)
        imfs = np.zeros((self.K, T), dtype=np.float64)
        # 简化为等间隔正弦波 + 残差 (骨架, 实际用 vmdpy 替换)
        for k in range(self.K - 1):
            freq = (k + 1) * 2 * np.pi * 0.1 / T
            imfs[k] = np.sin(freq * np.arange(T)) * np.std(signal) * 0.5
        imfs[-1] = signal - imfs[:-1].sum(axis=0)
        return imfs.astype(np.float32)

    def decompose_batch(self, X: np.ndarray, feature_idx: int = 0) -> np.ndarray:
        """对批次数据的指定特征列执行 VMD 分解.

        Args:
            X: (N, T, D) 输入批次
            feature_idx: 待分解的特征列索引 (默认 0 = pv_power)

        Returns:
            (N, K, T) 各样本的 IMF 分量
        """
        N, T, _ = X.shape
        imfs = np.zeros((N, self.K, T), dtype=np.float32)
        for i in range(N):
            result = self.decompose(X[i, :, feature_idx])
            if result is not None:
                imfs[i] = result
            else:
                # fallback: 复制原始信号为 K 个通道
                imfs[i] = np.tile(X[i, :, feature_idx], (self.K, 1))
        return imfs


# ── 配置默认值 ──────────────────────────────────────────────────

VMD_DEFAULT_CONFIG = {
    "enabled": False,       # 默认禁用, R2 由 --config 启用
    "K": 5,
    "alpha": 2000.0,
    "tau": 0.0,
    "tol": 1e-6,
    "max_iter": 500,
    "dc": False,
    "init_mode": 1,
}
