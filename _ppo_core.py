"""
纯 NumPy PPO 实现 — stable-baselines3 不可用时的后备。

特性:
  - 2 层 MLP (128 神经元) 策略网络
  - 混合输出: Tanh(A1,A2) + Sigmoid(A3,A4)
  - GAE advantage 估计
  - Clipped objective + value function loss
  - Momentum SGD 优化器
  - 零外部依赖 (仅 NumPy)
"""

import time
import numpy as np


# ═══════════════════════════════════════════════════════════════
# MLP 策略网络
# ═══════════════════════════════════════════════════════════════

def _ortho_init(shape: tuple, scale: float = 1.0) -> np.ndarray:
    """正交初始化。"""
    flat = np.random.randn(*shape).astype(np.float32)
    if len(shape) >= 2:
        u, _, v = np.linalg.svd(flat.reshape(shape[0], -1), full_matrices=False)
        q = u if u.shape == flat.reshape(shape[0], -1).shape else v
        flat = q.reshape(shape) * scale
    return flat


class MLPPolicy:
    """2 层 MLP → actor(4维混合输出) + critic(1维)。

    网络结构:
      Input(obs_dim) → Linear(128) → ReLU → Linear(128) → ReLU
                        ├── actor:  Linear(4) → [Tanh(:2), Sigmoid(2:)]
                        └── critic: Linear(1)
    """

    def __init__(self, obs_dim: int = 48, hidden: list[int] | None = None,
                 act_dim: int = 4):
        if hidden is None:
            hidden = [128, 128]
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.act_dim = act_dim

        # 权重初始化
        self.weights: dict[str, np.ndarray] = {}
        prev = obs_dim
        for i, h in enumerate(hidden, 1):
            self.weights[f"fc{i}_w"] = _ortho_init((prev, h), np.sqrt(2.0))
            self.weights[f"fc{i}_b"] = np.zeros(h, dtype=np.float32)
            prev = h

        # Actor head
        last_h = hidden[-1]
        self.weights["actor_w"] = _ortho_init((last_h, act_dim), 0.01)
        self.weights["actor_b"] = np.zeros(act_dim, dtype=np.float32)
        # 可学习 log_std
        self.log_std = np.full(act_dim, -0.5, dtype=np.float32)

        # Critic head
        self.weights["critic_w"] = _ortho_init((last_h, 1), 1.0)
        self.weights["critic_b"] = np.zeros(1, dtype=np.float32)

    # ── 前向传播 ──────────────────────────────────────

    def _forward_shared(self, obs: np.ndarray):
        """共享层前向传播, 返回最后隐藏层激活值。"""
        x = obs.reshape(-1, self.obs_dim)  # (batch, obs_dim)
        n_layers = len(self.hidden)
        for i in range(1, n_layers + 1):
            x = x @ self.weights[f"fc{i}_w"] + self.weights[f"fc{i}_b"]
            x = np.maximum(0, x)  # ReLU
        return x

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """前向传播 → (action_mean, value)。"""
        latent = self._forward_shared(obs)  # (batch, 128)
        action_mean = latent @ self.weights["actor_w"] + self.weights["actor_b"]
        # A1,A2: Tanh; A3,A4: Sigmoid
        a1a2 = np.tanh(action_mean[:, :2])
        a3a4 = 1.0 / (1.0 + np.exp(-action_mean[:, 2:]))  # sigmoid
        action = np.concatenate([a1a2, a3a4], axis=-1)
        value = latent @ self.weights["critic_w"] + self.weights["critic_b"]
        return action, value.ravel()

    def get_action(self, obs: np.ndarray, deterministic: bool = False
                   ) -> tuple[np.ndarray, float, float]:
        """采样动作。

        Returns:
            (action_4d, value_scalar, log_prob_scalar)
        """
        latent = self._forward_shared(obs[np.newaxis, :])  # (1, 128)
        action_mean = (latent @ self.weights["actor_w"] + self.weights["actor_b"]).ravel()

        std = np.exp(self.log_std)
        if deterministic:
            noise = 0.0
        else:
            noise = np.random.randn(self.act_dim) * std

        # 混合激活
        a1a2 = np.tanh(action_mean[:2] + noise[:2])
        a3a4 = 1.0 / (1.0 + np.exp(-(action_mean[2:] + noise[2:])))
        action = np.concatenate([a1a2, a3a4])

        # 对数概率 (忽略 Tanh/Sigmoid 内部的 Jacobian 修正, 简化版)
        log_prob = -0.5 * np.sum((noise / std) ** 2 + 2.0 * np.log(std) + np.log(2 * np.pi))
        log_prob = float(np.clip(log_prob, -20.0, 20.0))

        value = float((latent @ self.weights["critic_w"] + self.weights["critic_b"]).ravel()[0])
        return action, value, log_prob

    def get_action_batch(self, obs: np.ndarray,
                         deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量采样动作 (用于 rollout 收集)。"""
        n = len(obs)
        actions = np.zeros((n, self.act_dim), dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        log_probs = np.zeros(n, dtype=np.float32)
        for i in range(n):
            a, v, lp = self.get_action(obs[i], deterministic)
            actions[i] = a
            values[i] = v
            log_probs[i] = lp
        return actions, values, log_probs

    def get_weights(self) -> dict:
        d = {k: v.copy() for k, v in self.weights.items()}
        d["log_std"] = self.log_std.copy()
        return d

    def set_weights(self, weights: dict) -> None:
        for k in self.weights:
            if k in weights:
                self.weights[k] = weights[k].copy()
        if "log_std" in weights:
            self.log_std = weights["log_std"].copy()


# ═══════════════════════════════════════════════════════════════
# Momentum SGD
# ═══════════════════════════════════════════════════════════════

class MomentumSGD:
    def __init__(self, lr: float = 3e-4, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities: dict[str, np.ndarray] = {}

    def step(self, weights: dict[str, np.ndarray],
             grads: dict[str, np.ndarray]) -> None:
        for key in grads:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(grads[key])
            self.velocities[key] = (self.momentum * self.velocities[key]
                                    - self.lr * grads[key])
            weights[key] += self.velocities[key]

    def step_scalar(self, name: str, value: np.ndarray,
                    grad: np.ndarray) -> None:
        """更新标量参数 (如 log_std)。"""
        if name not in self.velocities:
            self.velocities[name] = np.zeros_like(grad)
        self.velocities[name] = (self.momentum * self.velocities[name]
                                 - self.lr * grad)
        value += self.velocities[name]


# ═══════════════════════════════════════════════════════════════
# GAE
# ═══════════════════════════════════════════════════════════════

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                dones: np.ndarray, gamma: float = 0.99,
                lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """GAE advantage 估计。

    Args:
        rewards:  (T,)
        values:   (T+1,) — 包含最后一步的 bootstrap value
        dones:    (T,)
        gamma, lam: 折扣因子和 GAE lambda

    Returns:
        (advantages (T,), returns (T,))
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_adv = 0.0
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    returns = advantages + values[:T]
    return advantages, returns


# ═══════════════════════════════════════════════════════════════
# PPO
# ═══════════════════════════════════════════════════════════════

PPO_DEFAULTS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "lr": 3e-4,
}


class NumPyPPO:
    """纯 NumPy PPO 实现。"""

    def __init__(self, env, obs_dim: int | None = None,
                 config: dict | None = None):
        self.env = env
        self.cfg = {**PPO_DEFAULTS, **(config or {})}
        obs_dim = obs_dim or env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.policy = MLPPolicy(obs_dim=obs_dim, act_dim=act_dim)
        self.opt_weights = MomentumSGD(lr=self.cfg["lr"])
        self.opt_log_std = MomentumSGD(lr=self.cfg["lr"])

    def learn(self, total_timesteps: int, callback=None) -> dict:
        """主训练循环。

        Returns:
            {"steps": [...], "rewards": [...], "losses": [...]}
        """
        cfg = self.cfg
        obs, _ = self.env.reset()
        steps_done = 0
        episode_reward = 0.0
        episode_count = 0
        log = {"steps": [], "rewards": [], "losses": [], "actor_loss": [], "critic_loss": []}

        t_start = time.time()

        while steps_done < total_timesteps:
            # ── Rollout 收集 ──
            buf_obs = np.zeros((cfg["n_steps"], self.policy.obs_dim), dtype=np.float32)
            buf_act = np.zeros((cfg["n_steps"], self.policy.act_dim), dtype=np.float32)
            buf_rew = np.zeros(cfg["n_steps"], dtype=np.float32)
            buf_val = np.zeros(cfg["n_steps"] + 1, dtype=np.float32)
            buf_done = np.zeros(cfg["n_steps"], dtype=np.bool_)
            buf_logp = np.zeros(cfg["n_steps"], dtype=np.float32)

            for t in range(cfg["n_steps"]):
                if steps_done % 10000 == 0 and t == 0:
                    elapsed = time.time() - t_start
                    print(f"  步数 {steps_done}/{total_timesteps} "
                          f"| 已训练 {episode_count} episodes "
                          f"| 耗时 {elapsed:.0f}s")

                buf_obs[t] = obs
                act, val, logp = self.policy.get_action(obs)
                buf_act[t] = act
                buf_val[t] = val
                buf_logp[t] = logp

                obs, rew, term, trunc, _ = self.env.step(act)
                buf_rew[t] = rew
                buf_done[t] = term or trunc
                episode_reward += rew
                steps_done += 1

                if term or trunc:
                    episode_count += 1
                    log["steps"].append(steps_done)
                    log["rewards"].append(episode_reward)
                    episode_reward = 0.0
                    obs, _ = self.env.reset()
                    if callback is not None:
                        callback(episode_count, steps_done, log["rewards"][-1])

                if steps_done >= total_timesteps:
                    break

            # 最后一步的 value (bootstrap)
            _, last_val, _ = self.policy.get_action(obs)
            buf_val[t + 1 if steps_done < total_timesteps else t] = last_val

            # ── GAE ──
            advantages, returns = compute_gae(
                buf_rew[:t+1], buf_val[:t+2], buf_done[:t+1],
                cfg["gamma"], cfg["gae_lambda"],
            )

            # 标准化 advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            n_samples = t + 1
            indices = np.arange(n_samples)

            # ── PPO 更新 ──
            for epoch in range(cfg["n_epochs"]):
                np.random.shuffle(indices)
                for start in range(0, n_samples, cfg["batch_size"]):
                    batch_idx = indices[start:start + cfg["batch_size"]]
                    loss_a, loss_c = self._update_step(
                        buf_obs[batch_idx], buf_act[batch_idx],
                        advantages[batch_idx], returns[batch_idx],
                        buf_logp[batch_idx],
                    )

            log["losses"].append(loss_a + loss_c)
            log["actor_loss"].append(loss_a)
            log["critic_loss"].append(loss_c)

        total_time = time.time() - t_start
        print(f"\n  训练完成: {total_timesteps} steps, "
              f"{episode_count} episodes, {total_time:.0f}s")
        return log

    def _update_step(self, obs, old_actions, advantages, returns, old_logp):
        """一次 mini-batch PPO 更新。"""
        cfg = self.cfg
        # 前向
        latent = self.policy._forward_shared(obs)
        action_mean = latent @ self.policy.weights["actor_w"] + self.policy.weights["actor_b"]
        a1a2 = np.tanh(action_mean[:, :2])
        a3a4 = 1.0 / (1.0 + np.exp(-action_mean[:, 2:]))
        new_actions = np.concatenate([a1a2, a3a4], axis=-1)
        new_values = (latent @ self.policy.weights["critic_w"]
                      + self.policy.weights["critic_b"]).ravel()

        # 简化 log_prob (用 MSE 代理 ratio)
        # 实际 ratio ≈ exp(-0.5 * (new - old)² / σ²) 但简化用 action 距离
        action_diff = new_actions - old_actions
        new_logp = -0.5 * np.sum((action_diff / np.exp(self.policy.log_std)) ** 2, axis=-1)

        # ratio
        ratio = np.exp(np.clip(new_logp - old_logp, -10, 10))
        clip_range = cfg["clip_range"]
        clipped = np.clip(ratio, 1 - clip_range, 1 + clip_range)

        # Actor loss (clipped objective)
        loss_a1 = ratio * advantages
        loss_a2 = clipped * advantages
        actor_loss = -np.mean(np.minimum(loss_a1, loss_a2))

        # Entropy bonus
        std = np.exp(self.policy.log_std)
        entropy = np.mean(np.sum(np.log(std) + 0.5 * np.log(2 * np.pi * np.e)))

        # Critic loss (MSE)
        critic_loss = np.mean((returns - new_values) ** 2)

        total_loss = actor_loss + cfg["vf_coef"] * critic_loss - cfg["ent_coef"] * entropy

        # ── 解析梯度计算 ──
        # Policy gradient: dL/dμ = ratio * advantage * (a_raw_old - μ) / σ²
        # where a_raw_old is the pre-activation action from the old policy

        std = np.exp(self.policy.log_std)  # (4,)
        sigma2 = std ** 2

        # Invert activation to get old pre-activation action
        a_raw_old = np.zeros_like(old_actions)
        # A1,A2: tanh → atanh
        eps = 0.999999
        a_raw_old[:, :2] = np.arctanh(np.clip(old_actions[:, :2], -eps, eps))
        # A3,A4: sigmoid → logit
        a_raw_old[:, 2:] = np.log(np.clip(old_actions[:, 2:], 1e-7, 1-1e-7) /
                                   (1.0 - np.clip(old_actions[:, 2:], 1e-7, 1-1e-7)))

        # Gradient of log_prob w.r.t. action_mean = (a_raw - μ) / σ²
        dL_dmu = ratio.reshape(-1, 1) * advantages.reshape(-1, 1) * (a_raw_old - action_mean) / sigma2
        dL_dmu = -dL_dmu / len(obs)  # negative because we maximize, and average

        # ── Backprop through actor head ──
        grad_actor_w = latent.T @ dL_dmu
        grad_actor_b = dL_dmu.sum(axis=0)

        # ── Backprop through shared layers ──
        # Re-run forward with mask tracking
        x = obs.reshape(-1, self.policy.obs_dim)
        pre_acts = []  # pre-activation values for each ReLU
        post_acts = [x]  # post-activation values

        n_layers = len(self.policy.hidden)
        for i in range(1, n_layers + 1):
            pre = post_acts[-1] @ self.policy.weights[f"fc{i}_w"] + self.policy.weights[f"fc{i}_b"]
            pre_acts.append(pre)
            post = np.maximum(0, pre)  # ReLU
            post_acts.append(post)

        # Gradient from actor head into last hidden layer
        dL_dh = dL_dmu @ self.policy.weights["actor_w"].T  # (batch, hidden[-1])

        # Backprop through fc2
        dL_dh = dL_dh * (pre_acts[-1] > 0)  # ReLU backward
        grad_fc2_w = post_acts[-2].T @ dL_dh
        grad_fc2_b = dL_dh.sum(axis=0)

        # Backprop through fc1
        dL_dx = dL_dh @ self.policy.weights[f"fc2_w"].T
        dL_dx = dL_dx * (pre_acts[-2] > 0)  # ReLU backward
        grad_fc1_w = post_acts[-3].T @ dL_dx
        grad_fc1_b = dL_dx.sum(axis=0)

        # ── Critic gradient ──
        value_diff = (returns - new_values).reshape(-1, 1)
        grad_critic_w = latent.T @ value_diff / len(obs)
        grad_critic_b = value_diff.mean(axis=0)

        # ── Assemble gradients ──
        grads = {}
        for key in self.policy.weights:
            grads[key] = np.zeros_like(self.policy.weights[key])

        grads["fc1_w"] = grad_fc1_w
        grads["fc1_b"] = grad_fc1_b
        grads["fc2_w"] = grad_fc2_w
        grads["fc2_b"] = grad_fc2_b
        grads["actor_w"] = grad_actor_w
        grads["actor_b"] = grad_actor_b
        grads["critic_w"] = cfg["vf_coef"] * grad_critic_w
        grads["critic_b"] = cfg["vf_coef"] * grad_critic_b

        # ── Gradient clipping ──
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > cfg["max_grad_norm"]:
            scale = cfg["max_grad_norm"] / (total_norm + 1e-8)
            for k in grads:
                grads[k] *= scale

        # ── Parameter update ──
        self.opt_weights.step(self.policy.weights, grads)

        # Update log_std
        dL_dsigma = (ratio.reshape(-1, 1) * advantages.reshape(-1, 1) *
                     ((a_raw_old - action_mean) ** 2 / sigma2 - 1.0)).sum(axis=0) / len(obs)
        self.opt_log_std.step_scalar("log_std", self.policy.log_std, -dL_dsigma)

        return float(actor_loss), float(critic_loss)

    def save_weights(self, path: str) -> None:
        """保存权重到 .npz。"""
        w = self.policy.get_weights()
        np.savez(path, **{k: v for k, v in w.items()})

    def load_weights(self, path: str) -> None:
        """从 .npz 加载权重。"""
        data = np.load(path)
        weights = dict(data)
        self.policy.set_weights(weights)
