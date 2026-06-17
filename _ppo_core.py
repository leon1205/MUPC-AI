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
    """2 层 MLP → actor(tanh) + critic(1维)。

    网络结构:
      Input(obs_dim) → Linear(128) → ReLU → Linear(128) → ReLU
                        ├── actor:  Linear(act_dim=2) → Tanh
                        └── critic: Linear(1)

    v2.15: 2维动作 [p_ref(tanh), k_droop(tanh)]
    load_shedding/pv_limit 下沉至 strategy-engine.
    """

    def __init__(self, obs_dim: int = 58, hidden: list[int] | None = None,
                 act_dim: int = 2):
        if hidden is None:
            hidden = [128, 128]
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.act_dim = act_dim  # 5 维，对齐下游 v2.13

        # 权重初始化
        self.weights: dict[str, np.ndarray] = {}
        prev = obs_dim
        for i, h in enumerate(hidden, 1):
            self.weights[f"fc{i}_w"] = _ortho_init((prev, h), np.sqrt(2.0))
            self.weights[f"fc{i}_b"] = np.zeros(h, dtype=np.float32)
            prev = h

        # Actor head
        last_h = hidden[-1]
        self.weights["actor_w"] = _ortho_init((last_h, self.act_dim), 0.01)
        self.weights["actor_b"] = np.zeros(self.act_dim, dtype=np.float32)
        # 可学习 log_std
        self.log_std = np.full(self.act_dim, -0.5, dtype=np.float32)

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
        """前向传播 → (action_mean, value)。2 维, 对齐下游 v2.15。"""
        latent = self._forward_shared(obs)  # (batch, 128)
        action_mean = latent @ self.weights["actor_w"] + self.weights["actor_b"]

        # 2 维: [p_ref(tanh), k_droop(tanh)]
        action = np.tanh(action_mean)

        value = latent @ self.weights["critic_w"] + self.weights["critic_b"]
        return action, value.ravel()

    def get_action(self, obs: np.ndarray, deterministic: bool = False
                   ) -> tuple[np.ndarray, float, float]:
        """采样动作。

        Returns:
            (action, value_scalar, log_prob_scalar)
        """
        latent = self._forward_shared(obs[np.newaxis, :])  # (1, 128)
        action_mean = (latent @ self.weights["actor_w"] + self.weights["actor_b"]).ravel()

        std = np.exp(self.log_std)
        if deterministic:
            a_raw = action_mean.copy()
        else:
            a_raw = action_mean + np.random.randn(self.act_dim) * std

        # 2 维: [p_ref(tanh), k_droop(tanh)], v2.15 全 tanh
        action = np.tanh(a_raw)
        # log_prob Jacobian: tanh 的 log|det| = log(1 - tanh²(z))
        eps = 1e-7
        log_jac = np.log(eps + 1.0 - action ** 2).sum()

        # Gaussian log_prob + Jacobian correction
        sigma2 = std ** 2
        log_gauss = -0.5 * np.sum((a_raw - action_mean) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        log_prob = float(np.clip(log_gauss + log_jac, -20.0, 20.0))

        value = float((latent @ self.weights["critic_w"] + self.weights["critic_b"]).ravel()[0])
        return action, value, log_prob

    def get_action_batch(self, obs: np.ndarray,
                         deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量采样动作 (用于 rollout 收集)。5 维, 对齐下游 v2.13。"""
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

        self.policy = MLPPolicy(obs_dim=obs_dim, act_dim=5)
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
                    # 提前终止：先将 bootstrap 写入 buf_val[t+1]（GAE 期望位置）
                    _, last_val, _ = self.policy.get_action(obs)
                    buf_val[t + 1] = last_val
                    break

            # 正常循环结束（或提前终止后到这里）：最后一步的 value (bootstrap)
            if steps_done < total_timesteps:
                _, last_val, _ = self.policy.get_action(obs)
                buf_val[t + 1] = last_val

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

        # 2 维 (v2.15): [p_ref(tanh), k_droop(tanh)]
        new_actions = np.tanh(action_mean)

        new_values = (latent @ self.policy.weights["critic_w"]
                      + self.policy.weights["critic_b"]).ravel()

        # 简化 log_prob (用 MSE 代理 ratio)
        action_diff = new_actions - old_actions
        new_logp = -0.5 * np.sum((action_diff / np.exp(self.policy.log_std[:old_actions.shape[1]])) ** 2, axis=-1)

        # ratio
        ratio = np.exp(np.clip(new_logp - old_logp, -10, 10))
        clip_range = cfg["clip_range"]
        clipped = np.clip(ratio, 1 - clip_range, 1 + clip_range)

        # Actor loss (clipped objective)
        loss_a1 = ratio * advantages
        loss_a2 = clipped * advantages
        actor_loss = -np.mean(np.minimum(loss_a1, loss_a2))

        # Entropy bonus (only over env dimensions, not extra policy dims)
        std = np.exp(self.policy.log_std[:old_actions.shape[1]])
        entropy = np.mean(np.sum(np.log(std) + 0.5 * np.log(2 * np.pi * np.e)))

        # Critic loss (MSE)
        critic_loss = np.mean((returns - new_values) ** 2)

        total_loss = actor_loss + cfg["vf_coef"] * critic_loss - cfg["ent_coef"] * entropy

        # ── 解析梯度计算 ──
        # Policy gradient: dL/dμ = ratio * advantage * (a_raw_old - μ) / σ²
        # where a_raw_old is the pre-activation action from the old policy

        std = np.exp(self.policy.log_std[:old_actions.shape[1]])  # only env dimensions
        sigma2 = std ** 2

        # Invert activation to get old pre-activation action (2 维, v2.15 全 tanh)
        eps = 0.999999
        a_raw_old = np.arctanh(np.clip(old_actions, -eps, eps))

        # Gradient of log_prob w.r.t. action_mean = (a_raw - μ) / σ²
        dL_dmu = ratio.reshape(-1, 1) * advantages.reshape(-1, 1) * (a_raw_old - action_mean[:, :old_actions.shape[1]]) / sigma2
        dL_dmu = -dL_dmu / len(obs)  # negative because we maximize, and average

        # Pad to policy.act_dim (extra dimensions don't affect loss)
        if self.policy.act_dim > old_actions.shape[1]:
            dL_dmu_full = np.zeros((dL_dmu.shape[0], self.policy.act_dim), dtype=np.float32)
            dL_dmu_full[:, :old_actions.shape[1]] = dL_dmu
        else:
            dL_dmu_full = dL_dmu

        # ── Backprop through actor head ──
        grad_actor_w = latent.T @ dL_dmu_full
        grad_actor_b = dL_dmu_full.sum(axis=0)

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
        dL_dh = dL_dmu_full @ self.policy.weights["actor_w"].T  # (batch, hidden[-1])

        # ── Backprop through shared layers (动态循环，支持任意层数) ──
        grads = {}
        for key in self.policy.weights:
            grads[key] = np.zeros_like(self.policy.weights[key])

        # 从后向前反向传播
        dL_dh_cur = dL_dh * (pre_acts[-1] > 0)  # ReLU backward at last layer
        for i in range(n_layers, 0, -1):
            grad_fc_w = post_acts[i - 1].T @ dL_dh_cur
            grad_fc_b = dL_dh_cur.sum(axis=0)
            grads[f"fc{i}_w"] = grad_fc_w
            grads[f"fc{i}_b"] = grad_fc_b
            if i > 1:
                dL_dh_cur = dL_dh_cur @ self.policy.weights[f"fc{i}_w"].T
                dL_dh_cur = dL_dh_cur * (pre_acts[i - 2] > 0)  # ReLU backward

        # ── Critic gradient ──
        value_diff = (returns - new_values).reshape(-1, 1)
        grad_critic_w = latent.T @ value_diff / len(obs)
        grad_critic_b = value_diff.mean(axis=0)

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
                     ((a_raw_old - action_mean[:, :old_actions.shape[1]]) ** 2 / sigma2 - 1.0)).sum(axis=0) / len(obs)
        # Only update the env-relevant dimensions; pad zeros for unused policy dimensions
        full_dL_dsigma = np.zeros(self.policy.act_dim, dtype=np.float32)
        full_dL_dsigma[:old_actions.shape[1]] = dL_dsigma
        self.opt_log_std.step_scalar("log_std", self.policy.log_std, -full_dL_dsigma)

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


# ═══════════════════════════════════════════════════════════════
# 自测入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_loader import SmartDSLoader
    from mupc_env import MupcEnv

    print("=" * 52)
    print("  NumPy PPO 自测")
    print("=" * 52)

    # 1. MLPPolicy 5维输出验证 (v2.13)
    print("\n[1] MLPPolicy 5维输出...")
    policy = MLPPolicy(obs_dim=58, act_dim=5)
    obs = np.random.randn(58).astype(np.float32)
    action, value = policy.forward(obs[np.newaxis, :])
    assert action.shape == (1, 5), f"action shape {action.shape} != (1,5)"
    assert -1.0 <= action[0, 0] <= 1.0, f"p_ref {action[0,0]} out of [-1,1]"
    assert -1.0 <= action[0, 1] <= 1.0, f"k_droop {action[0,1]} out of [-1,1]"
    assert 0.0 <= action[0, 2] <= 1.0, f"load {action[0,2]} out of [0,1]"
    assert 0.0 <= action[0, 3] <= 1.0, f"pv {action[0,3]} out of [0,1]"
    assert 0.0 <= action[0, 4] <= 1.0, f"conf {action[0,4]} out of [0,1]"
    print(f"  p_ref={action[0,0]:.3f}, k_droop={action[0,1]:.3f}, load={action[0,2]:.3f} [OK]")

    # 2. get_action 确定性/随机（5维）
    print("[2] get_action 采样...")
    act_det, _, _ = policy.get_action(obs, deterministic=True)
    assert act_det.shape == (5,), f"det shape {act_det.shape}"
    act_stoch, _, _ = policy.get_action(obs, deterministic=False)
    assert act_stoch.shape == (5,), f"stoch shape {act_stoch.shape}"
    print(f"  deterministic: p_ref={act_det[0]:.3f}, k={act_det[1]:.3f}, l={act_det[2]:.3f} [OK]")
    print(f"  stochastic:   p_ref={act_stoch[0]:.3f}, k={act_stoch[1]:.3f}, l={act_stoch[2]:.3f} [OK]")

    # 3. ActionValidator 7 条约束规则 (v2.13)
    print("[3] ActionValidator 约束规则...")
    from action_validator import ActionValidator
    v = ActionValidator()
    # 5维动作: [p_ref, k_droop, load_shedding, pv_limit, confidence]
    act_init = np.array([0.0, 0.0, 0.3, 0.5, 0.5], dtype=np.float32)
    # ACT-01: 小变化不触发 (p_ref 从 0→5kW < 50kW)
    v.validate(act_init, dispatch_p=None)
    _, violated, _ = v.validate(np.array([0.1, 0.0, 0.3, 0.5, 0.5]), dispatch_p=None)
    assert not violated, f"small delta should not trigger ACT-01: {violated}"
    # ACT-07: 调度约束 (p_ref=30kW > dispatch=20kW)
    v.reset()
    v.validate(act_init, dispatch_p=None)
    _, violated, violations = v.validate(np.array([0.6, 0.0, 0.3, 0.5, 0.5]),
                                          dispatch_p=20.0)
    assert violated and "ACT-07" in violations, f"ACT-07 not triggered: {violations}"
    print("  ACT-01 (delta p_ref <= 50kW) [OK]")
    print("  ACT-07 (|p_ref| <= |dispatch|) [OK]")

    # 4. NumPyPPO 训练一步
    print("[4] NumPyPPO 训练一步...")
    loader = SmartDSLoader()
    data = loader.load_all()
    train, _ = loader.split(data)
    env = MupcEnv(train, mode="MODE-01")
    config = {"n_steps": 96, "batch_size": 32, "n_epochs": 2, "lr": 3e-4}
    ppo = NumPyPPO(env, config=config)
    ppo.learn(96)  # 1 episode
    print("  训练一步完成 [OK]")

    print(f"\n[PASS] _ppo_core.py 自测通过")
