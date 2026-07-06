# 技术债务清单 (Technical Debt Register)

**项目**: MUPC-AI2 训练管线
**更新日期**: 2026-07-06
**状态**: 基于 5 轮代码审计（51 项问题已修复），以下为剩余已知债务

---

## 架构债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| A-1 | `mupc_env/constants.py` 与 `config/config_manager.py` 物理常量双重定义 | 修改常量时需同步两处，易遗漏 | 将 `PhysicalConfig` 默认值改为引用 `constants.py`，或反之统一 | 中 |
| A-2 | `config_manager.py` `ActionSpaceConfig` 保留已废弃的 5 维动作字段 (load_shedding/pv_limit/confidence) | 配置膨胀，YAML 可能写入无效值 | 清理废弃字段，仅保留 `p_ref_norm_*` 和 `k_droop_norm_*` | 低 |
| A-3 | `config_manager.py` `ObsNormalizationConfig` 范围与 `constants.py` 不一致 (d1_power: [-500,500] vs [-200,200]) | YAML 覆盖配置时会传播错误归一化范围 | 统一为 `constants.py` 的值 | 低 |
| A-4 | `mupc_env/core.py` 文件过大 (745行)，包含环境初始化/电压仿真/状态构建/奖励/Welford 多重职责 | 维护困难，单文件变更风险高 | 提取 `_make_env_state` / `_make_reward_dict` 到独立 state_builder 模块 | 低 |

---

## 集成债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| I-1 | VMD 预处理为实验性骨架 — `_apply_vmd()` 已接入但 `prepare_data()` 未使用 IMF 通道扩展 | 启用 VMD 不改变训练数据，浪费计算 | 在 `prepare_data()` 实现 `input_dim × K` 通道扩展逻辑；或安装 vmdpy 替换骨架 | 高 |
| I-2 | 测试文件 `test_modes.py` 缺少 `__name__ == "__main__"` 完整守护 | import 时意外执行训练 | 将业务逻辑包裹入 `main()` 函数 | 中 |
| I-3 | `export_onnx.py` 的 `SB3→SAC` fallback 在文件损坏时静默重试，无日志（第 97-104 行） | 排查困难 | 添加 DEBUG 级别日志记录原始异常 | 低 |

---

## 测试债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| T-1 | v3.1 新增核心功能均无自动化测试：QuantileLoss/P90覆盖率、TCNBlock/TCNFeatureExtractor、ErrorCorrection Bias Gate、Log-barrier safety margin | 回归风险，重构信心低 | 逐模块补测试，优先 QuantileLoss 和 ErrorCorrection | 高 |
| T-2 | MIC 特征筛选工具无测试入口（`tools/mic_analysis.py`） | 离线工具质量无保障 | 添加 `--dry-run` 模式和单元测试 | 中 |
| T-3 | MSSA 优化器包无测试（`tools/mssa_optimizer/`） | 核心超参搜索可靠性无保障 | 添加 search_space 编解码单元测试 + mock 目标函数集成测试 | 中 |
| T-4 | `_ppo_core.py` 自测仅覆盖单样本推理，无训练收敛验证 | NumPy PPO 后备路径质量未知 | 添加简单收敛测试（如 CartPole 50episode 达到阈值） | 低 |

---

## 代码质量债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| C-1 | `mupc_env/rewards.py` `_compute_safety_margin` 中 SOC 边界半权重 (0.5×) 与电压/负载全权重不一致，但未在文档说明理由 | 维护者不理解设计意图 | 在 docstring 中补充 "SOC 有硬约束, log-barrier 仅做预警" 说明 | 低 |
| C-2 | `tools/mssa_optimizer/` 中 `search_space.py` 的 `KEY_MAP` 与 `train.py` 重复定义，MSSA 包内的未被使用 | 维护者可能误改错误副本 | 删除 MSSA 包内的 `KEY_MAP`，统一使用 train.py 中的版本 | 低 |
| C-3 | `tools/mssa_optimizer/objective.py` 的 `cwd` 路径使用相对路径 `os.path.dirname(__file__) / ".." / ".."` | pip install 或符号链接场景下路径错误 | 使用环境变量 `MUPC_PROJECT_ROOT` 或从 git root 解析 | 低 |
| C-4 | `_ppo_core.py` 中 `MLPPolicy.__init__` 的 `obs_dim` 默认值仍为 78（已从 58 更新），但无文档说明此值应与 `mupc_env/core.py` 对齐 | 未来观测空间变更时可能遗漏同步 | 添加注释 "对齐 mupc_env.core.MupcEnv.observation_space" | 低 |
| C-5 | `config/mupc_env_config.yaml` 中 `action_space` section 保留 `load_shed_norm_*` 等已废弃字段 | 配置用户可能误设无效值 | 删除废弃字段，或标记为 deprecated | 低 |

---

## 文档债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| D-1 | 设计文档 §7 (误差修正) 标注为 "v3.1 已集成"，但未记录具体的 Bias Gate 启用条件 (>3% MAPE) 和集成细节 | 新开发者不确定功能边界 | 补充集成接口文档: CLI 参数、checkpoint 路径、ONNX 导出方式 | 中 |
| D-2 | 设计文档 §5.2 (Log-barrier) 和 §5.3 (Welford EMA) 已添加但缺少具体参数说明 (参考边界值、EMA alpha) | 研究者无法复现或调优 | 补充参数表和典型工况计算示例 | 低 |

---

## 统计摘要

| 类别 | 高 | 中 | 低 | 合计 |
|------|----|----|----|----|
| 架构 | 0 | 1 | 3 | 4 |
| 集成 | 1 | 1 | 1 | 3 |
| 测试 | 1 | 2 | 1 | 4 |
| 代码质量 | 0 | 0 | 5 | 5 |
| 文档 | 0 | 1 | 1 | 2 |
| **合计** | **2** | **5** | **11** | **18** |

**高优先级项（建议下一迭代处理）**:
- I-1: VMD 预处理实现真实 IMF 通道扩展
- T-1: v3.1 核心功能补测试（QuantileLoss → ErrorCorrection → Log-barrier）
