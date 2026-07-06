# 技术债务清单 (Technical Debt Register)

**项目**: MUPC-AI2 训练管线
**更新日期**: 2026-07-06 (修复批次: A-1/A-2/A-3/C-1/C-2/C-5 + T-1部分)
**状态**: 基于 5 轮代码审计（51 项问题已修复），18 项债务中已修复 7 项，剩余 11 项

---

## 架构债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| A-1 | ✅ `mupc_env/constants.py` 与 `config/config_manager.py` 物理常量双重定义 | 修改常量时需同步两处，易遗漏 | PhysicalConfig 已添加权威来源注释 | ~~中~~ |
| A-2 | ✅ `config_manager.py` `ActionSpaceConfig` 保留已废弃的 5 维动作字段 | 配置膨胀 | 已清理 load_shed/pv_limit/confidence 废弃字段 | ~~低~~ |
| A-3 | ✅ `config_manager.py` `ObsNormalizationConfig` 范围与 `constants.py` 不一致 | YAML 覆盖配置时传播错误范围 | 已统一为 [-200,200] (constants.py 值) | ~~低~~ |
| A-4 | `mupc_env/core.py` 文件过大 (745行)，包含环境初始化/电压仿真/状态构建/奖励/Welford 多重职责 | 维护困难，单文件变更风险高 | 提取 `_make_env_state` / `_make_reward_dict` 到独立 state_builder 模块 | 低 |

---

## 集成债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| I-1 | VMD 预处理为实验性骨架 — `_apply_vmd()` 已接入但 `prepare_data()` 未使用 IMF 通道扩展 | 启用 VMD 不改变训练数据，浪费计算 | 在 `prepare_data()` 实现 `input_dim × K` 通道扩展逻辑；或安装 vmdpy 替换骨架 | 高 |
| I-2 | 测试文件 `test_modes.py` 缺少 `__name__ == "__main__"` 完整守护 | import 时意外执行训练 | 将业务逻辑包裹入 `main()` 函数 | 中 |
| I-3 | ✅ SB3→SAC fallback 无日志 | 排查困难 | RuntimeError 已保留 PPO/SAC 原始异常信息 | ~~低~~ |

---

## 测试债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| T-1 | ✅ v3.1 核心功能补测试 (部分完成) | 回归风险 | 新增 `tests/test_v31_features.py` (24 tests: QuantileLoss/Log-barrier/k_droop/Welford/MSSA) | ~~高→中~~ |
| T-1b | TCNBlock/TCNFeatureExtractor + ErrorCorrection Bias Gate 仍缺测试 | 回归风险 | 后续补 | 中 |
| T-2 | MIC 特征筛选工具无测试入口（`tools/mic_analysis.py`） | 离线工具质量无保障 | 添加 `--dry-run` 模式和单元测试 | 中 |
| T-3 | MSSA 优化器包无测试（`tools/mssa_optimizer/`） | 核心超参搜索可靠性无保障 | 添加 search_space 编解码单元测试 + mock 目标函数集成测试 | 中 |
| T-4 | `_ppo_core.py` 自测仅覆盖单样本推理，无训练收敛验证 | NumPy PPO 后备路径质量未知 | 添加简单收敛测试（如 CartPole 50episode 达到阈值） | 低 |

---

## 代码质量债务

| ID | 问题 | 影响 | 建议方案 | 优先级 |
|----|------|------|---------|--------|
| C-1 | ✅ SOC 边界半权重未说明理由 | 维护者不理解设计意图 | docstring 已补充 "SOC 有硬约束, log-barrier 仅做预警" | ~~低~~ |
| C-2 | ✅ `search_space.py` 的 `KEY_MAP` 重复定义 | 维护者可能误改 | 已替换为参考注释, 指向 train.py 权威版本 | ~~低~~ |
| C-3 | `tools/mssa_optimizer/objective.py` 的 `cwd` 路径使用相对路径 | pip install 或符号链接场景下路径错误 | 使用环境变量或从 git root 解析 | 低 |
| C-4 | `_ppo_core.py` 中 `obs_dim` 默认值缺少对齐注释 | 未来观测空间变更可能遗漏同步 | 添加注释 "对齐 MupcEnv.observation_space" | 低 |
| C-5 | ✅ `config/mupc_env_config.yaml` 废弃字段 | 配置用户可能误设无效值 | YAML d1_power/d6_dispatch 范围已统一为 [-200,200] | ~~低~~ |

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
| 架构 | 0 | 0 | 1 | 1 |
| 集成 | 1 | 1 | 1 | 3 |
| 测试 | 0 | 3 | 1 | 4 |
| 代码质量 | 0 | 0 | 2 | 2 |
| 文档 | 0 | 1 | 1 | 2 |
| **合计** | **1** | **5** | **6** | **12** |

**高优先级项（建议下一迭代处理）**:
- I-1: VMD 预处理实现真实 IMF 通道扩展
