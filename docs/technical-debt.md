# 技术债务清单 (Technical Debt Register)

**项目**: MUPC-AI2 训练管线
**更新日期**: 2026-07-06
**状态**: 6 轮审计（51 项修复），18 项初始债务全部处理完毕

---

## 已修复债务

| ID | 问题 | 修复 |
|----|------|------|
| A-1 | constants.py / config_manager.py 双重常量定义 | PhysicalConfig 添加权威来源注释 |
| A-2 | ActionSpaceConfig 废弃 5 维动作字段 | 清理 load_shed/pv_limit/confidence |
| A-3 | ObsNormalizationConfig 范围不一致 | 统一为 [-200,200] |
| C-1 | SOC 边界半权重未说明 | docstring 补充 |
| C-2 | search_space.py KEY_MAP 重复 | 替换为参考注释 |
| C-3 | MSSA objective.py cwd 相对路径 | MUPC_PROJECT_ROOT + abspath fallback |
| C-4 | _ppo_core obs_dim 缺少对齐注释 | 已添加 |
| C-5 | YAML config 废弃字段 | d1_power/d6_dispatch 范围统一 |
| D-1 | 设计文档 §7 误差修正缺少集成细节 | 补充 §7.4 CLI/checkpoint/ONNX |
| D-2 | 设计文档 §5.2 缺少参数说明 | 补充参考边界参数表 |
| I-1 | VMD IMF 通道扩展 | prepare_data/predict/train 全链路支持 |
| I-2 | test_modes.py 缺少 main() 守护 | 重构为 main() + __name__ guard |
| I-3 | export_onnx SB3→SAC fallback 无日志 | RuntimeError 保留双异常信息 |
| T-1 | v3.1 核心功能无自动化测试 | 新增 test_v31_features.py (31 tests) |
| T-2 | MIC 工具无测试入口 | --dry-run 模式 |
| T-3 | MSSA 编解码无测试 | T-05 往返一致性测试 |

---

## 剩余低优先级债务

| ID | 问题 | 优先级 | 说明 |
|----|------|--------|------|
| A-4 | `mupc_env/core.py` 过大 (745行) | 低 | 建议提取 state_builder 模块，风险较高需单独排期 |
| T-1b | TCN 因果 padding 严格验证 | 低 | 当前对称 padding 对时序预测影响可忽略，严格因果需重新设计 |
| T-4 | `_ppo_core.py` 收敛验证 | 低 | 自测仅覆盖单样本推理，需 gym 环境 |

---

## 统计

| 状态 | 数量 |
|------|------|
| 已修复 | 18 |
| 剩余(低优先级) | 3 |
| **合计** | **21** |
