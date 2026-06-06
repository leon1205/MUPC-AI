# MUPC-AI — 多功能电力电子变换器强化学习模型训练管线

为 MUPC（多功能电力电子变换器）训练基于强化学习的边缘 AI 控制模型，目标部署到 RK3588 NPU，实现毫秒级最优控制。

## 架构概览

```
SMART-DS 数据集 ──→ 环境仿真 (48维观测) ──→ PPO/SAC 训练 ──→ ONNX 导出
   (光伏+负荷)       mupc_env.py            train.py       export_onnx.py
                                                               │
                              ┌────────────────────────────────┘
                              ▼
              RK3588 NPU (RKNN Runtime INT8 推理)
              MUPC AI 引擎 (mupc-ai-engine crate)
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据集（如尚未下载）
python data/download_smart_ds.py

# 3. 运行环境自测
python data_loader.py
python mupc_env.py

# 4. 快速训练测试（5万步，约1分钟）
python train.py --mode MODE-01 --total-timesteps 50000 --no-lstm

# 5. 训练全部5个场景的独立模型
python train_single_modes.py
```

## 观测与动作空间

**观测 (48 维)**，对齐 MUPC AI 引擎 PRD v2.2：

```
索引      内容                      说明
[0..9]    D1 实时数据 (9标量)        SOC/光伏/负荷/电网/变压器/电池/三相电压
[9..24]   D2 光伏预测 (15维)        LSTM 或 Oracle
[24..39]  D2 负荷预测 (15维)        LSTM 或 Oracle
[39..42]  D3 电价 (3维)            当前价/下时段价/时段ID
[42..45]  D4 需量 (3维)            当前需量/合同值/本月峰值
[45..47]  D5 气象 (2维)            辐照/温度
[47]      D6 调度指令 (1维)        有功调度值
```

**动作 (4 维)**：`[p_batt, q_batt, load_shedding, pv_limit]`

## 5 种运行场景

| 场景 | 命令 | 优化目标 | 适用工况 |
|------|------|----------|----------|
| MODE-01 | 农网灌溉 | 最大化光伏消纳 + 防过载 | 农网灌溉季 |
| MODE-02 | 自主套利 | 峰谷价差套利 + 电池保护 | 工商业分时电价 |
| MODE-03 | 需量控制 | 减免需量罚金 + 最小切负荷 | 需量接近合同值 |
| MODE-04 | 虚拟电厂 | 辅助服务收益 + 响应精度 | VPP 调度指令 |
| MODE-05 | 极致绿色 | 最大化绿电比例 + 碳减排 | 绿色消纳目标 |

## 命令行参数

```bash
python train.py [选项]

  --mode MODE              训练模式: MODE-01~MODE-05 或 all (多模式)
  --algo {ppo,sac}         RL 算法 (默认: ppo)
  --total-timesteps N      总训练步数 (默认: 175200)
  --reward-weights W       自定义权重, 如 MODE-01=1.5,0.3,3.0
  --no-lstm                使用 Oracle 预测代替 LSTM
  --lstm-checkpoint PATH   预训练 LSTM 模型路径
  --export-onnx            训练完成后自动导出 ONNX
  --seed N                 随机种子 (默认: 42)
```

## 项目结构

```
MUPC-AI/
├── mupc_env.py              # Gymnasium 环境 (48/49维, 4维动作, 5场景奖励)
├── data_loader.py           # SMART-DS + 中国数据集加载
├── train.py                 # 训练主入口 (SB3 PPO/SAC + NumPy PPO 后备)
├── lstm_model.py            # LSTM 光伏/负荷预测 + Oracle 后备
├── action_validator.py      # 5 条动作约束校验 (ACT-01~05)
├── _ppo_core.py             # 纯 NumPy PPO (MLP + GAE, 零外部依赖)
├── _gym_stub.py             # Gymnasium 最小替代 (无 gymnasium 时降级)
├── export_onnx.py           # ONNX 模型导出
├── train_single_modes.py    # 单场景独立模型训练
├── requirements.txt
│
├── data/
│   ├── download_smart_ds.py     # SMART-DS 数据集下载
│   ├── download_china_data.py   # 中国区域数据生成 (离线)
│   └── smart_ds/                # SMART-DS 数据集
│
├── docs/
│   ├── MUPC/                    # MUPC 项目文档 (PRD + 设计)
│   └── superpowers/specs/       # 训练管线 PRD + 设计文档
│
├── checkpoints/             # 模型权重 (训练产出)
├── tensorboard_logs/        # TensorBoard 日志
└── exported_models/         # ONNX 导出模型
```

## 模型导出与部署

```bash
# 导出 RL 策略为 ONNX
python export_onnx.py --checkpoint checkpoints/MODE-01_model.zip

# 导出 LSTM 预测模型
python export_onnx.py --lstm checkpoints/lstm_model.pt

# INT8 量化 (需要 rknn-toolkit2)
python export_onnx.py --to-rknn
```

导出后的模型部署到 MUPC AI 引擎 (`mupc-ai-engine` crate)，由 RKNN Runtime 在 RK3588 NPU 上执行推理。

## 训练方案建议

| 方案 | 适用场景 | 命令 |
|------|----------|------|
| 单场景快速验证 | 调试/原型 | `python train.py --mode MODE-01 -total-timesteps 50000 --no-lstm` |
| 5 独立模型 (推荐) | 生产部署 | `python train_single_modes.py` |
| 多模式单模型 | 实验对比 | `python train.py --mode all --total-timesteps 200000` |

**推荐生产方案**：5 个独立模型，每个专精一个场景。部署时 MUPC 的 `ModeSelector` 切换场景时加载对应模型，方差低 (±4)，稳定性好。

## 中国数据集

除了 SMART-DS（旧金山），项目包含中国 12 城市数据集生成脚本：

```bash
python data/download_china_data.py    # 纯离线, 约 149MB
```

覆盖北京/上海/广东/四川/西藏/新疆等 12 省市，5 种建筑类型，12 套省级分时电价。数据格式与 SMART-DS 兼容。

## 依赖

- Python 3.9+
- **训练**: numpy, gymnasium, stable-baselines3, torch
- **导出**: onnx, onnxruntime
- **可选**: rknn-toolkit2 (INT8 量化)
- **零依赖降级**: SB3 → NumPy PPO, Gymnasium → _gym_stub

## 相关文档

- [MUPC AI 引擎 PRD v2.2](docs/MUPC/05-MUPC-AI引擎-PRD.md)
- [MUPC AI 引擎设计文档 v2.2](docs/MUPC/05-MUPC-AI引擎-设计文档.md)
- [训练管线 PRD v2.0](docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-PRD.md)
- [训练管线设计文档 v2.0](docs/superpowers/specs/2026-06-06-MUPC-RL训练管线-设计文档.md)
- [MUPC 项目主文档](docs/MUPC/PROJECT-MUPC-项目设计主文档.md)

## License

Internal research project.
