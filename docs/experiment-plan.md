# AlignReason 实验计划

## 实验问题

> 能否通过模仿学习 + 自搜索让 Qwen3-4B-Instruct 在小模型上获得可验证的推理能力提升？

### 核心动机

- 4B 小模型在端侧部署、高并发 API 上有实用价值
- Qwen3-4B-Instruct 已经经过大量推理训练，增量 SFT 能否进一步撬动提升？
- **模仿（Imitation）** 可以快速达到老师水平，**搜索（Search/Exploration）** 是突破老师天花板的唯一途径

### 关键观察（来自 baseline）

| 现象 | 含义 |
|------|------|
| AMPS_Hard 71.0 分，90% 有 `\boxed{}` 输出 | 计算类题模型能正确格式化，但仍有 29% 算错 |
| math_comp 39.13 分，仅 35% 有 `\boxed{}` | 多步竞赛推理是最大瓶颈，输出长但经常不收敛 |
| olympiad 8.8 分 | 特殊格式（IMO 填空题），模型不适应，忽略此维度 |
| 增大 context 到 16384 后 182 题全部跑通，0 ERROR | 工程瓶颈已解决 |

---

## 路线图：三级渐进

```
Phase 0: 全量 Baseline（当前）
     ↓
Phase 1: Imitation SFT（模仿，快速提升）
     ↓
Phase 2: Rejection + DCO（搜索，突破天花板）
     ↓
Phase 3: Self-Play / GRPO（自进化，探索上限）
```

---

## Phase 0: 全量 Baseline

### 状态

| 类别 | 状态 | 分数 |
|------|------|------|
| math | ✅ 完成 | 39.7（AMPS_Hard 71.0, math_comp 39.13, olympiad 8.8） |
| reasoning | ⏳ 待跑 | - |
| coding | ⏳ 待跑 | - |
| language | ⏳ 待跑 | - |
| data_analysis | ⏳ 待跑 | - |
| instruction_following | ⏳ 待跑 | - |

### 运行命令

```bash
vllm serve /path/to/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64
```

```bash
cd /root/autodl-tmp/LiveBench/livebench
rm -rf data/live_bench/*/*/model_answer/ data/live_bench/*/*/model_judgment/
python run_livebench.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-display-name qwen3-4b-instruct-2507-baseline \
  --api-base http://localhost:8000/v1 --api-key EMPTY \
  --parallel-requests 8 --max-tokens 4096 \
  --force-temperature 0 --livebench-release-option 2026-01-08
```

---

## Phase 1: Imitation SFT

### 目标

用 OpenThoughts3-1.2M 的数据直接做 LoRA SFT，验证：
- 4B 能否从 QwQ-32B 的推理轨迹中受益
- 哪些 domain（math / code / science）收益最大
- 输出长度的变化（是否变啰嗦了）

### 数据策略

| 难度 | 做法 | 数据来源 |
|------|------|---------|
| 简单（baseline 已答对） | 可忽略或保留少量 | OpenThoughts 自筛 |
| 中等（baseline 答错，QwQ 答对） | **重点学习**，Imitation 核心数据 | OpenThoughts 筛选 |
| 困难（QwQ 也答错） | 丢弃或留作搜索起点 | OpenThoughts 筛选 |

采样方案：

| Domain | phase1-小规模 | phase1-全量 | 目的 |
|--------|-------------|------------|------|
| Mathematics | 1K → 10K | math_comp 提升 |
| Code | 0.5K → 5K | coding 提升 |
| Science | 0.3K → 5K | science/language 提升 |

### 烟雾测试（Smoke Test）

| 步骤 | 规模 | 验证点 |
|------|------|--------|
| 采样 | 100 条 | 数据下载、chat template、格式转换 |
| 训练 | 50 步 | LoRA 配置、GPU 显存、梯度累积 |
| 推理 | 1-5 题 | adapter 加载、生成正常 |
| 评估 | math 子集 | pipeline 串通 |

### 训练配置

```yaml
max_seq_length: 4096
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_jproj, o_proj]
learning_rate: 1.0e-5
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
```

### 评估协议

与 baseline 完全相同：
- 同一台机器、同一 vLLM 配置
- 同一 LiveBench release（2026-01-08）
- 同一 max-tokens（4096）、并行（8）
- 同一 benchmark 列表

评估比较：

```text
baseline score → phase1 score → delta
               → Qwen3-4B-Thinking score → gap to thinking variant
```

---

## Phase 2: Rejection Sampling SFT + DCO

### 动机

Phase 1 的 Imitation SFT 上限受限于 QwQ-32B 老师的能力。要让模型超越老师，需要自己的搜索。

### 核心流程

```
1. Search（搜索）：4B 对 N 条题每条采样 K 次（temperature > 0）
     ↓
2. Filter（筛选）：保留至少 1 次答对的题（答案可用 QwQ 或 ground truth 验证）
     ↓
3. Train（训练）：在筛选出的轨迹上 SFT / DCO
     ↓
4. Repeat（迭代）：新模型 → 更难的题也能答对 → 更大搜索空间
```

### 关键参数

| 参数 | phase2-探索 | phase2-深度 | 说明 |
|------|------------|------------|------|
| N（题数） | 1K | 10K | 覆盖度 |
| K（采样数） | 8 | 16-32 | 搜索密度 |
| 温度 | 0.7 | 0.7-1.0 | 探索强度 |
| 验证源 | QwQ-32B 答案 | ground truth | 可信度 |

### DCO（Direct Coverage Optimization）

标准 SFT 和 test-time compute 存在根本冲突：
- SFT 最大化正确答案的 likelihood → 模型越自信
- 越自信 → 多次采样多样性下降 → pass@N 反而下降
- DCO 直接优化 **pass@N** 目标，保持推理路径多样性

实现：DCO loss 替代 CE loss，只需修改训练脚本中的损失函数。

### 对比实验设计

| 实验 | 方法 | 假设 |
|------|------|------|
| A | 纯 Imitation SFT（Phase 1） | 学到 QwQ 模式，快速提升 |
| B | Rejection SFT（CE loss） | 自己搜索 + SFT，应该 > 纯 Imitation |
| C | Rejection SFT + DCO | 保持多样性，pass@N 应该 > B |
| D | 只保留「4B 答错 + QwQ 答对」的数据 | 高质量纠正数据 > 全量数据 |

---

## Phase 3: 探索方向（备选）

### 3a: GRPO（RL 路线）

DeepSeek-R1 的方法，在 3090 上跑 Qwen3-4B：

- advantage = (当前 reward - group mean) / group std
- 不需要 critic model
- 关键是定义正确的 reward：答案正确性 + 格式合规 + 长度惩罚
- 参考 RESTRAIN（2025）：Qwen3-4B 上 AIME25 +140%

### 3b: SPICE 式 Self-Play（对抗路线）

Meta FAIR 2025：

- Challenge/Reasoner 双角色自博弈
- 信息不对称：Challenger 基于文档出题，Reasoner 看不到文档
- Qwen3-4B 上 +9.1 分已验证
- 适用于 OpenThoughts 数据（有 source/domain 字段）

### 3c: Multi-Agent Fine-Tuning（协作路线）

Google DeepMind 2025：

- 多个角色（Generator + Critic）协作推理
- 各自只学自己正确输出的轨迹
- 比单模型自训练更稳定，不容易退化

---

## 时间线估算（单卡 RTX 3090）

| 阶段 | 数据量 | 训练时间 | 评估时间 | 总耗时 |
|------|--------|---------|---------|--------|
| Phase 0: 全量 Baseline | - | - | ~40 min | ~40 min |
| Phase 1: Smoke Test | 100 条 | ~2 min | ~2 min | ~5 min |
| Phase 1: 小规模 | 1K | ~20 min | ~7 min | ~30 min |
| Phase 1: 全量 | 20K | ~4-6 hr | ~7 min | ~5 hr |
| Phase 2: Rejection（1K） | 1K × 8 | 采样 ~2 hr + 训练 ~20 min | ~7 min | ~2.5 hr |
| Phase 2: DCO 对比 | 1K | ~20 min | ~7 min | ~30 min |

---

## 成功标准

### 合格（项目有意义）

- Phase 1 后 math_comp 从 39 → 50+
- 总分相比 baseline 有统计显著提升
- 输出长度没有剧增（不会变啰嗦）

### 良好（有研究价值）

- Phase 2 后 math_comp 达到 55+，接近 Qwen3-4B-Thinking 水平
- 验证了 DCO > CE loss 在 pass@8 上的优势
- 发现数据质量 > 数据数量（2K 纠正数据 ≈ 20K 全量）

### 优秀（有发表潜力）

- 验证了 Imitation + Search 组合的有效性
- 量化了 SFT 和 test-time compute 的交互关系
- 证明小模型 self-play 可以持续多轮不退化

---

## 关键风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| SFT 后 math_comp 没提升 | 低 | 检查数据质量、是否 chat template 出错 |
| SFT 后输出长度剧增但分数不涨 | 中 | 输出长度作为副指标，增加长度惩罚 |
| Rejection sampling 筛不出足够数据 | 中 | 降低筛选门槛（pass@1 → pass@3）或改用软筛选 |
| 3090 显存不够跑 20K × 4096 | 低 | 梯度累积 + checkpointing 已验证 |
| GRPO 在 3090 上不稳定 | 中 | 先回退到 Rejection SFT 兜底 |
