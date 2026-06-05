# AlignReason

AlignReason 是一个小规模后训练实验：使用高质量显式推理数据对 4B instruct 模型做 LoRA SFT，并用 LiveBench 验证推理、数学、代码、科学等能力是否提升。

## 实验问题

OpenThoughts 的高质量 reasoning trace 能否进一步提升 `Qwen/Qwen3-4B-Instruct-2507` 在 LiveBench 上的表现？

本项目的微调对象是：

- `Qwen/Qwen3-4B-Instruct-2507`

对比对象包括：

- 微调前的 `Qwen/Qwen3-4B-Instruct-2507`
- 同规模 thinking 参考模型 `Qwen/Qwen3-4B-Thinking-2507`
- LiveBench 公榜上的大模型结果

这样可以把实验拆成两层：先看 LoRA 相对原始 instruct 模型是否带来增益，再看它在公榜大模型坐标系里的位置。

## 初始方案

| 项目 | 选择 |
| --- | --- |
| 微调基座 | `Qwen/Qwen3-4B-Instruct-2507` |
| 同规模参考 | `Qwen/Qwen3-4B-Thinking-2507` |
| 公榜参考 | LiveBench leaderboard 上的大模型 |
| 数据集 | `open-thoughts/OpenThoughts3-1.2M` |
| 采样量 | smoke test 后扩展到 20,000 条 |
| 数据形式 | 带 assistant reasoning trace 的 chat messages |
| 微调方式 | LoRA SFT |
| 目标硬件 | 单卡 RTX 3090 24GB |
| 第一版上下文 | 4096 |
| 后续尝试 | 显存允许时测试 8192 |
| 评测 | 微调前后都跑 LiveBench |

## 为什么微调 Instruct 模型

`Qwen3-4B-Instruct-2507` 是一个较强的 4B non-thinking instruct 模型，已经针对推理、数学、科学、代码和指令跟随做过优化。因此本项目不是测试“普通模型能否学会推理”，而是测试一个更窄的问题：

> 显式 reasoning trace SFT 能否进一步提升一个已经做过推理优化的 non-thinking instruct 模型？

`Qwen3-4B-Thinking-2507` 作为同规模参考很有价值，但不作为第一阶段微调对象。它本身默认长推理，直接微调它会让实验结论更难解释。

## 仓库结构

```text
alignreason/
  README.md
  requirements.txt
  configs/
    experiment.yaml
  docs/
    environment.md
    livebench-baseline.md
  scripts/
```

第一批实现目标：

1. 采样 OpenThoughts 并保存为 JSONL；
2. 用 100 条样本跑 LoRA smoke train；
3. 用 20K 样本训练正式 adapter；
4. 用同一套 LiveBench 流程评测 base 和 fine-tuned 模型。

## 环境

创建 Python 环境：

```bash
cd /path/to/alignreason
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Qwen3 需要较新的 Transformers。模型卡提示 `transformers<4.51.0` 可能出现 `KeyError: 'qwen3'`，因此本项目依赖中保留 `transformers>=4.51.0`。

更完整的环境说明见 [docs/environment.md](docs/environment.md)。

第一阶段 LiveBench baseline 评测流程见 [docs/livebench-baseline.md](docs/livebench-baseline.md)。

## 先做 Smoke Test

正式训练前先跑小规模检查：

| 步骤 | 目的 |
| --- | --- |
| 100 条样本 | 验证数据转换和 chat template |
| 20-50 个训练 step | 验证 LoRA target modules 和显存占用 |
| 一次短生成 | 验证 adapter 可以加载并正常回答 |
| 一个很小的评测子集 | 验证进入 LiveBench 前的链路正常 |

只有 smoke test 通过后，再进入 20K 样本训练。

## 训练默认值

这些是第一版起点，不是最终结论：

```yaml
model_name: Qwen/Qwen3-4B-Instruct-2507
dataset_name: open-thoughts/OpenThoughts3-1.2M
max_seq_length: 4096
sample_size: 20000
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
learning_rate: 1.0e-5
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
```

在 RTX 3090 上，`4096` 是保守的第一版上下文长度。`8192` 应作为后续显存实验。

## 评测设计

本地跑三个条件：

| 条件 | 模型 |
| --- | --- |
| Baseline | `Qwen/Qwen3-4B-Instruct-2507` |
| Fine-tuned | `Qwen/Qwen3-4B-Instruct-2507` + AlignReason LoRA |
| 同规模参考 | `Qwen/Qwen3-4B-Thinking-2507` |

最主要的结果是：

```text
fine-tuned instruct score - baseline instruct score
```

Thinking 模型是同规模参考，不是唯一外部目标。

同时把 fine-tuned 模型放到 LiveBench 公榜坐标中比较：

| 比较 | 目的 |
| --- | --- |
| Fine-tuned vs frontier models | 看和当前大闭源模型的绝对差距 |
| Fine-tuned vs large open models | 看 4B LoRA 是否接近更大开源模型 |
| Fine-tuned vs same-size Qwen thinking | 区分模型规模和 thinking mode 的影响 |
| Fine-tuned vs base instruct | 衡量本项目训练带来的真实增益 |

公榜比较必须记录 LiveBench snapshot 日期，因为公开排名会随时间变化。

## 成功标准

实验应该能用可复现产物回答这些问题：

- LoRA adapter 是否提升了 instruct baseline 的 LiveBench 总分？
- 哪些类别提升，哪些类别下降？
- Adapter 是否缩小了和 `Qwen3-4B-Thinking-2507` 的差距？
- Adapter 在 LiveBench 公榜大模型结果中处于什么位置？
- 显式 reasoning SFT 是否带来输出变长、格式退化或指令跟随下降？

## 参考

- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen3-4B-Thinking-2507: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
- OpenThoughts3-1.2M: https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M
- LiveBench leaderboard: https://livebench.ai/
- LiveBench: https://github.com/LiveBench/LiveBench
