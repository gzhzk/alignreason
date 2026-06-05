# AlignReason

AlignReason is a small post-training experiment for testing whether explicit reasoning SFT data can improve a strong 4B instruct model.

## Experiment Question

Can high-quality reasoning traces from OpenThoughts improve `Qwen/Qwen3-4B-Instruct-2507` on LiveBench, especially on reasoning, math, code, and science tasks?

The main fine-tuning target is:

- `Qwen/Qwen3-4B-Instruct-2507`

The comparison baselines are:

- `Qwen/Qwen3-4B-Instruct-2507` before fine-tuning
- `Qwen/Qwen3-4B-Thinking-2507` as a stronger thinking-model reference

This keeps the experiment clean: the LoRA adapter is trained only on the non-thinking instruct model, then compared against both its original baseline and Qwen's dedicated thinking variant.

## Initial Plan

| Area | Choice |
| --- | --- |
| Base model for SFT | `Qwen/Qwen3-4B-Instruct-2507` |
| Reference model | `Qwen/Qwen3-4B-Thinking-2507` |
| Dataset | `open-thoughts/OpenThoughts3-1.2M` |
| Sample size | 20,000 examples after smoke tests |
| Data style | Chat messages with assistant reasoning traces |
| Fine-tuning method | LoRA SFT |
| Hardware target | Single RTX 3090 24GB |
| First sequence length | 4096 |
| Stretch sequence length | 8192 if memory allows |
| Evaluation | LiveBench before and after SFT |

## Why Instruct for Fine-Tuning?

`Qwen3-4B-Instruct-2507` is the main target because it is a strong 4B non-thinking instruct model. Fine-tuning it with explicit reasoning data tests a clear hypothesis:

> Can reasoning traces teach a normal instruct model to reason better?

`Qwen3-4B-Thinking-2507` is useful as a comparison point, but it is not the first fine-tuning target. It already defaults to long-form reasoning, so gains from OpenThoughts would be harder to interpret.

## Repository Layout

```text
alignreason/
  README.md
  requirements.txt
  configs/
    experiment.yaml
  docs/
    environment.md
  scripts/
```

The first implementation milestone is to add scripts for:

1. sampling OpenThoughts into JSONL,
2. running a 100-example LoRA smoke train,
3. training the 20K-example adapter,
4. serving base and fine-tuned models for LiveBench.

## Environment

Create a Python environment:

```bash
cd /home/kai/workspace/alignreason
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Qwen3 requires a recent Transformers version. The model card warns that `transformers<4.51.0` can fail with `KeyError: 'qwen3'`, so this project keeps the dependency at or above that version.

## Smoke Test First

Before the full run, use a small sample:

| Step | Purpose |
| --- | --- |
| 100 examples | Validate dataset conversion and chat template |
| 20-50 training steps | Validate LoRA target modules and GPU memory |
| One short generation | Validate that the adapter loads and responds |
| Tiny eval subset | Check that the pipeline works before LiveBench |

Only after the smoke test passes should the project move to the 20K-example training run.

## Expected Training Defaults

These are starting points, not final claims:

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

On an RTX 3090, `4096` context is the conservative first target. `8192` should be treated as a follow-up memory experiment.

## Evaluation Design

Run LiveBench in three conditions:

| Condition | Model |
| --- | --- |
| Baseline | `Qwen/Qwen3-4B-Instruct-2507` |
| Fine-tuned | `Qwen/Qwen3-4B-Instruct-2507` + AlignReason LoRA |
| Reference | `Qwen/Qwen3-4B-Thinking-2507` |

The main reported result should be:

```text
fine-tuned instruct score - baseline instruct score
```

The thinking model score is a reference point, not the primary success criterion.

## Success Criteria

The experiment is useful if it can answer these questions with reproducible artifacts:

- Does the LoRA adapter improve total LiveBench score over the instruct baseline?
- Which categories improve or regress?
- Does the adapter close any part of the gap to `Qwen3-4B-Thinking-2507`?
- Does explicit reasoning SFT cause longer outputs, formatting issues, or degraded instruction following?

## References

- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen3-4B-Thinking-2507: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
- OpenThoughts3-1.2M: https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M
- LiveBench: https://github.com/LiveBench/LiveBench
