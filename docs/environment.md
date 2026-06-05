# Environment Notes

## Hardware Target

The first target machine is a single RTX 3090 with 24GB VRAM.

Start with:

- sequence length: 4096
- per-device batch size: 1
- gradient accumulation: 8
- gradient checkpointing: enabled
- fp16: enabled

Move to 8192 context only after the 4096 smoke test has passed.

## Python Setup

```bash
cd /home/kai/workspace/alignreason
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

This project uses `uv` for environment setup because some minimal Debian/Ubuntu Python installs do not include `ensurepip`, which can make `python3 -m venv` fail unless `python3-venv` is installed.

If `uv` cannot write to its default cache directory, use a workspace or temporary cache:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

## Hugging Face Login

If model downloads fail because of gated access, rate limits, or private cache settings:

```bash
huggingface-cli login
```

## CUDA Check

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

## Minimal Model Load Check

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Solve: 2x + 3 = 9"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
PY
```

## vLLM Serving Checks

Base instruct model:

```bash
vllm serve "Qwen/Qwen3-4B-Instruct-2507"
```

Thinking reference model:

```bash
vllm serve "Qwen/Qwen3-4B-Thinking-2507"
```

Keep the generation settings consistent when comparing models in LiveBench.
