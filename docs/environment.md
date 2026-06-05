# 环境说明

本文档记录 AlignReason 的基础环境准备。LiveBench baseline 的完整执行流程见 [livebench-baseline.md](livebench-baseline.md)。

## 目标硬件

第一目标环境是单卡 RTX 3090，24GB 显存。

建议从以下配置开始：

- sequence length: 4096
- per-device batch size: 1
- gradient accumulation: 8
- gradient checkpointing: enabled
- fp16: enabled

4096 smoke test 通过后，再尝试 8192 上下文。

## Python 环境

推荐使用 `uv` 创建环境：

```bash
cd /path/to/alignreason
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

本项目使用 `uv` 是因为部分精简 Debian/Ubuntu Python 环境没有内置 `ensurepip`，直接运行 `python3 -m venv` 可能因为缺少 `python3-venv` 而失败。

如果 `uv` 默认缓存目录不可写，可以指定临时缓存：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

## Hugging Face 登录

如果模型或数据下载遇到访问限制、限速或私有缓存问题，先登录：

```bash
huggingface-cli login
```

也可以使用新版命令：

```bash
hf auth login
```

## CUDA 检查

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

## 最小模型加载检查

在正式训练或评测前，先确认 tokenizer、chat template 和模型加载正常：

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

## vLLM 启动检查

Base instruct 模型：

```bash
vllm serve "Qwen/Qwen3-4B-Instruct-2507"
```

Thinking 参考模型：

```bash
vllm serve "Qwen/Qwen3-4B-Thinking-2507"
```

比较不同模型时，需要保持生成参数一致。
