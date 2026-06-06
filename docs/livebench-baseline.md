# LiveBench Baseline 流程

本文档记录 `Qwen/Qwen3-4B-Instruct-2507` 的原始模型 baseline 评测流程：用 vLLM 启动本地 OpenAI-compatible API，再用 LiveBench 官方评测脚本获取分数。

## 目标

第一阶段只评测原始 instruct 模型：

```text
Qwen/Qwen3-4B-Instruct-2507
  -> vLLM OpenAI-compatible API
  -> LiveBench official harness
  -> baseline score
```

这个分数作为后续 AlignReason LoRA 微调结果的对照基线。

本阶段使用的是 LiveBench 官方脚本：

- `run_livebench.py`：生成模型回答并执行评测；
- `show_livebench_result.py`：汇总并展示结果。

AlignReason 仓库当前不封装自己的 baseline 评测脚本，避免和官方评测口径产生偏差。

## 路径约定

先选择一个模型权重目录：

```bash
export MODEL_DIR=/path/to/your/models/Qwen3-4B-Instruct-2507
```

建议放在数据盘或其他大容量持久化目录中。模型权重大约 8GB。

## 环境检查

```bash
nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 安装运行依赖

Qwen 官方模型卡要求：

```text
transformers >= 4.51.0
vllm >= 0.8.5
```

安装：

```bash
pip install -U "vllm>=0.8.5" "transformers>=4.51.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

安装后确认版本：

```bash
python - <<'PY'
import torch, transformers, vllm

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY
```

## 下载模型

优先使用 Hugging Face 官方源。先确保当前 shell 没有设置镜像：

```bash
unset HF_ENDPOINT
mkdir -p "$MODEL_DIR"

hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir "$MODEL_DIR"
```

如果匿名下载较慢，可以登录 Hugging Face：

```bash
hf auth login
```

如果服务器提供网络加速，可以在下载前启用。例如 AutoDL：

```bash
source /etc/network_turbo
```

## 检查模型文件

确认关键文件都存在：

```bash
cd "$MODEL_DIR"

for f in \
  config.json \
  generation_config.json \
  model.safetensors.index.json \
  model-00001-of-00003.safetensors \
  model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  merges.txt
do
  test -f "$f" && echo "OK  $f" || echo "MISS $f"
done

du -sh "$MODEL_DIR"
```

模型应包含 3 个 safetensors shard，总大小约 8GB。

也可以从 index 文件反查需要的权重分片：

```bash
python - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
idx = json.loads((d / "model.safetensors.index.json").read_text())
for f in sorted(set(idx["weight_map"].values())):
    p = d / f
    print(f, "OK" if p.exists() else "MISS")
PY
```

## 启动 vLLM

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64
```

`--served-model-name` 必须和 LiveBench 里传入的 `--model` 保持一致。

如果希望先用更保守的显存配置：

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64
```

`Qwen3-4B-Instruct-2507` 是 non-thinking instruct 模型，baseline 阶段不需要 vLLM 的 reasoning parser 相关参数。

## API Smoke Test

另开一个 shell：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Solve: 2x + 3 = 9"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

本地 API 能正常返回答案后，再进入 LiveBench 评测。

## 安装 LiveBench

```bash
cd /path/to/your/workspace
git clone https://github.com/LiveBench/LiveBench.git
cd LiveBench
pip install -e .
```

如果要跑 coding 任务，还需要安装额外依赖：

```bash
cd livebench/code_runner
pip install -r requirements_eval.txt
cd ../..
```

## 先跑 Math 子集

先跑一个子集，确认官方评测链路正常：

```bash
cd /path/to/your/workspace/LiveBench

python run_livebench.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-display-name qwen3-4b-instruct-2507-baseline \
  --bench-name live_bench/math \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --parallel-requests 8 \
  --max-tokens 4096 \
  --force-temperature 0 \
  --livebench-release-option 2026-01-08 \
  --resume
```

## 跑全量 Baseline

Math 子集跑通后，再跑全量：

```bash
python run_livebench.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-display-name qwen3-4b-instruct-2507-baseline \
  --bench-name live_bench \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --parallel-requests 8 \
  --max-tokens 4096 \
  --force-temperature 0 \
  --livebench-release-option 2026-01-08 \
  --resume
```

如果本地 vLLM 服务在并发请求下不稳定，可以把并发降到：

```bash
--parallel-requests 1
```

## 查看结果

Math 子集：

```bash
python show_livebench_result.py \
  --bench-name live_bench/math \
  --model-list qwen3-4b-instruct-2507-baseline \
  --livebench-release-option 2024-11-25
```

全量结果：

```bash
python show_livebench_result.py \
  --bench-name live_bench \
  --model-list qwen3-4b-instruct-2507-baseline \
  --livebench-release-option 2024-11-25
```

保留 LiveBench 生成的结果文件，后续微调后使用同一套设置复测。

## Baseline 分数

2026-01-08 release 下各 task 和 category 的详细得分：

| Task | 分数 |
|------|:----:|
| AMPS_Hard | 67.0 |
| math_comp | 45.7 |
| olympiad | 10.8 |
| spatial | 50.0 |
| zebra_puzzle | 52.8 |
| connections | 28.7 |
| tablereformat | 76.0 |
| cta | 46.0 |
| tablejoin | 33.5 |
| simplify | 79.1 |

| Category | 分数 |
|----------|:----:|
| instruction_following | **79.1** |
| data_analysis | **51.8** |
| reasoning | **51.4** |
| math | **41.1** |
| language | **28.7** |
| **Average** | **50.4** |

## 清理

安装完成后可以清理 pip 下载缓存：

```bash
rm -rf /root/.cache/pip
```

不要手动删除已经安装的 Python 库目录，例如：

```text
/root/miniconda3/lib
```

## 参考

- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- LiveBench official repo: https://github.com/LiveBench/LiveBench
- LiveBench leaderboard: https://livebench.ai/
