[![☕ Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](https://github.com/AEON-7/AEON-7#-support-the-work)

---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Jiunsong/supergemma4-26b-abliterated-multimodal
tags:
  # Model family
  - gemma4
  - gemma
  - google
  # Architecture
  - moe
  - mixture-of-experts
  - transformer
  - 26b
  # Quantization
  - nvfp4
  - fp4
  - 4-bit
  - quantized
  - modelopt
  - weight-quantization
  # Abliteration / Uncensored
  - uncensored
  - abliterated
  - unfiltered
  - refusal-removed
  # Capabilities
  - vision
  - multimodal
  - text-generation
  - image-text-to-text
  - tool-calling
  - function-calling
  - reasoning
  - thinking
  - chat
  - instruct
  - agentic
  - coding
  - creative-writing
  # Hardware
  - dgx-spark
  - blackwell
  - gb10
  - grace-blackwell
  - nvidia
  - gpu
  # Framework / serving
  - vllm
  - openai-api
  - openai-compatible
  # Performance
  - fp8-kv-cache
  - prefix-caching
  - chunked-prefill
  - sliding-window-attention
  # Other
  - english
  - safetensors
  - production-ready
model_type: gemma4
quantization: nvfp4
language:
  - en
---

# SuperGemma4-26B-Abliterated-Multimodal — NVFP4

NVFP4-quantized version of [Jiunsong/supergemma4-26b-abliterated-multimodal](https://huggingface.co/Jiunsong/supergemma4-26b-abliterated-multimodal) — an abliterated (uncensored) Gemma 4 26B Mixture-of-Experts multimodal model with thinking/reasoning capabilities.

Quantized using **NVIDIA ModelOpt 0.43 (main)** with `NVFP4_DEFAULT_CFG` on a native Blackwell GPU. Vision encoder preserved in full BF16. Calibration done in 12 minutes using [`modelopt-fast-moe`](https://github.com/AEON-7/modelopt-fast-moe) adaptive batching — vs ~50h projected for naive bs=1 on this MoE.

Optimized for deployment on **NVIDIA DGX Spark** (GB10, SM 12.0) and other Blackwell-architecture GPUs. **Verified end-to-end**: calibrated → exported → served on Spark → benchmarked.

**[GitHub Repo](https://github.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4)** — deployment configs, patches, quantization script, Dockerfile

**[Pre-Built Container on GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq)** — ready-to-run vLLM image for DGX Spark

## Benchmarks (DGX Spark GB10, vLLM 0.19.1rc1.dev110)

Measured against this NVFP4 checkpoint, FP8 E4M3 KV cache, `--max-num-seqs 4`, `--max-model-len 65536`. Single NVIDIA GB10 Blackwell iGPU (102 GB unified).

| Configuration | Value |
|---|---|
| **Single-stream decode** | **52.6 tok/s** |
| **Single-stream TTFT** (greedy, short prompt) | **54 ms** |
| Aggregate @ 4 parallel streams | 143.3 tok/s |
| Per-stream @ 4 parallel | 35.8 tok/s |
| TTFT @ 4 parallel | 514 ms |
| 16K-input prefill | 744 tok/s |
| 16K-input TTFT | 21.5 s |

MoE backend: **MARLIN** NVFP4 (FlashInfer variants aren't compatible with the 704-per-expert intermediate dim). NVFP4 GEMM backend: FLASHINFER_CUTLASS. Attention backend: TRITON_ATTN (Gemma 4 has heterogeneous head dimensions).

## Key Specs

| | Original (BF16) | NVFP4 (this model) |
|---|---|---|
| **Size on disk** | ~49 GB | ~16.4 GB |
| **Total parameters** | 25.2B | 25.2B |
| **Active parameters** | 3.8B / token | 3.8B / token |
| **Architecture** | MoE: 128 experts, 8 active / token | same |
| **Context window** | 262K tokens | 262K tokens |
| **Modalities** | Text, Image, Video | Text, Image, Video |
| **Quantization** | — | NVFP4 (W4A4, block size 16) |
| **Vision encoder** | BF16 | BF16 (preserved, not quantized) |

## Model Details

| Property | Value |
|---|---|
| **Architecture** | Gemma 4 MoE (26B total, 3.8B active / token) |
| **Layers** | 30 (25 sliding-window + 5 full-attention) |
| **Experts** | 128 total, top-8 active per token |
| **Sliding Window** | 1024 tokens |
| **Max Context** | 262,144 tokens |
| **Hidden Size** | 2816 |
| **MoE Intermediate** | 704 per expert |
| **Attention Heads** | 16 (8 KV heads), head_dim=256, global_head_dim=512 |
| **Vision Encoder** | 27-layer ViT (1152 hidden, 16 heads, patch_size=16) |
| **Vocabulary** | 262,144 tokens |
| **Quantization** | NVFP4 (ModelOpt 0.43 main + 2 pending PRs) |

## Pre-Built Container Image

A pre-built vLLM container compiled for NVIDIA DGX Spark (GB10, SM 12.1) is available with all required patches pre-applied:

```bash
docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
```

**Image contents:**
- vLLM 0.19.1rc1 compiled for SM 12.1 (Blackwell GB10)
- PyTorch 2.12.0 + CUDA 13.0
- transformers 5.5.0 + FlashInfer 0.6.7
- Patched `gemma4.py` — extends expert_params_mapping to the modelopt suffix set (`weight`, `weight_scale`, `weight_scale_2`, `input_scale`)
- Patched `serving.py` — fixes non-streaming reasoning parser for Gemma 4
- Built from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with `--tf5` flag

> [Container on GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq)

## Quick Start

### 1. Pull the container

```bash
docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
```

### 2. Download the model

```bash
pip install -U huggingface-hub hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4 \
  --local-dir ~/models/supergemma4-26b
```

### 3. Get the patched files

Only **one** patch file is required for this v6 checkpoint (down from three in the original v3 release — the plain-NVFP4 config simplifies the loading path). Download from the [GitHub repo](https://github.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4):

```bash
for f in gemma4_patched.py serving_chat_patched.py; do
  curl -LO https://raw.githubusercontent.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4/main/$f
done
```

### 4. Launch with Docker Compose

Save the following as `docker-compose.yml`:

```yaml
services:
  vllm:
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
    container_name: vllm-supergemma4-26b
    restart: unless-stopped
    network_mode: host
    volumes:
      - ~/models/supergemma4-26b:/models/supergemma4
      - ./gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py
      - ./serving_chat_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
    environment:
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - NVIDIA_FORWARD_COMPAT=1
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/supergemma4 \
          --served-model-name supergemma4-26b \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype fp8_e4m3 \
          --tensor-parallel-size 1 \
          --max-model-len 65536 \
          --max-num-seqs 4 \
          --gpu-memory-utilization 0.90 \
          --trust-remote-code \
          --host 0.0.0.0 --port 8000 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser gemma4 \
          --reasoning-parser gemma4
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then start:

```bash
docker compose up -d
```

Startup takes ~4-5 minutes (weight loading + torch.compile + CUDA graph capture + FP4 GEMM autotuning).

### 5. Test

```bash
# Text
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "supergemma4-26b",
    "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
    "max_tokens": 300
  }'
```

```bash
# With reasoning
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "supergemma4-26b",
    "messages": [{"role": "user", "content": "What is the derivative of x^3 * sin(x)?"}],
    "max_tokens": 500,
    "include_reasoning": true
  }'
```

```bash
# Vision — image understanding
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "supergemma4-26b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},
        {"type": "text", "text": "Describe what you see."}
      ]
    }],
    "max_tokens": 300
  }'
```

The API is fully **OpenAI-compatible** — use it with any OpenAI SDK, LangChain, LiteLLM, Open WebUI, or other client at `http://<your-ip>:8000/v1`.

## Key Deployment Flags

| Flag | Purpose |
|---|---|
| `--quantization modelopt` | **Required** — tells vLLM to use NVIDIA ModelOpt NVFP4 format |
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache — doubles available token budget vs BF16 |
| `--max-model-len 65536` | 64K context for DGX Spark. Model supports 262K; increase with fewer concurrent sequences |
| `--max-num-seqs 4` | Max concurrent sequences — balance with context length for KV budget |
| `--gpu-memory-utilization 0.90` | MoE parameter footprint leaves 10% headroom |
| `--reasoning-parser gemma4` | Extracts `<think>` blocks into `reasoning_content` in API response |
| `--tool-call-parser gemma4` | Enables native Gemma 4 function / tool calling |
| `--enable-chunked-prefill` | Processes long prompts in chunks to avoid OOM |
| `--enable-prefix-caching` | Caches common prompt prefixes for faster responses |

## Quantization Details

| Parameter | Value |
|---|---|
| **Tool** | NVIDIA ModelOpt 0.43.0rc2.dev (from upstream main) |
| **Config** | `NVFP4_DEFAULT_CFG` (plain NVFP4, no AWQ) |
| **Weight dtype** | NVFP4 (FP4 E2M1, block size 16) |
| **Calibration samples** | 512 (CNN/DailyMail train split) |
| **Calibration seq_len** | 4096 tokens |
| **Batch size** | 3 (VRAM-probed) |
| **Calibration hardware** | NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM) |
| **Calibration wall-clock** | **12.75 min** |
| **Excluded from quantization** | `vision_tower`, `embed_vision`, `multi_modal_projector`, routers (BF16) |
| **Exported size** | 16.42 GB |

### Why plain NVFP4 instead of NVFP4_AWQ?

Earlier experiments used `NVFP4_AWQ_FULL_CFG` (AWQ with exhaustive alpha grid search) but ran into a deployment-stack limitation: **vLLM's `ModelOptNvFp4FusedMoE` does not support per-expert `pre_quant_scale`**. On MoE models, AWQ calibration computes a per-expert scaling factor that can't be consumed by the MoE kernel path — any AWQ work on experts is wasted at serve time.

Switching to plain NVFP4 (`algorithm=max`):
- Cuts calibration time from ~2.5h to ~12 min (no alpha search phase)
- Produces a checkpoint vLLM's FusedMoE loads natively without any tensor surgery
- Quality hit is negligible since the AWQ benefit on MoE experts was already unavailable at inference time (vLLM doesn't apply expert-level pre_quant_scale even if present)

Attention and dense shared MLP layers still benefit from NVFP4's per-block scaling. Router weights stay in BF16 (routing quality is critical for MoE accuracy and experts are cheap to leave un-quantized there).

### Applied modelopt patches

Two upstream PR fixes applied locally (pending review as of this writing):
- **[PR #1264](https://github.com/NVIDIA/TensorRT-Model-Optimizer/pull/1264)** — `preprocess_linear_fusion` non-scalar amax fix (NVFP4 per-channel activation quantizer in MoE path)
- **[PR #1265](https://github.com/NVIDIA/TensorRT-Model-Optimizer/pull/1265)** — `get_activation_scaling_factor` zero-amax handling (MoE routing sparsity leaves some expert channels un-activated during calibration)

Both are also blockers for anyone quantizing per-expert-decomposed MoEs in NVFP4 with modelopt 0.42 or 0.43.

### fast-moe adaptive batched calibration

Calibration uses [`modelopt-fast-moe`](https://github.com/AEON-7/modelopt-fast-moe) — a drop-in replacement for modelopt's naive `for ids in calib_data: model(ids)` forward loop. On MoE models, the naive loop leaves the GPU at 25-30% utilization because Python dispatch overhead dominates across hundreds of tiny per-expert GEMMs. Adaptive VRAM-probed batching fixes that.

End-to-end calibration wall-clock on this model:

| Configuration | Wall-clock |
|---|---|
| Naive bs=1 loop (modelopt default) | ~50h projected (killed at 18h) |
| fast-moe + NVFP4_AWQ_FULL (earlier v3 attempt) | 2h 24min |
| **fast-moe + NVFP4_DEFAULT (this v6)** | **12 min** |

### NVFP4 Weight Format

Each quantized layer stores:
- `weight` (uint8) — packed FP4 E2M1 pairs (16-element blocks)
- `weight_scale` (float8_e4m3fn) — per-block scale (1 per 16 elements)
- `weight_scale_2` (float32) — per-tensor global scale
- `input_scale` (float32) — static activation scale from calibration

## Quality Validation

Greedy-sampled responses from this checkpoint (server defaults, `temperature=0.0`):

| Prompt | Response |
|---|---|
| *"What is the capital of France?"* | "The capital of France is Paris." |
| *"What is 17 * 23?"* | "17 * 23 = 391" ✓ |
| *"Write a haiku about autumn."* | "Gold leaves drift to ground, / Crisp air chills the morning sun, / Winter's breath draws near." |
| *"Name three cities in Japan."* | "1. Tokyo  2. Osaka  3. Kyoto" |
| *"One-line Python Fibonacci function"* | Valid memoized implementation with explanation |
| *"Explain photosynthesis in one sentence."* | "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and glucose (sugar) for food." |

## Speculative Decoding (DFlash — Coming Soon)

A [DFlash block-diffusion drafter](https://arxiv.org/abs/2602.06036) paired with this model is in training. DFlash can provide 2-3× additional throughput over the numbers above by predicting multi-token blocks in a single draft forward pass. Will be published as a separate drafter repo once training completes.

## Dense (31B) vs MoE (26B) Comparison

| Metric | [31B DECKARD Dense](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) | This Model (26B MoE) |
|---|---|---|
| Active params / token | **31.3B** | ~3.8B |
| NVFP4 model size | 20.5 GB | 16.4 GB |
| Single-stream tok/s (Spark) | ~11 | **52.6** |
| Context window | 262K | 262K |
| Vision | Yes | Yes |
| Best for | Quality-critical tasks | Speed, concurrency, efficiency |

## Hardware Requirements

| Tier | GPU | Notes |
|---|---|---|
| **Target** | NVIDIA DGX Spark (128 GB unified) | Full 262K context, up to 6 concurrent sequences |
| **Compatible** | RTX 5090 (32 GB) | Reduced context, 1-2 sequences |
| **Compatible** | B200 / GB200 | Full context, high concurrency |
| **Compatible** | RTX PRO 6000 Blackwell (97 GB) | Calibration + serving (used for quantization here) |
| **Minimum** | Any Blackwell GPU (SM 10.0+) | Required for native FP4 |

> Native FP4 hardware (Blackwell architecture) is required. This model will not run on Ampere or Ada GPUs.

## Related Projects

### Models

| Model | Type | Size | Link |
|---|---|---|---|
| **SuperGemma4 26B NVFP4** (this) | MoE NVFP4 | 16.4 GB | [GitHub](https://github.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4) |
| **Gemma 4 31B DECKARD** | Dense NVFP4 AWQ | 20.5 GB | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) |
| **DECKARD E4B Drafter** | EAGLE NVFP4 | 9.6 GB | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4) |

### Infrastructure

| Resource | Description | Link |
|---|---|---|
| **vLLM AWQ Container** | Pre-built for DGX Spark (SM 12.1) with all patches | [GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq) |
| **Build System** | spark-vllm-docker | [GitHub](https://github.com/eugr/spark-vllm-docker) |
| **modelopt-fast-moe** | Adaptive batched calibration (used for this model) | [GitHub](https://github.com/AEON-7/modelopt-fast-moe) |
| **Base Model** | SuperGemma4 26B Abliterated Multimodal (BF16) | [HuggingFace](https://huggingface.co/Jiunsong/supergemma4-26b-abliterated-multimodal) |

## Disclaimer

**THIS IS AN UNCENSORED MODEL.** By downloading, accessing, or using this model, you expressly acknowledge that you assume full and sole responsibility for all outputs generated, all actions taken based on outputs, and compliance with applicable laws. The authors are not responsible for any harmful, illegal, or objectionable content produced by the model. These tools serve legitimate purposes including security research, red-teaming, content analysis, and creative work. Implement safeguards appropriate to your use case and jurisdiction.

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from Google.

## Credits

Quantized by [AEON-7](https://github.com/AEON-7) on NVIDIA Blackwell hardware. Built and validated with AI-engineering assistance from Anthropic.

Shout-out to [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) for the DGX Spark-optimized vLLM build, NVIDIA for [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer), and the z-lab / ModelOpt teams for [DFlash](https://arxiv.org/abs/2602.06036).

---

## ☕ Support the work

If this release has been useful, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>₿ Bitcoin (BTC)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="BTC QR" width="200"/><br/>
      <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Ξ Ethereum (ETH)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="ETH QR" width="200"/><br/>
      <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>◎ Solana (SOL)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="SOL QR" width="200"/><br/>
      <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>ⓜ Monero (XMR)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="XMR QR" width="200"/><br/>
      <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
    </td>
  </tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.
