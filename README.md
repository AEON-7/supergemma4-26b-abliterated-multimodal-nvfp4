# SuperGemma4-26B-Abliterated-Multimodal-NVFP4

NVFP4-quantized version of [Jiunsong/supergemma4-26b-abliterated-multimodal](https://huggingface.co/Jiunsong/supergemma4-26b-abliterated-multimodal) — an abliterated, uncensored Gemma 4 26B MoE multimodal model. Quantized using NVIDIA ModelOpt 0.42.0 with **AWQ Full** (activation-aware weight quantization with exhaustive grid search) for maximum fidelity at 4-bit precision.

Optimized for deployment on **NVIDIA DGX Spark** (GB10, SM 12.1) and other Blackwell-architecture GPUs.

**[Model on HuggingFace](https://huggingface.co/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4)** | **[Pre-built Container on GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq)**

## Key Specs

| | Original (BF16) | NVFP4 AWQ Full (this) |
|---|---|---|
| **Size on disk** | ~49 GB | ~13 GB + ~1.2 GB vision |
| **Total parameters** | 25.2B | 25.2B |
| **Active parameters** | 3.8B/token | 3.8B/token |
| **Architecture** | MoE: 128 experts, 8 active/token | same |
| **Context window** | 262K tokens | 262K tokens |
| **Modalities** | Text, Image, Video | Text, Image, Video |
| **Quantization** | — | NVFP4 AWQ Full (W4A4, block size 16) |
| **Vision encoder** | BF16 | BF16 (preserved, not quantized) |

## Pre-Built Container Image

A pre-built vLLM container compiled for NVIDIA DGX Spark (GB10, SM 12.1) is available with all required patches:

```bash
docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
```

**Image contents:**
- vLLM 0.19.1rc1 compiled for SM 12.1 (Blackwell GB10)
- PyTorch 2.12.0 + CUDA 13.0
- transformers 5.5.0 + FlashInfer 0.6.7
- **Patched `modelopt.py`** — fixes FP8 NaN in weight scales, adds NVFP4_AWQ support, AWQ pre_quant_scale handling
- **Patched `serving.py`** — fixes non-streaming reasoning parser (re-decodes with special tokens when delimiters are stripped)
- Built from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with `--tf5` flag

> [AWQ Container on GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq) | For AWQ quantized models

## Quick Start

### Docker Compose (DGX Spark)

```yaml
services:
  vllm:
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
    container_name: vllm-supergemma4-26b
    restart: unless-stopped
    network_mode: host
    volumes:
      - /path/to/model:/models/supergemma4
      - ./modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
      - ./serving_chat_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
    environment:
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      - VLLM_USE_FLASHINFER_MOE_FP4=1
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/supergemma4 \
          --served-model-name supergemma4-26b \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype fp8_e4m3 \
          --calculate-kv-scales \
          --tensor-parallel-size 1 \
          --max-model-len 65536 \
          --max-num-seqs 4 \
          --gpu-memory-utilization 0.90 \
          --trust-remote-code \
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

### Direct vLLM Serve

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1

vllm serve AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4 \
    --quantization modelopt \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --max-model-len 262144 \
    --max-num-seqs 6 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code
```

### Testing

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "supergemma4-26b",
    "messages": [{"role": "user", "content": "Hello! Tell me a joke."}],
    "max_tokens": 300,
    "include_reasoning": true
  }'
```

### Key Deployment Flags

| Flag | Purpose |
|---|---|
| `--quantization modelopt` | **Required** — NVIDIA ModelOpt NVFP4 format |
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache — doubles token budget |
| `--reasoning-parser gemma4` | Extracts thinking/reasoning into separate field |
| `--tool-call-parser gemma4` | Enables native Gemma 4 function calling |
| `--enable-chunked-prefill` | Processes long prompts in chunks to avoid OOM |
| `--enable-prefix-caching` | Caches common prompt prefixes |
| `VLLM_USE_FLASHINFER_MOE_FP4=1` | FlashInfer MoE FP4 kernels |

## Critical Fix: FP8 NaN in Weight Scales

The NVFP4 checkpoint produced by ModelOpt 0.42.0 contains scattered FP8 NaN values (0x7F / 0xFF) in `weight_scale` tensors. The included [`modelopt_patched.py`](modelopt_patched.py) automatically scrubs these at load time.

If you're using the pre-built container, this is already applied.

## Non-Streaming Reasoning Parser Fix

vLLM's default `skip_special_tokens=True` strips `<|channel>` / `<channel|>` delimiters from decoded text, causing the non-streaming reasoning parser to fail (reasoning lands in `content` with `reasoning: None`). The included [`serving_chat_patched.py`](serving_chat_patched.py) fixes this by re-decoding token IDs with special tokens preserved when text-based extraction fails.

**Streaming** works correctly out of the box (uses token IDs). This fix is only needed for non-streaming completions.

## Quantization Details

| Parameter | Value |
|---|---|
| **Tool** | NVIDIA ModelOpt 0.42.0 |
| **Method** | NVFP4 AWQ Full (`NVFP4_AWQ_FULL_CFG`) |
| **Weight dtype** | NVFP4 (FP4 E2M1, block size 16) |
| **Calibration** | 4096 samples x 4096 tokens from CNN/DailyMail |
| **Batch size** | 16 |
| **Excluded** | `vision_tower`, `embed_vision` (kept in BF16) |
| **Expert routing** | Natural (router decides, not forced uniform) |

### Why AWQ Full?

AWQ Full performs exhaustive grid search with `alpha_step=0.1` across scaling factors per layer, plus clipping ratio optimization. For MoE models with 128 experts where each expert only sees ~6% of calibration tokens, this thorough search matters — it finds mathematically optimal per-channel scaling that basic RTN quantization misses.

### NVFP4 Weight Format

Each quantized layer stores:
- `weight` (uint8) — packed FP4 E2M1 pairs (16-element blocks)
- `weight_scale` (float8_e4m3fn) — per-block scale (1 per 16 elements)
- `weight_scale_2` (float32) — per-tensor global scale
- `pre_quant_scale` (bfloat16) — AWQ per-channel pre-scaling factors
- `input_scale` (float32) — static activation scale from calibration

## Included Files

| File | Purpose |
|---|---|
| [`modelopt_patched.py`](modelopt_patched.py) | vLLM ModelOpt patch — FP8 NaN fix + NVFP4_AWQ support |
| [`serving_chat_patched.py`](serving_chat_patched.py) | vLLM serving patch — non-streaming reasoning parser fix |
| [`Dockerfile`](Dockerfile) | Container build with both patches applied |
| [`docker-compose.yml`](docker-compose.yml) | Production deployment configuration |
| [`quantize_gemma4_moe.py`](quantize_gemma4_moe.py) | Quantization script with MoE expert plugin |

## Manual Patching

If you're using your own vLLM installation:

```bash
VLLM_DIR=$(python3 -c "import vllm; print(vllm.__path__[0])")

# ModelOpt NVFP4_AWQ patch
curl -L -o "$VLLM_DIR/model_executor/layers/quantization/modelopt.py" \
  https://raw.githubusercontent.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4/main/modelopt_patched.py

# Non-streaming reasoning parser fix
curl -L -o "$VLLM_DIR/entrypoints/openai/chat_completion/serving.py" \
  https://raw.githubusercontent.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4/main/serving_chat_patched.py
```

## Hardware Requirements

- **Inference**: Blackwell GPU required (SM 10.0+) for native FP4
- **Target**: NVIDIA DGX Spark (128 GB unified memory)
- **Compatible**: RTX Pro 6000, B200, GB200, RTX 5090

## Related Projects

| Resource | Description | Link |
|---|---|---|
| **SuperGemma4 26B NVFP4** | This model on HuggingFace | [HuggingFace](https://huggingface.co/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4) |
| **Gemma 4 31B DECKARD NVFP4** | Dense 31B uncensored + thinking | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Gemma 4 26B MoE Uncensored NVFP4** | Earlier 26B MoE quantization | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
| **vLLM AWQ Container** | Pre-built for DGX Spark | [GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq) |
| **Base Model** | SuperGemma4 26B Abliterated | [HuggingFace](https://huggingface.co/Jiunsong/supergemma4-26b-abliterated-multimodal) |

## Disclaimer

**THIS IS AN UNCENSORED MODEL.** By downloading, accessing, or using this model, you assume full responsibility for all outputs generated and compliance with applicable laws. The authors are not responsible for any harmful or objectionable content produced. Implement your own safeguards appropriate to your use case and jurisdiction.

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from Google.

## Credits

Quantized by [AEON-7](https://github.com/AEON-7) on NVIDIA Blackwell hardware. Built and validated with AI-engineering assistance from Anthropic.

Shout-out to [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) for the DGX Spark-optimized vLLM build.
