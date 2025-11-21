# Multi-Container Architecture Guide

This guide explains the new multi-container architecture for Qwen3 on-policy distillation, which solves dependency conflicts and provides clean separation of concerns.

## Architecture Overview

The pipeline is split into **three independent containers**:

```
┌─────────────────────────────────────────────────────────────┐
│                      GPU (128GB)                             │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │  vLLM Student    │  │  SGLang Teacher  │  │  Training │ │
│  │  Container       │  │  Container       │  │  Container│ │
│  │                  │  │                  │  │           │ │
│  │  Qwen3-4B        │  │  Qwen3-32B (FP8) │  │  unsloth  │ │
│  │  OpenAI API      │  │  OpenAI API      │  │  + LoRA   │ │
│  │  Port: 8000      │  │  Port: 30000     │  │           │ │
│  └────────┬─────────┘  └────────┬─────────┘  └─────┬─────┘ │
│           │                     │                   │       │
│           └─────────────────────┴───────────────────┘       │
│                       Docker Network                        │
└─────────────────────────────────────────────────────────────┘
```

### Container Responsibilities

#### 1. vLLM Student Container (`qwen3-vllm-student`)
- **Base Image**: `nvcr.io/nvidia/vllm:25.09-py3` (official NVIDIA vLLM for GB10)
- **Purpose**: Student model trajectory generation
- **Model**: Qwen3-4B-Instruct
- **Memory**: ~25GB (20% of 128GB)
- **API**: OpenAI-compatible API on port 8000
- **Features**:
  - Fast inference with vLLM optimizations
  - Batch generation support
  - Logprob extraction for on-policy learning

#### 2. SGLang Teacher Container (`qwen3-sglang-teacher`)
- **Base Image**: `nvcr.io/nvidia/pytorch:24.09-py3` + SGLang
- **Purpose**: Teacher model scoring with FP8 quantization
- **Model**: Qwen3-32B-Instruct (FP8)
- **Memory**: ~45GB (35% of 128GB)
- **API**: OpenAI-compatible API on port 30000
- **Features**:
  - FP8 quantization (50% memory savings)
  - Per-token logprob computation
  - Efficient batch scoring

#### 3. Training Container (`qwen3-training`)
- **Base Image**: `nvcr.io/nvidia/pytorch:24.09-py3` + unsloth
- **Purpose**: LoRA fine-tuning and pipeline orchestration
- **Model**: Qwen3-4B-Instruct (loaded for training)
- **Memory**: ~40GB (remaining GPU memory + overhead)
- **Communication**: HTTP API calls to vLLM and SGLang
- **Features**:
  - unsloth for memory-efficient LoRA training
  - Experience replay buffer
  - Checkpoint management
  - Metric logging

## Why Multi-Container?

### Problems with Single Container:
- ❌ Dependency conflicts between vLLM, SGLang, and unsloth
- ❌ PyTorch version incompatibilities
- ❌ Complex build process
- ❌ Difficult to debug
- ❌ Cannot use official NVIDIA vLLM container

### Benefits of Multi-Container:
- ✅ **No dependency conflicts** - each service has its own environment
- ✅ **Use official containers** - NVIDIA vLLM container works perfectly
- ✅ **Clean separation** - easier to debug and maintain
- ✅ **Independent scaling** - can adjust resources per service
- ✅ **Service resilience** - one service failure doesn't crash everything
- ✅ **Development flexibility** - modify one service without rebuilding all

## Quick Start

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

Key settings:
- `STUDENT_MODEL`: Student model name (default: Qwen/Qwen3-4B-Instruct)
- `TEACHER_MODEL`: Teacher model name (default: Qwen/Qwen3-32B-Instruct)
- `HF_CACHE_DIR`: HuggingFace cache directory
- `WANDB_API_KEY`: Optional, for experiment tracking

### 2. Build Containers

```bash
# Build all containers
docker-compose build

# Or build individually
docker-compose build vllm-student
docker-compose build sglang-teacher
docker-compose build training
```

### 3. Start Services

```bash
# Start all services in background
docker-compose up -d

# Or start with logs visible
docker-compose up

# Check service health
docker-compose ps
```

Services will start in order:
1. vLLM student (waits for model download + warmup)
2. SGLang teacher (waits for vLLM + own model download)
3. Training (waits for both APIs to be healthy)

### 4. Monitor Startup

```bash
# Watch vLLM logs
docker-compose logs -f vllm-student

# Watch SGLang logs
docker-compose logs -f sglang-teacher

# Watch all logs
docker-compose logs -f
```

Wait for health checks to pass:
- vLLM: `http://localhost:8000/health`
- SGLang: `http://localhost:30000/health`

### 5. Run Training

```bash
# Enter training container
docker-compose exec training bash

# Run pipeline
python -m qwen3_distill.pipeline \
  --config configs/default.yaml \
  --prompts data/sample_prompts.txt \
  --num-epochs 10
```

## API Endpoints

### vLLM Student API (Port 8000)

**Health Check**:
```bash
curl http://localhost:8000/health
```

**List Models**:
```bash
curl http://localhost:8000/v1/models
```

**Generate (OpenAI-compatible)**:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### SGLang Teacher API (Port 30000)

**Health Check**:
```bash
curl http://localhost:30000/health
```

**Score Text (with logprobs)**:
```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-Instruct",
    "prompt": "Machine learning is a subset of AI...",
    "max_tokens": 1,
    "logprobs": 5,
    "echo": true,
    "temperature": 0.0
  }'
```

## Configuration

### Environment Variables

**In docker-compose.yml** (preferred):
```yaml
environment:
  - STUDENT_MODEL=Qwen/Qwen3-7B-Instruct
  - TEACHER_MODEL=Qwen/Qwen3-14B-Instruct
```

**In .env file** (alternative):
```bash
STUDENT_MODEL=Qwen/Qwen3-7B-Instruct
TEACHER_MODEL=Qwen/Qwen3-14B-Instruct
```

**Python code automatically reads**:
- `VLLM_API_URL` - vLLM API endpoint (default: http://vllm-student:8000)
- `SGLANG_API_URL` - SGLang API endpoint (default: http://sglang-teacher:30000)

### Model Switching

To use different models, update `.env`:

```bash
# For 80GB GPU
STUDENT_MODEL=Qwen/Qwen3-7B-Instruct
TEACHER_MODEL=Qwen/Qwen3-14B-Instruct
STUDENT_GPU_MEM=0.25
TEACHER_GPU_MEM=0.35

# Restart services
docker-compose down
docker-compose up -d
```

See `QUICKSTART.md` for pre-configured model pairs.

## Troubleshooting

### Service Won't Start

**Check logs**:
```bash
docker-compose logs vllm-student
docker-compose logs sglang-teacher
```

**Common issues**:
- GPU not visible: Check `nvidia-docker` runtime
- Port conflict: Change ports in `.env`
- Out of memory: Reduce `*_GPU_MEM` values

### Health Check Failing

**vLLM health check fails**:
```bash
# Check if model is downloading
docker-compose exec vllm-student ls -lh /root/.cache/huggingface

# Test API manually
docker-compose exec vllm-student curl localhost:8000/health
```

**SGLang health check fails**:
```bash
# Check memory usage
docker-compose exec sglang-teacher nvidia-smi

# Test API manually
docker-compose exec sglang-teacher curl localhost:30000/health
```

### API Connection Errors

**From training container**:
```bash
# Enter training container
docker-compose exec training bash

# Test vLLM connection
curl http://vllm-student:8000/health

# Test SGLang connection
curl http://sglang-teacher:30000/health

# Check environment
echo $VLLM_API_URL
echo $SGLANG_API_URL
```

**From host machine**:
```bash
# vLLM
curl http://localhost:8000/health

# SGLang
curl http://localhost:30000/health
```

### Out of Memory

**Check GPU usage**:
```bash
nvidia-smi
```

**Reduce memory allocation**:
```bash
# In .env
STUDENT_GPU_MEM=0.15  # Reduce from 0.2
TEACHER_GPU_MEM=0.30  # Reduce from 0.35
```

**Or use smaller models**:
```bash
STUDENT_MODEL=Qwen/Qwen3-1.5B-Instruct
TEACHER_MODEL=Qwen/Qwen3-7B-Instruct
```

## Maintenance

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop vllm-student
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart sglang-teacher
```

### Update Containers

```bash
# Pull latest images
docker-compose pull

# Rebuild
docker-compose build --no-cache

# Restart
docker-compose up -d
```

### Clean Up

```bash
# Remove containers
docker-compose down

# Remove containers + volumes
docker-compose down -v

# Remove images
docker rmi qwen3-distill-vllm qwen3-distill-sglang qwen3-distill-training
```

## Performance Tuning

### Increase Throughput

**vLLM batching**:
```yaml
# In docker-compose.yml
command: >
  python -m vllm.entrypoints.openai.api_server
  --model ${STUDENT_MODEL}
  --max-num-batched-tokens 8192  # Increase batch size
  --max-num-seqs 256             # More concurrent requests
```

**SGLang batching**:
```yaml
command: >
  python -m sglang.launch_server
  --model-path ${TEACHER_MODEL}
  --max-running-requests 128     # More concurrent requests
```

### Reduce Latency

**Enable continuous batching**:
```yaml
# vLLM
command: >
  python -m vllm.entrypoints.openai.api_server
  --disable-log-requests         # Less logging overhead
```

## Development

### Local Development

```bash
# Mount code directory for live updates
docker-compose up training

# In container, code changes are immediately visible
docker-compose exec training bash
python -m qwen3_distill.pipeline ...
```

### Debugging

**Attach to running container**:
```bash
docker-compose exec training bash
docker-compose exec vllm-student bash
docker-compose exec sglang-teacher bash
```

**View logs**:
```bash
docker-compose logs -f --tail=100 training
```

**Python debugging**:
```bash
# In training container
pip install ipdb
# Add breakpoint in code: import ipdb; ipdb.set_trace()
python -m qwen3_distill.pipeline ...
```

## Deployment to Remote Server

See `deploy.sh` for automated deployment to `pokazge@spark-129a.local`.

```bash
# Deploy with multi-container architecture
./deploy.sh
```

The script will:
1. Sync code to remote server
2. Build all containers on remote
3. Start services
4. Run health checks
5. Report status

## Comparison: Single vs Multi-Container

| Aspect | Single Container | Multi-Container |
|--------|-----------------|-----------------|
| **Dependencies** | ❌ Conflicts | ✅ Isolated |
| **vLLM Version** | ❌ Custom build | ✅ Official NVIDIA |
| **Debugging** | ❌ Complex | ✅ Simple |
| **Resource Control** | ❌ Limited | ✅ Fine-grained |
| **Build Time** | ❌ 10-20 min | ✅ 5 min (cached) |
| **Flexibility** | ❌ Monolithic | ✅ Modular |
| **Memory Overhead** | ✅ Lower (~5GB) | ❌ Higher (~10GB) |
| **Startup Time** | ✅ Faster (~2 min) | ❌ Slower (~5 min) |

**Recommendation**: Use multi-container for development and production. The slight overhead is worth the reliability and maintainability.

## Next Steps

1. **Configure models**: Edit `.env` for your GPU size
2. **Start services**: `docker-compose up -d`
3. **Monitor startup**: `docker-compose logs -f`
4. **Run training**: See `QUICKSTART.md` for training examples
5. **Deploy**: Use `deploy.sh` for remote deployment

For questions, see `README.md` or check logs with `docker-compose logs`.
