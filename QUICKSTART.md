# Quick Start Guide - Model Switching

The pipeline supports easy model switching without any code changes. Just use different config files!

## 🎯 Quick Model Switch

**Three simple steps:**

1. **Choose a config** from `configs/`
2. **Run validation** (optional but recommended)
3. **Start training**

## 📋 Pre-Configured Model Pairs

| Config File | Student | Teacher | GPU Memory | Best For |
|-------------|---------|---------|------------|----------|
| `default.yaml` | Qwen3-4B | Qwen3-32B | 128GB | GB10, A100-80GB, H100 |
| `qwen3-7b-14b.yaml` | Qwen3-7B | Qwen3-14B | 70GB | A100-80GB, H100 |
| `qwen3-1.5b-7b.yaml` | Qwen3-1.5B | Qwen3-7B | 35GB | A6000, RTX 6000 Ada |

## 🚀 Usage Examples

### Example 1: Default (Qwen3-4B ← Qwen3-32B)

```bash
python -m qwen3_distill.pipeline \
  --config configs/default.yaml \
  --prompts data/sample_prompts.txt \
  --num-epochs 10
```

### Example 2: Smaller Models (Qwen3-1.5B ← Qwen3-7B)

```bash
python -m qwen3_distill.pipeline \
  --config configs/qwen3-1.5b-7b.yaml \
  --prompts data/sample_prompts.txt \
  --num-epochs 10
```

### Example 3: Custom Models

**Step 1: Validate**
```bash
python scripts/validate_models.py \
  --student "Qwen/Qwen3-7B-Instruct" \
  --teacher "Qwen/Qwen3-32B-Instruct" \
  --gpu-memory 128
```

**Step 2: Create Config**
```bash
cp configs/default.yaml configs/my-custom.yaml
# Edit my-custom.yaml:
#   - Change generation.model_name to your student
#   - Change scoring.model_name to your teacher
#   - Change training.model_name to same as student
```

**Step 3: Run**
```bash
python -m qwen3_distill.pipeline \
  --config configs/my-custom.yaml \
  --prompts data/prompts.txt \
  --num-epochs 10
```

## 🔍 Finding Compatible Pairs

**List all compatible pairs for your GPU:**

```bash
python scripts/validate_models.py --list-all --gpu-memory 128
```

**List all available models:**

```bash
python scripts/validate_models.py --list-models
```

**Validate specific pair:**

```bash
python scripts/validate_models.py \
  --student "Qwen/Qwen3-4B-Instruct" \
  --teacher "Qwen/Qwen3-32B-Instruct" \
  --gpu-memory 128
```

## ✅ Automatic Validation

The pipeline **automatically validates** your model pair at startup:

```
Validating model pair:
  Student: Qwen/Qwen3-4B-Instruct
  Teacher: Qwen/Qwen3-32B-Instruct
  GPU Memory: 128.0GB
  Teacher Quantization: FP8

✓ Valid model pair:
  Student: Qwen/Qwen3-4B-Instruct (4.0B)
  Teacher: Qwen/Qwen3-32B-Instruct (32.0B)
  Memory: 110.0GB / 128.0GB (85.9% utilization)
```

If invalid, you'll get a clear error with suggestions.

## 🎨 Supported Models

### Qwen3 Series (Recommended)
- Qwen3-0.5B, 1.5B, 4B, 7B, 14B, 32B, 72B

### Qwen2.5 Series (Backward Compatible)
- Qwen2.5-0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B

### Adding New Models

Edit `src/qwen3_distill/utils/models.py`:

```python
MODEL_REGISTRY = {
    ...
    "your-org/your-model": ModelSpec(
        name="your-org/your-model",
        size_params=8.0,  # In billions
        memory_fp16_gb=32,  # Estimated FP16 memory
        memory_fp8_gb=16,   # Estimated FP8 memory
        max_seq_length=4096,
        architecture="qwen",  # Model family
    ),
    ...
}
```

## 🔧 Memory Optimization Tips

If you get OOM errors:

1. **Enable FP8** (already enabled by default in configs)
   ```yaml
   scoring:
     quantization: "fp8"  # ← Halves teacher memory
   ```

2. **Reduce batch size**
   ```yaml
   generation:
     batch_size: 16  # ← Reduce from 32
   training:
     batch_size: 2   # ← Reduce from 4
   ```

3. **Reduce parallel instances**
   ```yaml
   generation:
     num_parallel_instances: 1  # ← Already minimal
   ```

4. **Smaller replay buffer**
   ```yaml
   replay:
     capacity: 5000  # ← Reduce from 10000
   ```

5. **Use smaller models** (see compatible pairs above)

## 📊 Memory Estimation

Quick formula:
```
Total ≈ Teacher_FP8 + Student_FP16 + 40GB_overhead

Examples:
- 4B + 32B: 8GB + 16GB + 40GB = 64GB  ✗ Too tight
- 4B + 32B (FP8): 16GB + 45GB + 40GB = 101GB  ✓ Good
- 7B + 14B (FP8): 28GB + 28GB + 40GB = 96GB   ✓ Good
- 1.5B + 7B (FP8): 6GB + 14GB + 40GB = 60GB   ✓ Great
```

## 🐳 Docker Usage

All configs work with Docker:

```bash
docker run -d --gpus all --shm-size=32g \
  -v ~/qwen3-distill/code:/workspace/qwen3-distill \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  qwen3-distill:latest \
  python -m qwen3_distill.pipeline \
    --config configs/qwen3-7b-14b.yaml \
    --prompts data/prompts.txt \
    --num-epochs 10
```

## ❓ FAQ

**Q: Can I mix model families (e.g., Qwen student + Llama teacher)?**
A: No, student and teacher must be from the same family. The validation will catch this.

**Q: Does the teacher always need to be larger?**
A: Yes, for effective distillation. The validation enforces this.

**Q: Can I use non-instruct models?**
A: Yes, but add them to MODEL_REGISTRY first.

**Q: What if my model isn't in the registry?**
A: Add it to `utils/models.py` MODEL_REGISTRY with proper specs.

## 📚 More Information

- Full config documentation: `configs/README.md`
- Model registry details: `src/qwen3_distill/utils/models.py`
- Deployment guide: `deploy.sh`
- Main README: `README.md`
