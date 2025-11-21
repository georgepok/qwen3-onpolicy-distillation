# Configuration Files for Model Switching

This directory contains pre-configured YAML files for different student-teacher model pairs. Simply use the `--config` flag to switch between models without any code changes.

## Available Configurations

### 1. **default.yaml** - Qwen3-4B ← Qwen3-32B (128GB GPU)
- **Student**: Qwen3-4B-Instruct
- **Teacher**: Qwen3-32B-Instruct (FP8)
- **Memory**: ~128GB required
- **Best for**: NVIDIA GB10, A100 80GB, H100

```bash
python -m qwen3_distill.pipeline \
  --config configs/default.yaml \
  --prompts data/prompts.txt \
  --num-epochs 10
```

### 2. **qwen3-7b-14b.yaml** - Qwen3-7B ← Qwen3-14B (80GB GPU)
- **Student**: Qwen3-7B-Instruct
- **Teacher**: Qwen3-14B-Instruct (FP8)
- **Memory**: ~70GB required
- **Best for**: A100 80GB, H100

```bash
python -m qwen3_distill.pipeline \
  --config configs/qwen3-7b-14b.yaml \
  --prompts data/prompts.txt \
  --num-epochs 10
```

### 3. **qwen3-1.5b-7b.yaml** - Qwen3-1.5B ← Qwen3-7B (48GB GPU)
- **Student**: Qwen3-1.5B-Instruct
- **Teacher**: Qwen3-7B-Instruct (FP8)
- **Memory**: ~35GB required
- **Best for**: A6000, RTX 6000 Ada

```bash
python -m qwen3_distill.pipeline \
  --config configs/qwen3-1.5b-7b.yaml \
  --prompts data/prompts.txt \
  --num-epochs 10
```

## Creating Custom Configurations

### Step 1: Validate Model Pair

Use the model validation utility to check if your desired model pair fits:

```bash
python -c "
from qwen3_distill.utils.models import validate_model_pair, list_compatible_pairs

# Validate specific pair
is_valid, msg = validate_model_pair(
    'Qwen/Qwen3-4B-Instruct',
    'Qwen/Qwen3-32B-Instruct',
    total_gpu_memory_gb=128,
    use_fp8_teacher=True
)
print(msg)

# Or list all compatible pairs
list_compatible_pairs(total_gpu_memory_gb=128, use_fp8_teacher=True)
"
```

### Step 2: Copy Template

```bash
cp configs/default.yaml configs/my-custom.yaml
```

### Step 3: Update Model Names

Edit the YAML file and change:
- `generation.model_name` - Your student model
- `scoring.model_name` - Your teacher model
- `training.model_name` - Same as student

### Step 4: Adjust Memory Allocation

Update GPU memory utilization based on your hardware:
- `generation.gpu_memory_utilization` - Fraction for student
- `scoring.gpu_memory_utilization` - Fraction for teacher (FP8)

### Step 5: Run

```bash
python -m qwen3_distill.pipeline \
  --config configs/my-custom.yaml \
  --prompts data/prompts.txt \
  --num-epochs 10
```

## Supported Model Families

Currently supported model families (must use same family for student and teacher):

### Qwen3 Series
- Qwen3-0.5B-Instruct
- Qwen3-1.5B-Instruct
- Qwen3-4B-Instruct
- Qwen3-7B-Instruct
- Qwen3-14B-Instruct
- Qwen3-32B-Instruct
- Qwen3-72B-Instruct

### Qwen2.5 Series (Backward Compatible)
- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct
- Qwen2.5-72B-Instruct

## Adding New Model Families

To add support for Llama, Mistral, or other model families:

1. Edit `src/qwen3_distill/utils/models.py`
2. Add model specs to `MODEL_REGISTRY`
3. Specify architecture type (e.g., "llama", "mistral")
4. Provide memory estimates
5. Create config file in `configs/`

Example:

```python
# In src/qwen3_distill/utils/models.py
MODEL_REGISTRY = {
    ...
    "meta-llama/Llama-3-8B-Instruct": ModelSpec(
        name="meta-llama/Llama-3-8B-Instruct",
        size_params=8,
        memory_fp16_gb=32,
        memory_fp8_gb=16,
        max_seq_length=8192,
        architecture="llama",
    ),
    ...
}
```

## Memory Estimation Formula

Rough estimates for total GPU memory needed:

```
Total = Teacher_Memory + Student_Memory + Training_Overhead

Where:
- Teacher_Memory = Model_Size_GB * (0.5 if FP8 else 1.0)
- Student_Memory = Model_Size_GB * 1.0  (FP16)
- Training_Overhead ≈ 35-40GB (LoRA, optimizer, replay buffer)
```

## Tips for Memory Optimization

1. **Enable FP8**: Always use `quantization: "fp8"` for teacher
2. **Reduce Parallel Instances**: Set `num_parallel_instances: 1`
3. **Smaller Batch Sizes**: Reduce `batch_size` if OOM
4. **Gradient Accumulation**: Increase `gradient_accumulation_steps`
5. **Replay Buffer**: Reduce `capacity` if needed
6. **Sequence Length**: Reduce `max_tokens` for longer contexts

## Troubleshooting

### Out of Memory (OOM)
- Use smaller models
- Enable FP8 quantization
- Reduce batch sizes
- Disable replay buffer GPU storage

### Model Not Found
- Add model to `MODEL_REGISTRY` in `utils/models.py`
- Check HuggingFace model ID is correct
- Ensure model is publicly accessible or HF token set

### Architecture Mismatch
- Ensure student and teacher are from same model family
- Don't mix Qwen + Llama or other combinations
- Use validation utility to check compatibility
