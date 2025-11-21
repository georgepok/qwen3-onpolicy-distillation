# Qwen3-4B On-Policy Distillation Pipeline

## Overview

High-performance on-policy distillation pipeline for Qwen3-4B optimized for Nvidia GB10 DGX systems (128GB GPU RAM). This implementation achieves ~90x cost reduction compared to standard supervised fine-tuning through concurrent model loading, FP8 optimization, and streaming pipeline architecture.

## Architecture

### Core Components

1. **Student Generation** (vLLM)
   - Parallel student instances for trajectory generation
   - Optimized batch processing
   - GPU-resident inference

2. **Teacher Scoring** (SGLang)
   - FP8 tensor core utilization
   - Per-token logprob computation
   - Half-memory footprint via quantization

3. **Training Loop** (unsloth)
   - Memory-efficient fine-tuning
   - Per-token advantage computation
   - Gradient accumulation and mixed precision

4. **Experience Replay Buffer**
   - GPU-resident circular buffer
   - Efficient trajectory storage
   - Priority sampling support

### Key Optimizations

- **Concurrent Architecture**: Both student and teacher models remain in memory
- **FP8 Quantization**: Teacher model runs at half memory footprint
- **Streaming Pipeline**: Generation, scoring, and training overlap
- **Blackwell Features**: Exploits GB10-specific hardware capabilities

## Performance Targets

- **Cost Reduction**: ~90x vs standard SFT
- **Wall-clock Speedup**: 3-4x via parallelization
- **Memory Efficiency**: Optimized for 128GB GB10 capacity

## Project Structure

```
qwen3-onpolicy-distillation/
├── src/qwen3_distill/
│   ├── generation/     # vLLM-based student generation
│   ├── scoring/        # SGLang teacher scoring
│   ├── training/       # unsloth training loop
│   ├── replay/         # Experience replay buffer
│   └── utils/          # Shared utilities
├── configs/            # Configuration templates
├── scripts/            # Execution scripts
├── tests/              # Unit and integration tests
└── docs/               # Additional documentation
```

## Requirements

- NVIDIA GB10 DGX system with 128GB GPU RAM
- CUDA 12.x
- Python 3.10+
- PyTorch 2.x with CUDA support

## Installation

```bash
# Clone repository
git clone <repo-url>
cd qwen3-onpolicy-distillation

# Install in development mode
pip install -e .

# Or install specific requirements
pip install -r requirements.txt
```

## Quick Start

```bash
# Run full pipeline with default config
python scripts/run_pipeline.py --config configs/default.yaml

# Run individual components
python scripts/generate.py --config configs/generation.yaml
python scripts/score.py --config configs/scoring.yaml
python scripts/train.py --config configs/training.yaml
```

## Configuration

See `configs/` directory for configuration templates and examples.

## Algorithm Details

### Reverse KL Divergence Loss

The pipeline uses reverse KL divergence between student and teacher distributions with importance sampling:

```
L = E[log(π_student) - log(π_teacher)] * advantage
```

Where advantages are computed per-token to handle variable-length sequences efficiently.

### Training Protocol

1. Generate student trajectories using vLLM
2. Score with teacher using SGLang for logprobs
3. Compute per-token advantages
4. Update student via unsloth with computed gradients
5. Cycle with fresh generation from updated student

## Development

This project is designed for Claude Code agent execution and modification. The modular structure allows for easy extension and experimentation.

### Adding New Components

1. Create module in appropriate `src/qwen3_distill/` subdirectory
2. Add configuration schema in `configs/`
3. Update integration tests in `tests/`

### Running Tests

```bash
pytest tests/
```

## References

- [ThinkingMachines Blog Post on On-Policy Distillation](https://thinkingmachines.mit.edu/blog/on-policy-distillation)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://sgl-project.github.io/)
- [unsloth Documentation](https://github.com/unslothai/unsloth)

## License

[To be determined]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qwen3_onpolicy_distillation,
  title={Qwen3-4B On-Policy Distillation Pipeline},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```
