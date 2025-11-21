# TRL-Based On-Policy Distillation Pipeline

**Framework**: [HuggingFace TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning)  
**Algorithm**: Reverse KL On-Policy Distillation  
**Reference**: [Thinking Machines - On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

## Overview

This pipeline implements **on-policy distillation** using TRL's `PPOTrainer` with custom **Reverse KL rewards**. The student model learns to match the teacher's token-level distribution through dense, per-token supervision.

### Key Characteristics

- ✅ **Per-token Reverse KL**: Every token graded by teacher
- ✅ **Dense supervision**: Immediate feedback per token
- ✅ **Mode-seeking**: Learns teacher's specific behavior
- ✅ **Policy gradient**: PPO optimization
- ✅ **Infrastructure**: TRL logging, distributed training, checkpointing

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PIPELINE ORCHESTRATOR                 │
│              (DistillationPipeline)                     │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
│   STUDENT    │   │    TEACHER     │   │   TRL TRAINER    │
│  GENERATOR   │   │    SCORER      │   │   (PPOTrainer)   │
│   (vLLM)     │   │ (vLLM/SGLang)  │   │  + Reverse KL    │
└──────────────┘   └────────────────┘   └──────────────────┘
```

### Component Breakdown

#### 1. **Student Generator** ([`generation/generator.py`](file:///Users/George/Documents/GitHub/qwen3-onpolicy-distillation/src/qwen3_distill/generation/generator.py))
- Runs student model via vLLM API
- Generates trajectories (completions)
- Extracts student logprobs during generation
- Handles batching and parallelization

#### 2. **Teacher Scorer** ([`scoring/scorer.py`](file:///Users/George/Documents/GitHub/qwen3-onpolicy-distillation/src/qwen3_distill/scoring/scorer.py))
- Runs teacher model via vLLM/SGLang API
- Computes teacher logprobs on student's tokens
- Calculates KL divergence
- Provides dense per-token feedback

#### 3. **TRL Trainer** ([`training/trl_trainer.py`](file:///Users/George/Documents/GitHub/qwen3-onpolicy-distillation/src/qwen3_distill/training/trl_trainer.py))
- Wraps TRL's `PPOTrainer`
- Computes Reverse KL rewards
- Executes policy gradient updates
- Handles logging and checkpointing

#### 4. **Pipeline Orchestrator** ([`pipeline.py`](file:///Users/George/Documents/GitHub/qwen3-onpolicy-distillation/src/qwen3_distill/pipeline.py))
- Coordinates all components
- Manages training loop
- Synchronizes LoRA adapters
- Handles checkpointing

---

## Training Flow

### High-Level Process

```
For each epoch:
  1. Generate trajectories from student
  2. Score trajectories with teacher
  3. Compute Reverse KL rewards
  4. Train student with PPO
  5. Synchronize LoRA adapters
  6. Checkpoint models
```

### Detailed Flow

```
┌─────────────────────────────────────────────────────────┐
│ EPOCH START                                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 1. GENERATION PHASE                                     │
│    StudentGenerator (vLLM)                              │
│    ┌─────────────────────────────────────────────────┐  │
│    │ • Load prompts from dataset                     │  │
│    │ • Generate completions (batch_size × samples)   │  │
│    │ • Extract student logprobs                      │  │
│    │ • Return: trajectories with tokens + logprobs   │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. SCORING PHASE                                        │
│    TeacherScorer (vLLM/SGLang)                          │
│    ┌─────────────────────────────────────────────────┐  │
│    │ • Load student's generated trajectories         │  │
│    │ • Compute teacher logprobs on same tokens       │  │
│    │ • Calculate per-token KL divergence             │  │
│    │ • Return: scored trajectories with teacher data │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. REWARD COMPUTATION                                   │
│    TRLDistillationTrainer._compute_reverse_kl_rewards() │
│    ┌─────────────────────────────────────────────────┐  │
│    │ For each token:                                 │  │
│    │   reverse_kl = student_logprob - teacher_logprob│  │
│    │   reward = -reverse_kl                          │  │
│    │                                                 │  │
│    │ Returns: Per-token rewards (minimize KL)        │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. PPO TRAINING                                         │
│    TRL PPOTrainer                                       │
│    ┌─────────────────────────────────────────────────┐  │
│    │ • Compute advantages from rewards               │  │
│    │ • Policy gradient with importance sampling      │  │
│    │ • Gradient accumulation                         │  │
│    │ • Optimizer step (AdamW)                        │  │
│    │ • Learning rate scheduling                      │  │
│    │ • Log metrics (TensorBoard/Wandb)               │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. SYNCHRONIZATION                                      │
│    Pipeline.sync_lora_adapters()                        │
│    ┌─────────────────────────────────────────────────┐  │
│    │ • Export updated LoRA adapters                  │  │
│    │ • Reload into vLLM generator                    │  │
│    │ • Ensures student generates from updated policy │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 6. CHECKPOINTING                                        │
│    Pipeline.save_checkpoint()                           │
│    ┌─────────────────────────────────────────────────┐  │
│    │ • Save model state                              │  │
│    │ • Save training metrics                         │  │
│    │ • Save LoRA adapters                            │  │
│    └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
                  NEXT EPOCH
```

---

## Algorithm: Reverse KL Distillation

### Mathematical Formulation

**Reverse KL Divergence** (per token):
```
KL(π_student || π_teacher) = log π_student(token) - log π_teacher(token)
```

**Reward** (minimize KL):
```
reward(token) = -KL(π_student || π_teacher)
              = -(log π_student(token) - log π_teacher(token))
              = log π_teacher(token) - log π_student(token)
```

**Policy Gradient Objective**:
```
J(θ) = E[Σ_t reward_t]
     = E[Σ_t -(log π_θ(token_t) - log π_teacher(token_t))]
```

### Why Reverse KL?

1. **Mode-seeking**: Pulls student toward teacher's mode, not spreading across multiple modes
2. **Dense supervision**: Every token gets immediate feedback
3. **Unhackable**: Low KL always means better alignment with teacher
4. **Reduces exposure bias**: Student learns from its own distribution

From the [original article](https://thinkingmachines.ai/blog/on-policy-distillation/):
> "Reverse KL has natural synergy with RL, which generally optimizes a form of sequence-level reverse KL induced by the reward model. However, unlike most reward models in practice, the reverse KL is 'unhackable' in the sense that low KL always corresponds to a high probability of desirable behavior from the teacher model's point of view."

---

## Code Example

### Basic Usage

```python
from qwen3_distill.pipeline import DistillationPipeline
from qwen3_distill.utils.config import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Initialize pipeline
pipeline = DistillationPipeline(config)

# Load training prompts
prompts = load_prompts("data/math_prompts.jsonl")

# Run training
for epoch in range(num_epochs):
    # Generate trajectories
    trajectories = pipeline.generate_trajectories(prompts)
    
    # Score with teacher
    scored = pipeline.score_trajectories(trajectories)
    
    # Train with Reverse KL rewards
    metrics = pipeline.train_on_trajectories(scored)
    
    # Synchronize LoRA adapters
    pipeline.sync_lora_adapters()
    
    # Save checkpoint
    pipeline.save_checkpoint(epoch, metrics)
```

### TRL Trainer Direct Usage

```python
from qwen3_distill.training.trl_trainer import TRLDistillationTrainer
from qwen3_distill.scoring.scorer import TeacherScorer

# Initialize scorer
teacher_scorer = TeacherScorer(config["scoring"])

# Initialize trainer
trainer = TRLDistillationTrainer(
    student_model=student_model,
    teacher_scorer=teacher_scorer,
    tokenizer=tokenizer,
    config={
        "learning_rate": 2e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
    }
)

# Train
prompts = ["Solve: 2 + 2 = ?", "What is 5 × 3?"]
trainer.train(
    prompt_dataset=prompts,
    num_epochs=10,
    output_dir="./checkpoints"
)
```

---

## Configuration

### Training Config ([`configs/default.yaml`](file:///Users/George/Documents/GitHub/qwen3-onpolicy-distillation/configs/default.yaml))

```yaml
training:
  model_name: "Qwen/Qwen3-4B-Instruct-2507"
  
  # LoRA configuration
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  
  # Optimization
  learning_rate: 2.0e-5
  batch_size: 4
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0
  
  # Training duration
  warmup_steps: 100
  total_steps: 10000
  
  # Mixed precision
  mixed_precision: "bf16"
  use_flash_attention: true
  
  # KL coefficient (PPO uses this for Reverse KL)
  kl_coef: 1.0
```

### Key Parameters

- **`lora_rank`**: LoRA adapter rank (16-64)
- **`learning_rate`**: AdamW learning rate (1e-5 to 5e-5)
- **`batch_size`**: Batch size per device
- **`gradient_accumulation_steps`**: Accumulation for larger effective batch
- **`kl_coef`**: Weight for Reverse KL rewards (typically 1.0)

---

## TRL PPOTrainer Configuration

### PPOConfig Parameters

```python
from trl import PPOConfig

config = PPOConfig(
    # Optimization
    learning_rate=2e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    ppo_epochs=1,
    
    # Disable default KL penalty (we use our Reverse KL)
    init_kl_coef=0.0,
    target_kl=None,
    
    # Regularization
    max_grad_norm=1.0,
    
    # Generation
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    
    # Logging
    log_with="tensorboard",
    logging_steps=10,
    
    # Mixed precision
    bf16=True,
)
```

---

## Key Features

### 1. **Dense Per-Token Supervision**

Unlike sequence-level rewards, every token gets graded:

```python
# For each token in student's response:
for token in response:
    student_logprob = model.get_logprob(token)
    teacher_logprob = teacher.get_logprob(token)
    
    # Compute Reverse KL for this token
    reverse_kl = student_logprob - teacher_logprob
    
    # Negative KL as reward
    reward = -reverse_kl
```

### 2. **Mode-Seeking Behavior**

Reverse KL encourages the student to match the teacher's specific distribution:

- **Forward KL** (coverage): Student covers all teacher's modes → spreading
- **Reverse KL** (mode-seeking): Student focuses on teacher's peak mode → matching

### 3. **TRL Infrastructure**

Benefits from TRL's battle-tested implementation:

- ✅ Distributed training (multi-GPU)
- ✅ Mixed precision (BF16/FP16)
- ✅ Gradient accumulation
- ✅ Learning rate scheduling
- ✅ TensorBoard/Wandb logging
- ✅ Standard checkpointing

### 4. **LoRA Adapter Synchronization**

After each training step:
1. Export updated LoRA adapters
2. Reload into vLLM generator
3. Next generation uses updated policy

This ensures on-policy learning (student generates from current policy).

---

## Performance & Efficiency

### Compute Savings

From the [original article](https://thinkingmachines.ai/blog/on-policy-distillation/):

> "We find a baseline cost reduction of **9x** when the SFT dataset is given, or **30x** when including the full cost of teacher sampling for off-policy distillation."

### Memory Efficiency

- **LoRA adapters**: Only train small adapter weights (~1% of model)
- **Gradient checkpointing**: Reduced memory during backprop
- **vLLM serving**: Efficient inference for generation and scoring
- **Mixed precision**: BF16 reduces memory by 50%

---

## Logging & Metrics

### TensorBoard Metrics

PPOTrainer automatically logs:

- `loss/policy_loss`: Policy gradient loss
- `loss/value_loss`: Value function loss (if using critic)
- `rewards/mean`: Mean reward per batch
- `rewards/std`: Reward standard deviation
- `kl/mean`: Mean KL divergence
- `learning_rate`: Current LR

### Custom Metrics

Additional metrics from the pipeline:

- `generation/num_trajectories`: Trajectories generated
- `scoring/teacher_logprob_mean`: Average teacher confidence
- `scoring/kl_divergence_mean`: Average KL
- `training/grad_norm`: Gradient norm before clipping

---

## Comparison: PPO vs DPO

| Aspect | PPO + Reverse KL (Current) | DPO (Previous) |
|--------|---------------------------|----------------|
| **Rewards** | Per-token Reverse KL | Sequence-level preferences |
| **Supervision** | Dense (every token) | Sparse (pairwise) |
| **Objective** | Minimize KL(student ‖ teacher) | Maximize P(preferred > non-preferred) |
| **Loss** | Policy gradient | DPO preference loss |
| **From article?** | ✅ Yes | ❌ No |
| **Mode behavior** | Mode-seeking | May spread |

**Why PPO is correct**: The original article explicitly uses per-token Reverse KL with policy gradient, not preference-based DPO.

---

## File Structure

```
src/qwen3_distill/
├── generation/
│   └── generator.py         # Student trajectory generation (vLLM)
├── scoring/
│   └── scorer.py            # Teacher scoring (vLLM/SGLang)
├── training/
│   └── trl_trainer.py       # PPOTrainer with Reverse KL
├── utils/
│   ├── config.py            # Configuration loading
│   └── gpu.py               # GPU memory management
└── pipeline.py              # Main orchestrator

configs/
├── default.yaml             # Default configuration
├── qwen3-1.5b-7b.yaml      # Small model config
└── qwen3-7b-14b.yaml       # Large model config

tests/
└── test_trl_trainer.py      # TRL trainer tests
```

---

## Running the Pipeline

### 1. Start vLLM Servers

```bash
# Student generator (vLLM)
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --gpu-memory-utilization 0.2

# Teacher scorer (vLLM/SGLang)
python -m sglang.launch_server \
  --model Qwen/Qwen3-32B \
  --port 30000 \
  --quantization fp8
```

### 2. Run Training

```bash
python -m qwen3_distill.pipeline \
  --config configs/default.yaml \
  --num-epochs 10 \
  --output-dir ./checkpoints
```

### 3. Monitor Training

```bash
tensorboard --logdir ./checkpoints
```

---

## Best Practices

### 1. **Start Small**
- Use small batch sizes initially
- Test with 1-2 epochs first
- Validate metrics before scaling

### 2. **Monitor KL Divergence**
- Should decrease over time
- Typical range: 0.1 - 1.0
- If increasing: check learning rate

### 3. **Gradient Norm Tracking**
- Should be stable (< 10.0)
- Spikes indicate instability
- Adjust `max_grad_norm` if needed

### 4. **LoRA Synchronization**
- Sync after each epoch minimum
- More frequent for on-policy learning
- Verify generator reloads adapters

### 5. **Checkpointing**
- Save every epoch
- Keep last N checkpoints
- Test loading before long runs

---

## Troubleshooting

### Issue: High KL divergence not decreasing

**Causes**:
- Learning rate too low
- Student initialization too different from teacher
- Insufficient training steps

**Solutions**:
- Increase learning rate (2e-5 → 5e-5)
- Start from SFT checkpoint closer to teacher
- Train for more epochs

### Issue: Gradient explosion

**Causes**:
- Learning rate too high
- Batch size too small
- Numerical instability

**Solutions**:
- Reduce learning rate
- Increase gradient accumulation
- Enable gradient clipping (default: 1.0)

### Issue: OOM (Out of Memory)

**Causes**:
- Batch size too large
- Sequence length too long
- LoRA rank too high

**Solutions**:
- Reduce batch size
- Increase gradient accumulation
- Use smaller LoRA rank (16 → 8)
- Enable gradient checkpointing

---

## References

1. [On-Policy Distillation - Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
2. [TRL Documentation](https://huggingface.co/docs/trl)
3. [PPOTrainer API](https://huggingface.co/docs/trl/ppo_trainer)
4. [On-Policy Distillation of Language Models (Agarwal et al, 2023)](https://arxiv.org/abs/2306.13649)
5. [MiniLLM: Knowledge Distillation of LLMs (Gu et al, 2023)](https://arxiv.org/abs/2306.08543)

---

## Summary

This pipeline implements **Reverse KL on-policy distillation** using TRL's PPOTrainer:

✅ **Algorithm**: Per-token Reverse KL rewards  
✅ **Framework**: TRL PPOTrainer  
✅ **Supervision**: Dense (every token graded)  
✅ **Behavior**: Mode-seeking (matches teacher)  
✅ **Infrastructure**: Logging, distributed training, checkpointing  

The implementation follows the [Thinking Machines methodology](https://thinkingmachines.ai/blog/on-policy-distillation/) exactly, providing an efficient and correct approach to on-policy distillation.
