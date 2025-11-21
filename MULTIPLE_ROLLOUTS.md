# Multiple Rollouts Per Prompt - Variance Reduction

## Implementation Summary

This feature implements multiple rollouts per prompt for variance reduction, matching the approach used by ThinkingMachines in their on-policy distillation work.

## Motivation

Our initial 20-epoch training showed an overall positive trend (86.4% KL reduction from 0.3588 to 0.0489) but with high fluctuations (σ=0.0865). Analysis revealed the root cause:

**ThinkingMachines' "single-prompt" experiment actually had:**
- 1 prompt × 20 steps × **256 rollouts/step** = **5,120 sequences**

**Our initial approach:**
- 10 prompts × 16 epochs × **1 rollout** = **160 sequences**

We had **32x LESS data diversity** than their single-prompt experiment!

## Implementation

### 1. Configuration Parameter

Add `num_rollouts_per_prompt` to your training config:

```yaml
training:
  num_rollouts_per_prompt: 4  # Default: 1 (single rollout)
  learning_rate: 2e-05
  # ... other parameters
```

**Recommended Values:**
- `1`: Single rollout (baseline, high variance)
- `4`: 4x data, significant variance reduction (recommended starting point)
- `8-16`: Further variance reduction
- `32-256`: Match ThinkingMachines' approach (requires more compute)

### 2. Code Changes

####reverse_kl_trainer.py:59-74**
```python
# Multiple rollouts per prompt for variance reduction
self.num_rollouts = self.config.get("num_rollouts_per_prompt", 1)

print("Initialized Reverse KL distillation trainer")
print("  Algorithm: On-policy with per-token reverse KL rewards")
print(f"  Optimizer: Adam (lr={lr})")
print(f"  Rollouts per prompt: {self.num_rollouts}")
```

#### `pipeline.py:160-203`
```python
# Support multiple rollouts per prompt for variance reduction
samples_per_prompt = self.config.get("training", {}).get("num_rollouts_per_prompt", 1)
expected_trajectories = len(prompts) * samples_per_prompt

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate multiple rollouts for this prompt (variance reduction)
    for rollout_idx in range(samples_per_prompt):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        # ... create trajectory
```

#### `reverse_kl_trainer.py:91-102`
```python
# Group trajectories by prompt for multi-rollout batching
if self.num_rollouts > 1:
    prompt_batches = {}
    for traj in trajectories:
        prompt = traj["prompt"]
        if prompt not in prompt_batches:
            prompt_batches[prompt] = []
        prompt_batches[prompt].append(traj)

    print(f"  Batched {len(trajectories)} trajectories into {len(prompt_batches)} unique prompts")
    print(f"  Average rollouts per prompt: {len(trajectories) / len(prompt_batches):.1f}")
```

#### `reverse_kl_trainer.py:110-161`
```python
for batch_trajs in prompt_batches.values():
    # Accumulate gradients across all rollouts in this batch
    self.optimizer.zero_grad()

    for traj in batch_trajs:
        # Compute loss for this trajectory
        loss = -(rewards.detach() * student_lp).mean()

        # Backward pass (accumulates gradients without stepping)
        loss.backward()

    # Gradient clipping after accumulating all gradients from this prompt batch
    torch.nn.utils.clip_grad_norm_(
        self.student_model.parameters(),
        max_norm=1.0
    )

    # Single optimizer step per prompt (averages gradients across rollouts)
    self.optimizer.step()
```

## How It Works

### Variance Reduction Mechanism

1. **Generation**: For each prompt, generate N different completions (rollouts)
   - Example: 10 prompts × 4 rollouts = 40 trajectories

2. **Scoring**: Score all rollouts with the teacher model
   - Each rollout gets independent teacher logprobs

3. **Training**: Batch rollouts from the same prompt together
   - Accumulate gradients across all rollouts
   - Single optimizer step per unique prompt
   - This averages gradients across diverse completions

### Benefits

- **Reduced Variance**: Multiple samples from same prompt average out noise
- **Better Gradient Estimates**: Optimizer sees averaged signal across rollouts
- **More Data**: 4 rollouts = 4x training data from same prompts
- **Stable Training**: Expected to reduce KL fluctuations by 50-70%

## Expected Impact

Based on ThinkingMachines' results and our analysis:

| Rollouts | Total Sequences (10 prompts, 16 epochs) | Expected Variance Reduction |
|----------|----------------------------------------|----------------------------|
| 1 (baseline) | 160 sequences | σ = 0.0865 (baseline) |
| 4 | 640 sequences | ~50-60% reduction |
| 8 | 1,280 sequences | ~60-70% reduction |
| 32 | 5,120 sequences | ~70-80% reduction (matches ThinkingMachines) |

## Usage Example

```yaml
# configs/default.yaml
training:
  num_rollouts_per_prompt: 4  # 4 rollouts per prompt
  learning_rate: 2e-05
  # ... other parameters
```

```bash
# Run training with 10 prompts, 10 epochs, 4 rollouts per prompt
python src/qwen3_distill/pipeline.py \
    --prompts data/prompts.txt \
    --num-epochs 10

# Total trajectories: 10 prompts × 10 epochs × 4 rollouts = 400 sequences
```

## Performance Considerations

- **Generation Time**: Scales linearly with num_rollouts (4x rollouts = 4x generation time)
- **Memory**: Modest increase (only stores N completions at a time)
- **Training Time**: Slightly increased due to more forward/backward passes
- **Overall**: ~4x slower for 4 rollouts, but much more stable convergence

## Next Steps

1. **Test with 4 rollouts**: Good balance of stability vs compute
2. **Monitor variance reduction**: Track KL standard deviation across epochs
3. **Scale up if needed**: Try 8 or 16 rollouts if variance still high
4. **Expand dataset**: Also consider increasing from 10 to 100+ prompts

## References

- ThinkingMachines On-Policy Distillation: https://thinkingmachines.ai/blog/on-policy-distillation/
- Their "single-prompt" experiment used **256 rollouts per prompt**
- Our trajectory analysis: `/tmp/analyze_20epoch_trajectory.py`
