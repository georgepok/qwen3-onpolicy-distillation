# Evaluation Tools - Ready to Use

## Summary

Three comprehensive evaluation tools have been prepared and deployed to the remote server. They are ready to run immediately after training completes.

## Tools Deployed

### 1. Quick Quality Evaluation (`quick_eval.py`)
- **Purpose**: Compare student vs teacher on diverse prompts
- **Runtime**: ~2-3 minutes
- **Metrics**: Diversity, coherence, sample outputs
- **Location**: `~/qwen3-distill/evaluation/quick_eval.py`

### 2. KL Divergence Diagnostic (`check_kl_concern.py`)
- **Purpose**: Analyze the -8.66 KL divergence
- **Runtime**: ~2-3 minutes
- **Checks**: Repetition, mode collapse, overconfidence
- **Location**: `~/qwen3-distill/evaluation/check_kl_concern.py`

### 3. GSM8K Math Benchmark (`benchmark_gsm8k.py`)
- **Purpose**: Test math reasoning on 10 problems
- **Runtime**: ~3-5 minutes
- **Metrics**: Accuracy, retention rate vs teacher
- **Location**: `~/qwen3-distill/evaluation/benchmark_gsm8k.py`

## How to Run After Training

### Option 1: Run All at Once (Recommended)

```bash
ssh pokazge@spark-129a.local
cd ~/qwen3-distill

# Run all evaluations sequentially
docker compose run --rm training python evaluation/quick_eval.py
docker compose run --rm training python evaluation/check_kl_concern.py
docker compose run --rm training python evaluation/benchmark_gsm8k.py

# Check results
cat evaluation/results/quick_eval_results.json
cat evaluation/results/kl_diagnostic.json
cat evaluation/results/gsm8k_results.json
```

**Total time**: ~10 minutes

### Option 2: Run Individual Tools

```bash
# Just check diversity and quality
docker compose run --rm training python evaluation/quick_eval.py

# Just check if negative KL is concerning
docker compose run --rm training python evaluation/check_kl_concern.py

# Just test math reasoning
docker compose run --rm training python evaluation/benchmark_gsm8k.py
```

## What to Look For

### ✅ Good Signs

**Diversity**:
- Type-token ratio gap < 0.1
- Unique n-gram ratios > 0.7
- Low repetition score < 0.15

**KL Divergence**:
- No mode collapse (output diversity > 0.8)
- Reasonable repetition
- Not overly confident (prob > 0.99 < 50%)

**Reasoning**:
- GSM8K retention ≥ 90%
- Accuracy within 1-2 problems of teacher

### ⚠️ Warning Signs

**Diversity**:
- Type-token ratio gap > 0.2
- Many repeated trigrams
- High repetition score > 0.3

**KL Divergence**:
- Mode collapse score > 0.5
- Very high confidence ratio > 0.7
- Many duplicate outputs

**Reasoning**:
- GSM8K retention < 85%
- Missing basic math problems

### ❌ Red Flags

- Mode collapsed (few unique outputs)
- Repetition score > 0.5
- GSM8K retention < 70%
- Teacher gap > 30%

## Expected Results (Healthy Distillation)

Based on your current metrics (Loss: 1.076, KL: -8.66):

### Diversity
```
Type-Token Ratio: 0.5-0.7 (student), 0.6-0.8 (teacher)
Gap: ±0.05
Unique Bigrams: >0.7
Unique Trigrams: >0.8
```

### KL Diagnostic
```
Repetition Score: 0.10-0.20 (acceptable)
Mode Collapse: No (diversity > 0.8)
Overconfidence: Moderate (expected with negative KL)
Verdict: Acceptable if outputs diverse
```

### GSM8K
```
Student Accuracy: 70-90% (7-9 out of 10)
Teacher Accuracy: 80-100% (8-10 out of 10)
Retention: 85-95%
Verdict: Good to Excellent
```

## Results Interpretation

### If All Tests Pass

**Verdict**: Distillation successful!
- Student learned effectively from teacher
- Negative KL is benign (student found good patterns)
- Reasoning capability preserved
- Multi-sampling and policy sync working

### If Diversity Issues Detected

**Action**: Negative KL is concerning
- Increase generation temperature
- Add repetition penalty
- Consider retraining with lower KL coefficient

### If Reasoning Issues Detected

**Action**: Capability degradation
- Add more reasoning examples to training
- Increase model capacity (higher LoRA rank)
- Review training data quality

## Next Steps After Evaluation

1. **Review Results**: Check all three output files
2. **Compare to Baseline**: If you have pre-change metrics
3. **Document Findings**: Note improvements from multi-sampling + policy sync
4. **Share Results**: Prepare summary of distillation quality

## Files Generated

```
evaluation/results/
├── quick_eval_results.json       # Diversity & coherence
├── kl_diagnostic.json            # KL analysis
└── gsm8k_results.json            # Math reasoning
```

## Troubleshooting

### If Evaluation Fails

```bash
# Check if services are running
docker compose ps

# Restart if needed
docker compose restart vllm-student vllm-teacher

# Check logs
docker compose logs vllm-student
docker compose logs vllm-teacher
```

### If Out of Memory

Evaluations use minimal GPU memory, but if issues occur:
```bash
# Reduce batch size in evaluation scripts
# Edit the scripts and set samples_per_prompt=1
```

## Full Documentation

See `evaluation/README.md` for:
- Detailed metric explanations
- Troubleshooting guide
- Advanced evaluation options
- Baseline comparisons

## Summary

You now have a complete, ready-to-run evaluation suite that will:
1. Assess generation quality (diversity, coherence)
2. Diagnose KL divergence concerns
3. Test reasoning capability retention

Total runtime: **~10 minutes** for all three evaluations.

Results will clearly show whether the distillation succeeded and whether the negative KL is benign or concerning.
