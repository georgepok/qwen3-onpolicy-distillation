# Evaluation Tools

Comprehensive evaluation suite for on-policy distillation quality assessment.

## Tools Overview

### 1. Quick Quality Evaluation (`quick_eval.py`)

**Purpose**: Fast evaluation comparing student vs teacher on diverse test prompts.

**Metrics**:
- **Diversity**: Type-token ratio, unique n-grams, repetition
- **Coherence**: Sentence structure, completion rate
- **Sample outputs**: Side-by-side comparison

**Usage**:
```bash
cd ~/qwen3-distill
docker compose run --rm training python evaluation/quick_eval.py
```

**Output**: `evaluation/results/quick_eval_results.json`

**Time**: ~2-3 minutes

---

### 2. KL Divergence Diagnostic (`check_kl_concern.py`)

**Purpose**: Analyze the -8.66 KL divergence to determine if it's problematic.

**Checks**:
- Repetition patterns in outputs
- Mode collapse detection
- Overconfidence analysis
- Distribution narrowing

**Usage**:
```bash
docker compose run --rm training python evaluation/check_kl_concern.py
```

**Output**: `evaluation/results/kl_diagnostic.json`

**Time**: ~2-3 minutes

**Interpretation**:
- If outputs are diverse → Negative KL is acceptable (student learned better patterns)
- If outputs are repetitive → Negative KL is concerning (mode collapse)

---

### 3. GSM8K Math Benchmark (`benchmark_gsm8k.py`)

**Purpose**: Test math reasoning capability on 10 GSM8K problems.

**Metrics**:
- Accuracy (% correct)
- Retention rate (student vs teacher)
- Gap analysis

**Usage**:
```bash
docker compose run --rm training python evaluation/benchmark_gsm8k.py
```

**Output**: `evaluation/results/gsm8k_results.json`

**Time**: ~3-5 minutes

**Success Criteria**:
- Retention ≥95%: Excellent
- Retention ≥90%: Good
- Retention ≥85%: Fair
- Retention <85%: Poor

---

## Recommended Workflow

### After Training Completes

**Step 1: Quick Quality Check** (2 min)
```bash
docker compose run --rm training python evaluation/quick_eval.py
```
Look for:
- Type-token ratio gap < 0.1 (diversity preserved)
- Completion rate ≥ 90%
- Sample outputs look coherent

**Step 2: KL Diagnostic** (2 min)
```bash
docker compose run --rm training python evaluation/check_kl_concern.py
```
Look for:
- No mode collapse
- Reasonable repetition scores
- Confidence levels not too peaked

**Step 3: Math Reasoning** (3 min)
```bash
docker compose run --rm training python evaluation/benchmark_gsm8k.py
```
Look for:
- Retention ≥90% of teacher's accuracy

**Total time**: ~7-10 minutes for complete evaluation

---

## Running All Evaluations at Once

Create and run this script:

```bash
#!/bin/bash
# evaluation/run_all.sh

echo "=== Running Complete Evaluation Suite ==="
echo ""

echo "[1/3] Quick Quality Evaluation..."
docker compose run --rm training python evaluation/quick_eval.py

echo ""
echo "[2/3] KL Divergence Diagnostic..."
docker compose run --rm training python evaluation/check_kl_concern.py

echo ""
echo "[3/3] GSM8K Math Benchmark..."
docker compose run --rm training python evaluation/benchmark_gsm8k.py

echo ""
echo "=== All Evaluations Complete ==="
echo "Results saved to: evaluation/results/"
echo ""
echo "Summary:"
cat evaluation/results/quick_eval_results.json | grep -A 3 "diversity_gap"
cat evaluation/results/kl_diagnostic.json | grep "issues_detected"
cat evaluation/results/gsm8k_results.json | grep "retention_rate"
```

Make it executable and run:
```bash
chmod +x evaluation/run_all.sh
./evaluation/run_all.sh
```

---

## Interpreting Results

### Diversity Metrics

**Type-Token Ratio** (unique words / total words):
- 0.6-0.8: Excellent diversity
- 0.4-0.6: Good diversity
- 0.2-0.4: Low diversity (repetitive)
- <0.2: Very repetitive (mode collapse)

**Unique N-grams**:
- Bigrams > 0.7: Good
- Trigrams > 0.8: Good
- Lower values indicate repetitive patterns

### Repetition Score

- <0.15: Low repetition (good)
- 0.15-0.30: Moderate repetition (acceptable)
- >0.30: High repetition (concerning)

### Mode Collapse Score

- <0.2: No mode collapse
- 0.2-0.5: Some mode collapse
- >0.5: Severe mode collapse

### Confidence Analysis

**High Confidence Ratio** (prob > 0.9):
- 0.3-0.6: Normal
- >0.7: Overconfident (may indicate overfitting)

### GSM8K Retention

**Retention Rate** = Student Accuracy / Teacher Accuracy

- ≥0.95 (95%): Excellent - Student retained almost all capability
- ≥0.90 (90%): Good - Minor capability loss
- ≥0.85 (85%): Fair - Noticeable but acceptable loss
- <0.85 (85%): Poor - Significant capability degradation

---

## Expected Baselines

For a well-distilled 4B student from 32B teacher:

### Diversity
- Type-token ratio gap: ±0.05
- Unique bigrams gap: ±0.03
- Unique trigrams gap: ±0.02

### Quality
- Completion rate: ≥95%
- Avg sentence length: ±2 words from teacher

### Reasoning
- GSM8K retention: ≥90%
- Math accuracy: 7-9 out of 10 problems

### KL Divergence
- Absolute value: <5.0 (ideally <2.0)
- If negative: Check that repetition_score < 0.2

---

## Troubleshooting

### Issue: High Repetition

**Symptoms**: Repetition score > 0.3, many repeated trigrams

**Fixes**:
1. Increase temperature during generation (0.7 → 0.9)
2. Add repetition penalty in vLLM config
3. Reduce KL coefficient in training

### Issue: Mode Collapse

**Symptoms**: Mode collapse score > 0.5, few unique outputs

**Fixes**:
1. Add diversity loss to training objective
2. Increase LoRA rank for more capacity
3. Reduce training epochs (early stopping)

### Issue: Overconfidence

**Symptoms**: Very high confidence ratio > 0.7

**Fixes**:
1. Use label smoothing in training
2. Increase LoRA dropout
3. Add entropy regularization

### Issue: Poor Reasoning

**Symptoms**: GSM8K retention < 85%

**Fixes**:
1. Add more reasoning examples to training data
2. Increase max_tokens for longer reasoning chains
3. Use chain-of-thought prompting format

---

## Advanced Evaluation (Optional)

For more comprehensive evaluation, consider:

1. **MT-Bench**: Instruction-following benchmark
2. **MMLU**: Knowledge retention across domains
3. **HumanEval**: Code generation capability
4. **AlpacaEval 2.0**: GPT-4 judge comparison
5. **Human Evaluation**: Blind A/B testing

See `evaluation_strategy.md` for full details.

---

## Files Generated

After running all evaluations:

```
evaluation/results/
├── quick_eval_results.json       # Diversity & coherence metrics
├── kl_diagnostic.json            # KL divergence analysis
└── gsm8k_results.json            # Math reasoning results
```

Each file contains:
- Quantitative metrics
- Detailed per-sample results
- Comparison with teacher
- Diagnostic assessments
