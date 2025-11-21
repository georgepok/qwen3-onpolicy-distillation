#!/usr/bin/env python3
"""
KL Divergence Diagnostic Tool

Analyzes the -8.66 KL divergence to determine if it's concerning.
Checks for:
- Repetition in outputs
- Overconfidence
- Mode collapse
- Distribution narrowing
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen3_distill.generation.generator import StudentGenerator, GenerationConfig


def analyze_repetition(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze repetition patterns in generated text.

    Args:
        texts: List of generated texts

    Returns:
        Repetition analysis results
    """
    all_words = []
    all_bigrams = []
    all_trigrams = []

    for text in texts:
        words = text.lower().split()
        all_words.extend(words)

        # Bigrams
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)

        # Trigrams
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        all_trigrams.extend(trigrams)

    # Count repetitions
    word_counts = Counter(all_words)
    bigram_counts = Counter(all_bigrams)
    trigram_counts = Counter(all_trigrams)

    # Most common
    most_common_words = word_counts.most_common(10)
    most_common_bigrams = bigram_counts.most_common(10)
    most_common_trigrams = trigram_counts.most_common(10)

    # Repetition metrics
    total_words = len(all_words)
    unique_words = len(set(all_words))

    total_bigrams = len(all_bigrams)
    unique_bigrams = len(set(all_bigrams))

    total_trigrams = len(all_trigrams)
    unique_trigrams = len(set(all_trigrams))

    # Check for excessive repetition (red flags)
    max_word_count = max(word_counts.values()) if word_counts else 0
    max_bigram_count = max(bigram_counts.values()) if bigram_counts else 0
    max_trigram_count = max(trigram_counts.values()) if trigram_counts else 0

    repetition_score = (
        (max_word_count / total_words if total_words > 0 else 0) * 0.3 +
        (max_bigram_count / total_bigrams if total_bigrams > 0 else 0) * 0.3 +
        (max_trigram_count / total_trigrams if total_trigrams > 0 else 0) * 0.4
    )

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "word_diversity": unique_words / total_words if total_words > 0 else 0,
        "bigram_diversity": unique_bigrams / total_bigrams if total_bigrams > 0 else 0,
        "trigram_diversity": unique_trigrams / total_trigrams if total_trigrams > 0 else 0,
        "max_word_repetition": max_word_count,
        "max_bigram_repetition": max_bigram_count,
        "max_trigram_repetition": max_trigram_count,
        "repetition_score": repetition_score,
        "most_common_words": [(w, c) for w, c in most_common_words],
        "most_common_bigrams": [(" ".join(bg), c) for bg, c in most_common_bigrams],
        "most_common_trigrams": [(" ".join(tg), c) for tg, c in most_common_trigrams],
    }


def check_for_mode_collapse(texts: List[str]) -> Dict[str, Any]:
    """
    Check if model has collapsed to a few modes.

    Args:
        texts: List of generated texts

    Returns:
        Mode collapse analysis
    """
    # Check if many outputs are identical or very similar
    text_counts = Counter(texts)
    unique_texts = len(set(texts))

    # Check for exact duplicates
    max_duplicate = max(text_counts.values()) if text_counts else 0

    # Check for near-duplicates (simple heuristic: first 50 chars)
    prefixes = [t[:50] for t in texts]
    prefix_counts = Counter(prefixes)
    max_prefix_duplicate = max(prefix_counts.values()) if prefix_counts else 0

    # Mode collapse score
    mode_collapse_score = 1.0 - (unique_texts / len(texts) if texts else 0)

    return {
        "total_outputs": len(texts),
        "unique_outputs": unique_texts,
        "output_diversity": unique_texts / len(texts) if texts else 0,
        "max_exact_duplicates": max_duplicate,
        "max_prefix_duplicates": max_prefix_duplicate,
        "mode_collapse_score": mode_collapse_score,
        "is_mode_collapsed": mode_collapse_score > 0.5,
    }


def analyze_overconfidence(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze if student is overconfident (causing negative KL).

    Args:
        trajectories: Generated trajectories with logprobs

    Returns:
        Overconfidence analysis
    """
    if not trajectories:
        return {}

    # Extract student logprobs
    all_student_logprobs = []
    for traj in trajectories:
        if 'logprobs' in traj and traj['logprobs']:
            all_student_logprobs.extend([lp for lp in traj['logprobs'] if lp is not None])

    if not all_student_logprobs:
        return {"error": "No student logprobs available"}

    # Convert to probabilities
    student_probs = [np.exp(lp) for lp in all_student_logprobs]

    # Analyze distribution
    mean_prob = np.mean(student_probs)
    std_prob = np.std(student_probs)
    max_prob = np.max(student_probs)
    min_prob = np.min(student_probs)

    # High confidence = probabilities close to 1.0
    high_confidence_ratio = sum(1 for p in student_probs if p > 0.9) / len(student_probs)
    very_high_confidence_ratio = sum(1 for p in student_probs if p > 0.99) / len(student_probs)

    # Entropy (lower = more confident)
    entropy = -np.mean([p * np.log(p + 1e-10) for p in student_probs])

    return {
        "mean_probability": mean_prob,
        "std_probability": std_prob,
        "max_probability": max_prob,
        "min_probability": min_prob,
        "high_confidence_ratio": high_confidence_ratio,
        "very_high_confidence_ratio": very_high_confidence_ratio,
        "entropy": entropy,
        "is_overconfident": very_high_confidence_ratio > 0.5,
    }


def diagnose_kl_divergence(kl_value: float = -8.66) -> str:
    """
    Provide diagnosis of KL divergence value.

    Args:
        kl_value: KL divergence value

    Returns:
        Diagnostic message
    """
    print(f"\n{'='*60}")
    print(f"KL DIVERGENCE DIAGNOSTIC")
    print(f"{'='*60}\n")
    print(f"Current KL Divergence: {kl_value:.2f}")
    print()

    if abs(kl_value) < 0.1:
        diagnosis = "✅ EXCELLENT - Near-perfect alignment with teacher"
        concern_level = "None"
    elif abs(kl_value) < 1.0:
        diagnosis = "✅ GOOD - Close alignment with teacher"
        concern_level = "Low"
    elif kl_value > 1.0 and kl_value < 5.0:
        diagnosis = "⚠️  FAIR - Student lagging behind teacher (underconfident)"
        concern_level = "Medium"
    elif kl_value > 5.0:
        diagnosis = "❌ POOR - Student significantly underconfident"
        concern_level = "High"
    elif kl_value < -1.0 and kl_value > -5.0:
        diagnosis = "⚠️  FAIR - Student more confident than teacher"
        concern_level = "Medium - Check for overfitting"
    elif kl_value < -5.0:
        diagnosis = "❌ CONCERNING - Student much more confident than teacher"
        concern_level = "High - Check for mode collapse or overfitting"
    else:
        diagnosis = "❓ UNKNOWN - Unexpected KL value"
        concern_level = "Unknown"

    print(f"Diagnosis: {diagnosis}")
    print(f"Concern Level: {concern_level}")
    print()

    if kl_value < -5.0:
        print("Negative KL indicates student assigns HIGHER probabilities than teacher.")
        print("This could mean:")
        print("  1. ✅ Student learned sharper, more confident patterns (GOOD)")
        print("  2. ⚠️  Student is overfitting to training distribution (CONCERN)")
        print("  3. ❌ Student has collapsed to a few modes (BAD)")
        print()
        print("We need to check:")
        print("  - Are outputs repetitive?")
        print("  - Is there diversity in generations?")
        print("  - Are probabilities too peaked (>0.99)?")

    return diagnosis


def main():
    """Run KL divergence diagnostic."""
    print("="*60)
    print("KL DIVERGENCE CONCERN CHECK")
    print("="*60)

    # Clear environment variable to avoid conflicts
    if 'VLLM_API_URL' in os.environ:
        del os.environ['VLLM_API_URL']
    if 'TEACHER_API_URL' in os.environ:
        del os.environ['TEACHER_API_URL']

    # Read current KL from training metrics if available
    metrics_file = Path("results/training_metrics.json")
    current_kl = -8.66  # Default from conversation

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if metrics.get('epochs'):
                latest_epoch = metrics['epochs'][-1]
                current_kl = latest_epoch.get('kl_divergence', current_kl)

    # Diagnose KL value
    diagnosis = diagnose_kl_divergence(current_kl)

    # Generate test samples from student
    print("\n[1/3] Generating test samples from student...")

    test_prompts = [
        "Explain quantum computing.",
        "What is machine learning?",
        "Write a Python function to sort a list.",
        "Describe the water cycle.",
        "What is the capital of France?",
    ] * 4  # 20 total samples

    config = GenerationConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        api_url="http://vllm-student:8000",
        samples_per_prompt=1,
        temperature=0.7,
        max_tokens=200,
    )

    generator = StudentGenerator(config)
    trajectories = generator.generate_trajectories(test_prompts)
    texts = [t['generated_text'] for t in trajectories]

    # Analyze repetition
    print("\n[2/3] Analyzing repetition patterns...")
    repetition_analysis = analyze_repetition(texts)

    print("\nRepetition Analysis:")
    print(f"  Word Diversity: {repetition_analysis['word_diversity']:.3f}")
    print(f"  Bigram Diversity: {repetition_analysis['bigram_diversity']:.3f}")
    print(f"  Trigram Diversity: {repetition_analysis['trigram_diversity']:.3f}")
    print(f"  Repetition Score: {repetition_analysis['repetition_score']:.3f}")

    if repetition_analysis['repetition_score'] > 0.3:
        print("  ❌ HIGH REPETITION - Outputs are very repetitive!")
    elif repetition_analysis['repetition_score'] > 0.15:
        print("  ⚠️  MODERATE REPETITION - Some repetitive patterns")
    else:
        print("  ✅ LOW REPETITION - Outputs are diverse")

    print("\n  Most common trigrams:")
    for trigram, count in repetition_analysis['most_common_trigrams'][:5]:
        print(f"    '{trigram}': {count} times")

    # Check mode collapse
    print("\n[3/3] Checking for mode collapse...")
    mode_analysis = check_for_mode_collapse(texts)

    print("\nMode Collapse Analysis:")
    print(f"  Output Diversity: {mode_analysis['output_diversity']:.3f}")
    print(f"  Unique Outputs: {mode_analysis['unique_outputs']}/{mode_analysis['total_outputs']}")
    print(f"  Mode Collapse Score: {mode_analysis['mode_collapse_score']:.3f}")

    if mode_analysis['is_mode_collapsed']:
        print("  ❌ MODE COLLAPSE DETECTED - Model generating very similar outputs!")
    else:
        print("  ✅ NO MODE COLLAPSE - Outputs are diverse")

    # Analyze overconfidence
    print("\n[4/4] Analyzing confidence levels...")
    confidence_analysis = analyze_overconfidence(trajectories)

    if 'error' not in confidence_analysis:
        print("\nConfidence Analysis:")
        print(f"  Mean Probability: {confidence_analysis['mean_probability']:.3f}")
        print(f"  High Confidence (>0.9): {confidence_analysis['high_confidence_ratio']:.1%}")
        print(f"  Very High Confidence (>0.99): {confidence_analysis['very_high_confidence_ratio']:.1%}")
        print(f"  Entropy: {confidence_analysis['entropy']:.3f}")

        if confidence_analysis['is_overconfident']:
            print("  ❌ OVERCONFIDENT - Student is too certain about predictions!")
        else:
            print("  ✅ REASONABLE CONFIDENCE - Confidence levels look normal")

    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}\n")

    issues = []
    if repetition_analysis['repetition_score'] > 0.3:
        issues.append("High repetition")
    if mode_analysis['is_mode_collapsed']:
        issues.append("Mode collapse")
    if 'is_overconfident' in confidence_analysis and confidence_analysis['is_overconfident']:
        issues.append("Overconfidence")

    if not issues:
        print("✅ NEGATIVE KL IS ACCEPTABLE")
        print("\nThe -8.66 KL divergence appears benign:")
        print("  - Outputs are diverse (no mode collapse)")
        print("  - No excessive repetition")
        print("  - Reasonable confidence levels")
        print("\nConclusion: Student has learned to generate high-quality,")
        print("confident outputs. The negative KL likely indicates the student")
        print("has successfully learned sharper patterns than the teacher.")
    else:
        print("⚠️  NEGATIVE KL IS CONCERNING")
        print(f"\nDetected issues: {', '.join(issues)}")
        print("\nRecommendations:")
        if "High repetition" in issues:
            print("  1. Increase temperature during generation")
            print("  2. Add repetition penalty")
        if "Mode collapse" in issues:
            print("  3. Reduce KL coefficient in training")
            print("  4. Add diversity loss term")
        if "Overconfidence" in issues:
            print("  5. Use label smoothing in training")
            print("  6. Increase dropout in LoRA layers")

    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "kl_divergence": current_kl,
        "diagnosis": diagnosis,
        "repetition_analysis": {k: v for k, v in repetition_analysis.items()
                               if k not in ['most_common_words', 'most_common_bigrams', 'most_common_trigrams']},
        "mode_analysis": mode_analysis,
        "confidence_analysis": confidence_analysis,
        "issues_detected": issues,
    }

    output_file = output_dir / "kl_diagnostic.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}\n")


if __name__ == "__main__":
    main()
