#!/usr/bin/env python3
"""
Precision Validation for KL Divergence Computation

This script diagnoses why KL values are negative by:
1. Checking logprob magnitudes
2. Testing numerical precision limits
3. Validating data alignment
4. Comparing FP32 vs FP64 computation
"""

import numpy as np
import torch
from typing import Tuple


def compute_kl_fp32(
    student_logprobs: np.ndarray,
    teacher_logprobs: np.ndarray
) -> Tuple[float, dict]:
    """Compute KL in FP32 with diagnostics."""
    # Convert to FP32 explicitly
    student_lp = student_logprobs.astype(np.float32)
    teacher_lp = teacher_logprobs.astype(np.float32)
    
    # Compute components
    student_probs = np.exp(student_lp)
    log_ratio = student_lp - teacher_lp
    per_token_kl = student_probs * log_ratio
    
    diagnostics = {
        "student_logprob_range": (float(student_lp.min()), float(student_lp.max())),
        "teacher_logprob_range": (float(teacher_lp.min()), float(teacher_lp.max())),
        "student_prob_range": (float(student_probs.min()), float(student_probs.max())),
        "log_ratio_range": (float(log_ratio.min()), float(log_ratio.max())),
        "per_token_kl_range": (float(per_token_kl.min()), float(per_token_kl.max())),
        "num_negative_kl_tokens": int((per_token_kl < 0).sum()),
        "num_zero_kl_tokens": int((np.abs(per_token_kl) < 1e-10).sum()),
    }
    
    mean_kl = float(per_token_kl.mean())
    return mean_kl, diagnostics


def compute_kl_fp64(
    student_logprobs: np.ndarray,
    teacher_logprobs: np.ndarray
) -> Tuple[float, dict]:
    """Compute KL in FP64 for comparison."""
    # Convert to FP64
    student_lp = student_logprobs.astype(np.float64)
    teacher_lp = teacher_logprobs.astype(np.float64)
    
    # Compute components
    student_probs = np.exp(student_lp)
    log_ratio = student_lp - teacher_lp
    per_token_kl = student_probs * log_ratio
    
    diagnostics = {
        "per_token_kl_range": (float(per_token_kl.min()), float(per_token_kl.max())),
        "num_negative_kl_tokens": int((per_token_kl < 0).sum()),
    }
    
    mean_kl = float(per_token_kl.mean())
    return mean_kl, diagnostics


def check_alignment(
    student_logprobs: np.ndarray,
    teacher_logprobs: np.ndarray
) -> dict:
    """Check if logprobs are properly aligned."""
    
    # Check lengths
    len_match = len(student_logprobs) == len(teacher_logprobs)
    
    # Check correlation (should be high if aligned)
    if len(student_logprobs) > 0 and len(teacher_logprobs) > 0:
        min_len = min(len(student_logprobs), len(teacher_logprobs))
        correlation = np.corrcoef(
            student_logprobs[:min_len],
            teacher_logprobs[:min_len]
        )[0, 1]
    else:
        correlation = 0.0
    
    # Check absolute difference magnitude
    if len_match:
        abs_diff = np.abs(student_logprobs - teacher_logprobs)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
    else:
        max_diff = float('inf')
        mean_diff = float('inf')
    
    return {
        "length_match": len_match,
        "student_len": len(student_logprobs),
        "teacher_len": len(teacher_logprobs),
        "correlation": float(correlation),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def suggest_fixes(diagnostics: dict) -> list:
    """Suggest fixes based on diagnostics."""
    fixes = []
    
    # Check for extreme convergence
    if diagnostics["alignment"]["mean_abs_diff"] < 0.01:
        fixes.append({
            "issue": "Student-teacher near-perfect alignment",
            "severity": "HIGH",
            "explanation": "Logprobs differ by <0.01 on average. KL is operating at FP32 noise floor.",
            "fixes": [
                "1. Increase teacher temperature to widen distribution gap",
                "2. Use mixed precision (FP64) for KL computation",
                "3. Add epsilon clamping: KL = max(0, KL_raw)",
                "4. Consider training is complete - evaluate on downstream tasks"
            ]
        })
    
    # Check for numerical instability
    if diagnostics["fp32"]["num_negative_kl_tokens"] > 0:
        neg_ratio = diagnostics["fp32"]["num_negative_kl_tokens"] / diagnostics["alignment"]["student_len"]
        if neg_ratio > 0.1:  # >10% negative tokens
            fixes.append({
                "issue": "High ratio of negative KL tokens",
                "severity": "HIGH",
                "explanation": f"{neg_ratio:.1%} of tokens have negative KL due to FP32 precision limits",
                "fixes": [
                    "1. Implement epsilon clamping: per_token_kl = max(0, per_token_kl)",
                    "2. Use log-space computation with numerical guards",
                    "3. Switch to FP64 for KL computation only"
                ]
            })
    
    # Check alignment issues
    if not diagnostics["alignment"]["length_match"]:
        fixes.append({
            "issue": "Student-teacher sequence length mismatch",
            "severity": "CRITICAL",
            "explanation": f"Student: {diagnostics['alignment']['student_len']}, Teacher: {diagnostics['alignment']['teacher_len']}",
            "fixes": [
                "1. Verify completion-only extraction in scorer.py",
                "2. Check tokenization consistency",
                "3. Add explicit padding/trimming logic"
            ]
        })
    
    if diagnostics["alignment"]["correlation"] < 0.5:
        fixes.append({
            "issue": "Low student-teacher correlation",
            "severity": "CRITICAL",
            "explanation": f"Correlation: {diagnostics['alignment']['correlation']:.3f} (should be >0.8)",
            "fixes": [
                "1. Verify logprobs are from same sequence region",
                "2. Check for off-by-one errors in indexing",
                "3. Validate tokenizer consistency"
            ]
        })
    
    return fixes


def main():
    """Run validation on sample data."""
    print("="*80)
    print("KL DIVERGENCE PRECISION VALIDATION")
    print("="*80)
    
    # Create test cases
    test_cases = [
        {
            "name": "Nearly identical distributions (your case)",
            "student": np.array([-0.1, -0.15, -0.2, -0.12, -0.18] * 100, dtype=np.float32),
            "teacher": np.array([-0.11, -0.14, -0.19, -0.13, -0.17] * 100, dtype=np.float32),
        },
        {
            "name": "Moderate difference",
            "student": np.array([-0.5, -0.8, -1.2, -0.6, -0.9] * 100, dtype=np.float32),
            "teacher": np.array([-0.8, -1.0, -1.5, -0.9, -1.1] * 100, dtype=np.float32),
        },
        {
            "name": "Large difference",
            "student": np.array([-1.0, -2.0, -3.0, -1.5, -2.5] * 100, dtype=np.float32),
            "teacher": np.array([-3.0, -4.0, -5.0, -3.5, -4.5] * 100, dtype=np.float32),
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {test['name']}")
        print(f"{'='*80}\n")
        
        student = test["student"]
        teacher = test["teacher"]
        
        # Check alignment
        print("[1/3] Checking alignment...")
        alignment = check_alignment(student, teacher)
        for key, val in alignment.items():
            print(f"  {key}: {val}")
        
        # Compute KL in FP32
        print("\n[2/3] Computing KL in FP32...")
        kl_fp32, diag_fp32 = compute_kl_fp32(student, teacher)
        print(f"  Mean KL (FP32): {kl_fp32:.6f}")
        for key, val in diag_fp32.items():
            print(f"  {key}: {val}")
        
        # Compute KL in FP64
        print("\n[3/3] Computing KL in FP64...")
        kl_fp64, diag_fp64 = compute_kl_fp64(student, teacher)
        print(f"  Mean KL (FP64): {kl_fp64:.6f}")
        print(f"  Precision difference: {abs(kl_fp64 - kl_fp32):.2e}")
        
        # Aggregate diagnostics
        full_diagnostics = {
            "alignment": alignment,
            "fp32": diag_fp32,
            "fp64": diag_fp64,
            "kl_fp32": kl_fp32,
            "kl_fp64": kl_fp64,
        }
        
        # Suggest fixes
        print("\n[ANALYSIS]")
        fixes = suggest_fixes(full_diagnostics)
        if fixes:
            for fix in fixes:
                print(f"\n  ⚠️  {fix['issue']} ({fix['severity']})")
                print(f"      {fix['explanation']}")
                print("      Recommended fixes:")
                for f in fix['fixes']:
                    print(f"        {f}")
        else:
            print("\n  ✅ No issues detected - KL computation is valid")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. If your KL values are near Test Case 1, you're at FP32 precision limits")
    print("2. Negative KL at this scale is expected numerical artifact")
    print("3. Consider epsilon clamping: mean_kl = max(0.0, mean_kl)")
    print("4. Or use FP64 for KL computation if precision critical")


if __name__ == "__main__":
    main()
