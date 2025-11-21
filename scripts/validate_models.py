#!/usr/bin/env python3
"""
Model validation and compatibility checker.

Usage:
    python scripts/validate_models.py --student Qwen/Qwen3-4B-Instruct --teacher Qwen/Qwen3-32B-Instruct
    python scripts/validate_models.py --list-all
    python scripts/validate_models.py --gpu-memory 80
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qwen3_distill.utils.models import (
    validate_model_pair,
    list_compatible_pairs,
    get_model_spec,
    MODEL_REGISTRY,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate student-teacher model pairs for on-policy distillation"
    )

    parser.add_argument(
        "--student",
        type=str,
        help="Student model name (e.g., Qwen/Qwen3-4B-Instruct)"
    )

    parser.add_argument(
        "--teacher",
        type=str,
        help="Teacher model name (e.g., Qwen/Qwen3-32B-Instruct)"
    )

    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=128.0,
        help="Total GPU memory in GB (default: 128)"
    )

    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Don't use FP8 quantization for teacher (default: use FP8)"
    )

    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all compatible model pairs for given GPU memory"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all models in registry"
    )

    args = parser.parse_args()

    # List all models in registry
    if args.list_models:
        print("\nAvailable Models in Registry:")
        print("=" * 80)

        # Group by architecture
        by_arch = {}
        for name, spec in MODEL_REGISTRY.items():
            if spec.architecture not in by_arch:
                by_arch[spec.architecture] = []
            by_arch[spec.architecture].append((name, spec))

        for arch in sorted(by_arch.keys()):
            print(f"\n{arch.upper()} Models:")
            print("-" * 80)
            models = sorted(by_arch[arch], key=lambda x: x[1].size_params)
            for name, spec in models:
                print(f"  {name:45s} {spec.size_params:6.1f}B  "
                      f"FP16:{spec.memory_fp16_gb:5.0f}GB  FP8:{spec.memory_fp8_gb:5.0f}GB")

        print("\n" + "=" * 80)
        return

    # List compatible pairs
    if args.list_all:
        list_compatible_pairs(
            total_gpu_memory_gb=args.gpu_memory,
            use_fp8_teacher=not args.no_fp8
        )
        return

    # Validate specific pair
    if not args.student or not args.teacher:
        print("Error: Both --student and --teacher are required for validation")
        print("Use --list-all to see compatible pairs or --help for usage")
        sys.exit(1)

    print(f"\nValidating Model Pair:")
    print("=" * 80)
    print(f"Student:      {args.student}")
    print(f"Teacher:      {args.teacher}")
    print(f"GPU Memory:   {args.gpu_memory}GB")
    print(f"Teacher FP8:  {not args.no_fp8}")
    print("=" * 80)

    is_valid, message = validate_model_pair(
        args.student,
        args.teacher,
        args.gpu_memory,
        not args.no_fp8
    )

    print(f"\n{message}\n")

    if is_valid:
        print("✓ This model pair is compatible!")
        print("\nRecommended config file:")

        student_spec = get_model_spec(args.student)
        teacher_spec = get_model_spec(args.teacher)

        config_name = f"{args.student.split('/')[-1].lower()}-{args.teacher.split('/')[-1].lower()}.yaml"
        print(f"  configs/{config_name}")

        print("\nTo use this pair:")
        print(f"  python -m qwen3_distill.pipeline \\")
        print(f"    --config configs/your-config.yaml \\")
        print(f"    --prompts data/prompts.txt \\")
        print(f"    --num-epochs 10")

        sys.exit(0)
    else:
        print("✗ This model pair is NOT compatible!")
        print("\nSuggestions:")
        print("1. Try smaller models")
        print("2. Use FP8 quantization (remove --no-fp8)")
        print("3. Use --list-all to see compatible pairs")
        sys.exit(1)


if __name__ == "__main__":
    main()
