#!/usr/bin/env python3
"""
Simplified TRL-based training script for on-policy distillation.

This script uses TRL's PPOTrainer with local model loading, eliminating
the need for vLLM student container. Only vLLM teacher is needed for scoring.

Architecture:
1. Load student model locally with Unsloth + LoRA (4-bit quantized)
2. Load teacher scorer (vLLM API for FP8 teacher)
3. Train with TRL PPOTrainer:
   - Generate locally via trainer.generate()
   - Score with teacher via API
   - Optimize with PPO using Reverse KL rewards
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwen3_distill.generation.local_generator import LocalStudentModel, LocalModelConfig
from qwen3_distill.scoring.scorer import TeacherScorer, ScoringConfig
from qwen3_distill.training.trl_trainer import TRLDistillationTrainer
from qwen3_distill.utils.gpu import log_gpu_memory, optimize_memory


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from file."""
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="TRL-based On-Policy Distillation Training"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/prompts.txt",
        help="Path to prompt dataset file",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Student model name",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Teacher model name",
    )
    parser.add_argument(
        "--teacher-api",
        type=str,
        default="http://vllm-teacher:30000",
        help="Teacher vLLM API URL",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    print("="*80)
    print("TRL-BASED ON-POLICY DISTILLATION TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Student Model: {args.student_model}")
    print(f"  Teacher Model: {args.teacher_model}")
    print(f"  Teacher API: {args.teacher_api}")
    print(f"  Num Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max Seq Length: {args.max_seq_length}")
    print(f"  Output Dir: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print(f"{'='*80}")
    prompts = load_prompts(args.prompts)

    # Initialize components
    print(f"\n{'='*80}")
    print("INITIALIZING COMPONENTS")
    print(f"{'='*80}")

    log_gpu_memory("Initial")

    # 1. Initialize teacher scorer (FP8, ~45GB for Qwen3-32B)
    print("\n[1/3] Initializing teacher scorer...")
    scorer_config = ScoringConfig(
        model_name=args.teacher_model,
        api_url=args.teacher_api,
        quantization="fp8",
        max_batch_size=args.batch_size,
    )
    teacher_scorer = TeacherScorer(scorer_config)
    log_gpu_memory("After teacher scorer")

    # 2. Load student model locally with Unsloth + LoRA
    print("\n[2/3] Loading student model locally...")
    local_model_config = LocalModelConfig(
        model_name=args.student_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )
    local_student = LocalStudentModel(local_model_config)
    local_student.print_trainable_parameters()
    log_gpu_memory("After student model")

    # 3. Initialize TRL trainer
    print("\n[3/3] Initializing TRL PPOTrainer...")
    training_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }

    trainer = TRLDistillationTrainer(
        student_model=local_student.get_model(),
        teacher_scorer=teacher_scorer,
        tokenizer=local_student.get_tokenizer(),
        config=training_config,
    )
    log_gpu_memory("After TRL trainer")

    print("\n✅ All components initialized successfully!")
    print("\nMemory Summary:")
    from qwen3_distill.utils.gpu import get_gpu_memory_stats
    stats = get_gpu_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Start training
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING ({args.num_epochs} EPOCHS)")
    print(f"{'='*80}")

    try:
        trainer.train(
            prompt_dataset=prompts,
            num_epochs=args.num_epochs,
            output_dir=str(output_dir),
        )

        print(f"\n{'='*80}")
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(str(output_dir / "interrupted"))

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        print("\nSaving emergency checkpoint...")
        try:
            trainer.save_checkpoint(str(output_dir / "emergency"))
        except:
            pass

        raise

    finally:
        # Cleanup
        print("\nCleaning up...")
        optimize_memory()
        log_gpu_memory("Final")


if __name__ == "__main__":
    main()
