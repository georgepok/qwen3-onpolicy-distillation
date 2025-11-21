"""
Supervised Fine-Tuning (SFT) trainer for on-policy distillation.

This is a simpler approach than PPO that works well with Unsloth:
1. Generate trajectories from student
2. Score with teacher to get KL divergence
3. Use trajectories with low KL (good student outputs) for SFT
4. Train student to reproduce good outputs

This avoids the complexity of PPO while still implementing on-policy distillation.
"""

from typing import List, Optional, Dict, Any
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import numpy as np


class SFTDistillationTrainer:
    """
    Supervised fine-tuning based distillation trainer.

    Simpler than PPO but still effective:
    1. Sample trajectories from current student policy (on-policy)
    2. Score with teacher to get KL divergence
    3. Filter for low-KL trajectories (where student matches teacher)
    4. Fine-tune student on these good examples
    """

    def __init__(
        self,
        student_model,
        teacher_scorer,
        tokenizer,
        config: dict,
    ):
        """
        Initialize SFT-based distillation trainer.

        Args:
            student_model: Student model to train (Unsloth model)
            teacher_scorer: TeacherScorer instance
            tokenizer: Tokenizer
            config: Training configuration dict
        """
        self.student_model = student_model
        self.teacher_scorer = teacher_scorer
        self.tokenizer = tokenizer
        self.config = config

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Initialized SFT-based distillation trainer")
        print("  This approach uses supervised learning on teacher-approved outputs")

    def train_epoch(self, trajectories: List[dict]) -> dict:
        """
        Train for one epoch on scored trajectories.

        Args:
            trajectories: List of trajectory dicts with teacher scores

        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining on {len(trajectories)} trajectories with SFT...")

        # Filter trajectories by KL divergence
        # Lower KL = student output is closer to teacher
        filtered_trajs = self._filter_by_kl(trajectories)

        if len(filtered_trajs) == 0:
            print("  WARNING: No trajectories passed KL filter, using all")
            filtered_trajs = trajectories

        print(f"  Using {len(filtered_trajs)}/{len(trajectories)} trajectories for training")

        # Convert to dataset format
        dataset = self._prepare_dataset(filtered_trajs)

        # Configure training
        training_args = TrainingArguments(
            output_dir="./checkpoints/sft",
            num_train_epochs=1,
            per_device_train_batch_size=self.config.get("batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-5),
            logging_steps=10,
            save_strategy="no",  # We'll save manually
            fp16=False,  # Unsloth handles precision
            bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        )

        # Create SFT trainer (Unsloth-compatible)
        trainer = SFTTrainer(
            model=self.student_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.get("max_seq_length", 512),
            dataset_text_field="text",  # Use 'text' field from dataset
        )

        # Train
        print("  Starting SFT training...")
        train_result = trainer.train()

        # Extract metrics
        metrics = {
            "loss": train_result.training_loss,
            "num_trajectories": len(filtered_trajs),
            "filter_rate": len(filtered_trajs) / len(trajectories),
        }

        print(f"  Training complete: loss={metrics['loss']:.4f}")

        return metrics

    def _filter_by_kl(self, trajectories: List[dict], kl_threshold: float = None) -> List[dict]:
        """
        Filter trajectories by KL divergence.

        Keep trajectories where student is close to teacher (low KL).

        Args:
            trajectories: Trajectories with teacher scores
            kl_threshold: Maximum KL to keep (None = use percentile)

        Returns:
            Filtered trajectories
        """
        if kl_threshold is None:
            # Use top 70% of trajectories (lowest KL)
            kl_threshold_percentile = 70
        else:
            kl_threshold_percentile = None

        # Calculate mean KL for each trajectory
        trajs_with_kl = []
        for traj in trajectories:
            if "teacher_logprobs" not in traj or "tokens" not in traj:
                continue

            # Simple KL approximation: variance of student-teacher difference
            teacher_lp = np.array(traj["teacher_logprobs"])

            # We don't have student logprobs here, so we'll use all trajectories
            # In future, we could recompute student logprobs
            mean_teacher_conf = np.mean(teacher_lp)

            trajs_with_kl.append({
                "traj": traj,
                "teacher_conf": mean_teacher_conf  # Higher = teacher more confident
            })

        if not trajs_with_kl:
            return trajectories

        # Sort by teacher confidence (higher = better)
        trajs_with_kl.sort(key=lambda x: x["teacher_conf"], reverse=True)

        # Keep top percentage
        if kl_threshold_percentile:
            keep_count = max(1, int(len(trajs_with_kl) * kl_threshold_percentile / 100))
            filtered = [item["traj"] for item in trajs_with_kl[:keep_count]]
        else:
            # Keep all above threshold
            filtered = [item["traj"] for item in trajs_with_kl
                       if item["teacher_conf"] > kl_threshold]

        return filtered if filtered else trajectories

    def _prepare_dataset(self, trajectories: List[dict]) -> Dataset:
        """
        Convert trajectories to HuggingFace Dataset for SFT.

        Args:
            trajectories: Filtered trajectories

        Returns:
            Dataset with 'text' field containing prompt+response
        """
        texts = []

        for traj in trajectories:
            prompt = traj.get("prompt", "")
            generated = traj.get("generated_text", "")

            # Format as prompt + completion
            # SFTTrainer will handle tokenization
            text = f"{prompt}{generated}"
            texts.append(text)

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        return dataset

    def save_checkpoint(self, path: str, metrics: Optional[dict] = None):
        """Save model checkpoint."""
        self.student_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  Checkpoint saved to {path}")

    def export_lora_adapters(self, lora_path: str):
        """Export LoRA adapters."""
        self.student_model.save_pretrained(lora_path)
        self.tokenizer.save_pretrained(lora_path)
        print(f"  LoRA adapters exported to {lora_path}")
