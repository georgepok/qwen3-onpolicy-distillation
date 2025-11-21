"""
Custom Reverse KL on-policy distillation trainer.

Implements the exact algorithm from:
https://thinkingmachines.ai/blog/on-policy-distillation/

WITHOUT using TRL's PPOTrainer (which is incompatible with Unsloth).
Instead, directly implements the core algorithm:
1. Sample trajectories from student
2. Get teacher logprobs for same tokens
3. Compute reverse KL: student_logprob - teacher_logprob
4. Use -reverse_KL as per-token rewards
5. Apply policy gradient with importance sampling
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import numpy as np


class ReverseKLDistillationTrainer:
    """
    On-policy distillation using Reverse KL divergence as rewards.

    Core algorithm:
    - Generate trajectories from current student policy (on-policy)
    - Score each token with teacher to get teacher_logprobs
    - Compute per-token reward = -(student_logprob - teacher_logprob)
    - Train with policy gradient to maximize rewards
    """

    def __init__(
        self,
        student_model,
        teacher_scorer,
        tokenizer,
        config: dict,
    ):
        """
        Initialize Reverse KL trainer.

        Args:
            student_model: Unsloth model to train
            teacher_scorer: TeacherScorer for getting teacher logprobs
            tokenizer: Tokenizer
            config: Training configuration
        """
        self.student_model = student_model
        self.teacher_scorer = teacher_scorer
        self.tokenizer = tokenizer
        self.config = config

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize Adam optimizer
        lr = self.config.get("learning_rate", 1e-5)
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        print("Initialized Reverse KL distillation trainer")
        print("  Algorithm: On-policy with per-token reverse KL rewards")
        print(f"  Optimizer: Adam (lr={lr})")

    def train_epoch(self, trajectories: List[dict]) -> dict:
        """
        Train for one epoch on scored trajectories.

        Args:
            trajectories: List of dicts with 'prompt', 'generated_text', 'tokens', 'teacher_logprobs'

        Returns:
            Training metrics
        """
        print(f"\nTraining on {len(trajectories)} trajectories...")

        # Compute student logprobs and reverse KL rewards
        all_rewards = []
        all_losses = []

        for traj in trajectories:
            prompt = traj["prompt"]
            generated_text = traj["generated_text"]
            teacher_logprobs = torch.tensor(traj["teacher_logprobs"]).to(self.student_model.device)

            # Get student logprobs WITH gradients for training
            student_logprobs = self._compute_student_logprobs(
                prompt,
                generated_text,
                requires_grad=True
            )

            if student_logprobs is None or len(student_logprobs) == 0:
                continue

            # Match lengths (teacher logprobs may be shorter due to truncation)
            min_len = min(len(student_logprobs), len(teacher_logprobs))
            student_lp = student_logprobs[:min_len]
            teacher_lp = teacher_logprobs[:min_len]

            # Reverse KL divergence: D_KL(student || teacher) = student_log - teacher_log
            reverse_kl = student_lp - teacher_lp

            # Reward is negative reverse KL (we want to minimize divergence)
            rewards = -reverse_kl

            all_rewards.extend(rewards.detach().cpu().tolist())

            # Compute policy gradient loss
            # Loss = -mean(reward * log_prob) for policy gradient
            loss = -(rewards.detach() * student_lp).mean()  # Detach rewards to avoid backprop through them

            all_losses.append(loss.item())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                max_norm=1.0
            )

            # Optimizer step with Adam
            self.optimizer.step()

        # Metrics
        metrics = {
            "loss": np.mean(all_losses) if all_losses else 0.0,
            "mean_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "mean_reverse_kl": -np.mean(all_rewards) if all_rewards else 0.0,
            "num_trajectories": len(trajectories),
        }

        print(f"  Training complete:")
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    Mean Reward: {metrics['mean_reward']:.4f}")
        print(f"    Mean Reverse KL: {metrics['mean_reverse_kl']:.4f}")

        return metrics

    def _compute_student_logprobs(
        self,
        prompt: str,
        generated_text: str,
        requires_grad: bool = False
    ):
        """
        Compute student model's logprobs for generated tokens.

        Args:
            prompt: Input prompt
            generated_text: Generated completion
            requires_grad: Whether to keep gradients (True for training, False for evaluation)

        Returns:
            Tensor of logprobs for each generated token (with or without gradients)
        """
        # Tokenize
        full_text = prompt + generated_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(
            self.student_model.device
        )

        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.student_model.device
        )

        prompt_len = len(prompt_inputs.input_ids[0])

        # Forward pass to get logits
        if requires_grad:
            outputs = self.student_model(**inputs)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        else:
            with torch.no_grad():
                outputs = self.student_model(**inputs)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Extract logprobs for generated tokens
        generated_token_ids = inputs.input_ids[0][prompt_len:]

        logprobs_list = []
        for i, token_id in enumerate(generated_token_ids):
            # Position in full sequence where this token is predicted
            pos = prompt_len + i - 1

            if pos >= 0 and pos < len(log_probs):
                token_logprob = log_probs[pos, token_id.item()]
                logprobs_list.append(token_logprob)

        if not logprobs_list:
            return None

        if requires_grad:
            return torch.stack(logprobs_list)
        else:
            return torch.tensor([lp.item() for lp in logprobs_list])

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
