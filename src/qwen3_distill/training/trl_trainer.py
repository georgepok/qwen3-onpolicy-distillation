"""
TRL PPO-based trainer for Reverse KL on-policy distillation.

This module implements the correct approach from the Thinking Machines article:
https://thinkingmachines.ai/blog/on-policy-distillation/

Uses PPOTrainer with custom per-token Reverse KL rewards instead of DPO's
preference-based learning.
"""

from typing import List, Optional, Dict, Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
import numpy as np


class TRLDistillationTrainer:
    """
    PPO-based trainer for Reverse KL on-policy distillation.
    
    Implements the algorithm from Thinking Machines' article:
    1. Sample trajectories from student
    2. Compute teacher logprobs on sampled tokens
    3. Calculate Reverse KL as reward: student_logprob - teacher_logprob
    4. Use negative Reverse KL as advantages (to minimize KL)
    5. Train with policy gradient (PPO)
    """
    
    def __init__(
        self,
        student_model,
        teacher_scorer,
        tokenizer,
        config: dict,
    ):
        """
        Initialize PPO-based distillation trainer.
        
        Args:
            student_model: Student model to train
            teacher_scorer: TeacherScorer instance for computing teacher logprobs
            tokenizer: Tokenizer for processing
            config: Training configuration dict
        """
        self.student_model = student_model
        self.teacher_scorer = teacher_scorer
        self.tokenizer = tokenizer
        self.config = config
        self.trainer: Optional[PPOTrainer] = None
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(
        self,
        prompt_dataset: List[str],
        num_epochs: int = 1,
        output_dir: str = "./checkpoints",
    ):
        """
        Train student model using PPO with Reverse KL rewards.
        
        Args:
            prompt_dataset: List of prompt strings
            num_epochs: Number of training epochs
            output_dir: Directory for checkpoints
        """
        # Configure PPO - simplified for Unsloth compatibility
        ppo_config = PPOConfig(
            # Training hyperparameters
            learning_rate=self.config.get("learning_rate", 2e-5),
            batch_size=self.config.get("batch_size", 4),
            mini_batch_size=1,
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
        )
        
        # Create PPO trainer
        self.trainer = PPOTrainer(
            config=ppo_config,
            model=self.student_model,
            tokenizer=self.tokenizer,
            ref_model=None,  # No reference model needed
        )
        
        print(f"Starting PPO training with Reverse KL rewards for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            for batch_idx, batch_prompts in enumerate(self._batch_prompts(prompt_dataset)):
                # Tokenize prompts
                query_tensors = [
                    self.tokenizer.encode(prompt, return_tensors="pt")[0]
                    for prompt in batch_prompts
                ]
                
                # Generate responses from student
                response_tensors = self.trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **ppo_config.generation_kwargs
                )
                
                # Compute Reverse KL rewards
                rewards = self._compute_reverse_kl_rewards(
                    batch_prompts,
                    query_tensors,
                    response_tensors
                )
                
                # PPO step with our rewards
                stats = self.trainer.step(query_tensors, response_tensors, rewards)
                
                # Log stats
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: {stats}")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(f"{output_dir}/epoch_{epoch}")
        
        print("\nTraining complete!")
    
    def _batch_prompts(self, prompts: List[str]) -> List[List[str]]:
        """Batch prompts for processing."""
        batch_size = self.config.get("batch_size", 4)
        for i in range(0, len(prompts), batch_size):
            yield prompts[i:i + batch_size]
    
    def _compute_reverse_kl_rewards(
        self,
        prompts: List[str],
        query_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute per-token Reverse KL as rewards.
        
        Reverse KL = student_logprob - teacher_logprob
        Reward = -Reverse_KL (we want to minimize KL)
        
        Args:
            prompts: Original prompts
            query_tensors: Tokenized prompts
            response_tensors: Generated responses
        
        Returns:
            List of reward tensors (one per response)
        """
        rewards_list = []
        
        # Decode responses
        responses = self.tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=False
        )
        
        # Get student logprobs
        student_token_logprobs = self._get_student_logprobs(
            query_tensors,
            response_tensors
        )
        
        # Create trajectories for teacher scoring
        trajectories = [
            {
                "prompt": prompt,
                "generated_text": response,
                "tokens": response_tensor.tolist(),
            }
            for prompt, response, response_tensor in zip(prompts, responses, response_tensors)
        ]
        
        # Get teacher logprobs via scorer
        print(f"  Scoring {len(trajectories)} trajectories with teacher...")
        scored = self.teacher_scorer.score_trajectories(trajectories)
        
        # Compute Reverse KL rewards for each trajectory
        for i, score in enumerate(scored):
            teacher_logprobs = torch.tensor(
                score["teacher_logprobs"],
                dtype=torch.float32
            )
            student_lp = student_token_logprobs[i]
            
            # Align lengths (teacher might return different length)
            min_len = min(len(student_lp), len(teacher_logprobs))
            student_lp = student_lp[:min_len]
            teacher_lp = teacher_logprobs[:min_len]
            
            # Reverse KL: student_log - teacher_log
            reverse_kl = student_lp - teacher_lp
            
            # Negative KL as reward (we want to minimize KL)
            rewards = -reverse_kl
            
            rewards_list.append(rewards)
        
        return rewards_list
    
    def _get_student_logprobs(
        self,
        query_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Extract student's log probabilities for generated tokens.
        
        Args:
            query_tensors: Prompt tokens
            response_tensors: Generated response tokens
        
        Returns:
            List of log probability tensors
        """
        logprobs_list = []
        
        with torch.no_grad():
            for query, response in zip(query_tensors, response_tensors):
                # Concatenate query and response
                full_sequence = torch.cat([query, response])
                
                # Get model outputs
                outputs = self.student_model(
                    full_sequence.unsqueeze(0).to(self.student_model.device)
                )
                
                # Get log probabilities
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Extract logprobs for response tokens only
                # We want logprobs at positions where we predict response tokens
                response_start_idx = len(query)
                response_logprobs = []
                
                for i, token_id in enumerate(response):
                    # Position in full sequence where this token is predicted
                    pos = response_start_idx + i - 1
                    if pos >= 0 and pos < len(log_probs):
                        token_logprob = log_probs[pos, token_id.item()]
                        response_logprobs.append(token_logprob.item())
                
                logprobs_list.append(torch.tensor(response_logprobs, dtype=torch.float32))
        
        return logprobs_list
    
    def train_epoch(self, trajectories: List[dict]) -> dict:
        """
        Compatibility method for existing pipeline.
        
        Extracts prompts from trajectories and trains for 1 epoch.
        
        Args:
            trajectories: List of trajectory dicts with 'prompt' field
        
        Returns:
            Dictionary with training metrics
        """
        # Extract prompts from trajectories
        prompts = [traj["prompt"] for traj in trajectories]
        
        # Train for 1 epoch
        self.train(prompts, num_epochs=1)
        
        # Return metrics
        return {"status": "trained_via_ppo_reverse_kl"}
    
    def save_checkpoint(self, path: str, metrics: Optional[dict] = None):
        """Save model checkpoint."""
        if self.trainer:
            self.trainer.save_pretrained(path)
        else:
            # Fallback if trainer not initialized
            self.student_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        
        print(f"  Checkpoint saved to {path}")
    
    def export_lora_adapters(self, lora_path: str):
        """Export LoRA adapters for vLLM loading."""
        self.student_model.save_pretrained(lora_path)
        self.tokenizer.save_pretrained(lora_path)
        print(f"  LoRA adapters exported to {lora_path}")
