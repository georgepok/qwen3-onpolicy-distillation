#!/usr/bin/env python3
"""
Main pipeline orchestration for Qwen3-4B on-policy distillation.

This script coordinates all components:
1. Student generation (vLLM)
2. Teacher scoring (SGLang with FP8)
3. Training (unsloth)
4. Experience replay buffer

Optimized for Nvidia GB10 DGX systems with 128GB GPU RAM.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

from qwen3_distill.generation.generator import StudentGenerator, GenerationConfig
from qwen3_distill.generation.local_generator import LocalStudentModel, LocalModelConfig
from qwen3_distill.scoring.scorer import TeacherScorer, ScoringConfig
# from qwen3_distill.training.trainer import DistillationTrainer, TrainingConfig  # DEPRECATED
# from qwen3_distill.training.trl_trainer import TRLDistillationTrainer  # DEPRECATED - PPO incompatible with Unsloth
# from qwen3_distill.training.sft_distill_trainer import SFTDistillationTrainer  # DEPRECATED - Not true on-policy distillation
from qwen3_distill.training.reverse_kl_trainer import ReverseKLDistillationTrainer
from qwen3_distill.utils.config import load_config, merge_configs, DEFAULT_CONFIG
from qwen3_distill.utils.gpu import (
    get_gpu_memory_stats,
    log_gpu_memory,
    check_memory_available,
    optimize_memory,
)
from qwen3_distill.utils.models import validate_model_pair, get_model_spec


class OnPolicyDistillationPipeline:
    """
    Orchestrates the complete on-policy distillation pipeline.
    
    This class manages:
    1. Initialization of all components
    2. Coordination of generation -> scoring -> training loop
    3. Memory management across concurrent models
    4. Checkpointing and logging
    5. Graceful shutdown and cleanup
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with all component configs
            
        TODO for Claude Code:
            1. Parse configuration for each component
            2. Initialize generator, scorer, trainer, buffer
            3. Verify GPU memory availability
            4. Set up logging and checkpointing
            5. Validate concurrent model loading feasible
        """
        self.config = config
        self.generator: Optional[StudentGenerator] = None
        self.scorer: Optional[TeacherScorer] = None
        self.trainer: Optional[TRLDistillationTrainer] = None
        
        self._init_components()
        
    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        print("Initializing on-policy distillation pipeline...")

        # Validate model pair before initialization
        student_model = self.config["generation"]["model_name"]
        teacher_model = self.config["scoring"]["model_name"]
        use_fp8 = self.config["scoring"].get("quantization", "fp8") == "fp8"

        # Get total GPU memory
        import torch
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            total_memory_gb = 128  # Default assumption

        print(f"\nValidating model pair:")
        print(f"  Student: {student_model}")
        print(f"  Teacher: {teacher_model}")
        print(f"  GPU Memory: {total_memory_gb:.1f}GB")
        print(f"  Teacher Quantization: {'FP8' if use_fp8 else 'FP16'}")

        is_valid, message = validate_model_pair(
            student_model,
            teacher_model,
            total_memory_gb,
            use_fp8
        )

        print(f"\n{message}\n")

        if not is_valid:
            raise ValueError(
                f"Invalid model pair! {message}\n\n"
                "To fix:\n"
                "1. Use the model validation utility to find compatible pairs:\n"
                "   python -c 'from qwen3_distill.utils.models import list_compatible_pairs; "
                "list_compatible_pairs()'\n"
                "2. Or choose a different config file from configs/\n"
                "3. Or add your custom models to utils/models.py MODEL_REGISTRY"
            )

        log_gpu_memory("Initial")

        # 1. Initialize teacher scorer (FP8, ~45GB for Qwen3-32B)
        print("\n[1/4] Initializing teacher scorer...")
        scorer_config = ScoringConfig(**self.config["scoring"])
        self.scorer = TeacherScorer(scorer_config)
        log_gpu_memory("After teacher scorer")

        # 2. Initialize student model locally for TRL training
        print("\n[2/4] Loading student model locally...")
        local_model_config = LocalModelConfig(
            model_name=self.config["generation"]["model_name"],
            max_seq_length=self.config["generation"].get("max_tokens", 2048),
            load_in_4bit=True,
            lora_r=self.config["training"].get("lora_r", 16),
            lora_alpha=self.config["training"].get("lora_alpha", 32),
            lora_dropout=self.config["training"].get("lora_dropout", 0.0),
        )
        self.local_model = LocalStudentModel(local_model_config)
        self.local_model.print_trainable_parameters()
        log_gpu_memory("After student model loaded")

        # 3. Initialize Reverse KL trainer with local model
        print("\n[3/4] Initializing Reverse KL distillation trainer...")
        self.trainer = ReverseKLDistillationTrainer(
            student_model=self.local_model.get_model(),
            teacher_scorer=self.scorer,
            tokenizer=self.local_model.get_tokenizer(),
            config=self.config["training"],
        )
        log_gpu_memory("After Reverse KL trainer")

        print("\nPipeline initialization complete!")
        print(f"\nMemory summary:")
        stats = get_gpu_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def generate_trajectories(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate trajectories using student model.

        Args:
            prompts: List of prompts for generation

        Returns:
            List of generated trajectories
        """
        # Use local student model for generation
        # Support multiple rollouts per prompt for variance reduction
        samples_per_prompt = self.config.get("training", {}).get("num_rollouts_per_prompt", 1)
        expected_trajectories = len(prompts) * samples_per_prompt
        print(f"Generating {len(prompts)} prompts × {samples_per_prompt} rollouts = {expected_trajectories} trajectories...")
        log_gpu_memory("Before generation")

        trajectories = []
        tokenizer = self.local_model.get_tokenizer()
        model = self.local_model.get_model()

        # Generate completions using local model
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

                # Decode the generated text
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

                # Create trajectory dict
                trajectory = {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "full_text": full_text,
                    "tokens": outputs[0].tolist(),
                }
                trajectories.append(trajectory)

        log_gpu_memory("After generation")
        print(f"Generated {len(trajectories)} trajectories (expected {expected_trajectories})")
        if samples_per_prompt > 1:
            print(f"  → {len(prompts)} unique prompts with {samples_per_prompt} rollouts each")

        return trajectories
    
    def score_trajectories(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score trajectories with teacher model.
        
        Args:
            trajectories: Trajectories to score
            
        Returns:
            Scored trajectories with teacher logprobs
            
        TODO for Claude Code:
            1. Use teacher scorer to compute logprobs
            2. Calculate KL divergence
            3. Monitor memory during scoring
            4. Handle scoring failures
        """
        print(f"Scoring {len(trajectories)} trajectories...")
        log_gpu_memory("Before scoring")
        
        scored = self.scorer.score_trajectories(trajectories)
        
        log_gpu_memory("After scoring")
        print(f"Scored {len(scored)} trajectories")
        
        return scored
    
    def train_on_trajectories(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Train student on scored trajectories.
        
        Args:
            trajectories: Scored trajectories
            
        Returns:
            Training metrics
            
        TODO for Claude Code:
            1. Compute advantages from trajectories
            2. Add to replay buffer
            3. Sample training batches
            4. Execute training steps
            5. Return aggregated metrics
        """
        print(f"Training on {len(trajectories)} trajectories...")
        log_gpu_memory("Before training")
        
        # TRL trainer handles trajectories internally
        
        # Train for one epoch
        metrics = self.trainer.train_epoch(trajectories)
        
        log_gpu_memory("After training")
        print(f"Training metrics: {metrics}")
        
        return metrics
    
    def run_epoch(
        self,
        prompts: List[str],
        epoch: int
    ) -> Dict[str, Any]:
        """
        Run one complete epoch of the on-policy pipeline.
        
        Args:
            prompts: Prompts for trajectory generation
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
            
        TODO for Claude Code:
            1. Generate fresh trajectories (on-policy)
            2. Score with teacher
            3. Train on scored trajectories
            4. Aggregate metrics
            5. Save checkpoint if needed
        """
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}")
        print(f"{'='*80}\n")
        
        # Generate
        trajectories = self.generate_trajectories(prompts)
        
        # Score
        scored_trajectories = self.score_trajectories(trajectories)
        
        # Train
        train_metrics = self.train_on_trajectories(scored_trajectories)

        # Export LoRA adapters for checkpointing (optional - policy sync happens via local model)
        if epoch % 5 == 0:  # Save every 5 epochs
            print(f"\nExporting LoRA adapters...")
            lora_path = f"./checkpoints/epoch_{epoch}_lora"
            self.trainer.export_lora_adapters(lora_path)
            print(f"  LoRA adapters saved to: {lora_path}")

        # Aggregate metrics
        epoch_metrics = {
            "epoch": epoch,
            "num_trajectories": len(trajectories),
            **train_metrics,
        }

        return epoch_metrics
    
    def run(
        self,
        prompt_dataset: List[str],
        num_epochs: int = 10
    ) -> None:
        """
        Run the complete on-policy distillation pipeline.
        
        Args:
            prompt_dataset: Dataset of prompts for generation
            num_epochs: Number of training epochs
            
        TODO for Claude Code:
            1. Implement main training loop
            2. Run epochs with fresh trajectory generation
            3. Log metrics to wandb/tensorboard
            4. Save checkpoints periodically
            5. Handle interruptions gracefully
            6. Cleanup on completion
        """
        print(f"Starting on-policy distillation for {num_epochs} epochs")
        print(f"Prompt dataset size: {len(prompt_dataset)}")
        
        try:
            for epoch in range(num_epochs):
                # Sample prompts for this epoch
                epoch_prompts = self._sample_prompts(prompt_dataset, epoch)
                
                # Run epoch
                metrics = self.run_epoch(epoch_prompts, epoch)
                
                # Save checkpoint
                if epoch % self.config["pipeline"]["save_interval"] == 0:
                    self._save_checkpoint(epoch, metrics)
                
                # Optimize memory between epochs
                optimize_memory()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self._save_checkpoint(-1, {})
        
        finally:
            self.shutdown()
    
    def _sample_prompts(
        self,
        prompt_dataset: List[str],
        epoch: int
    ) -> List[str]:
        """
        Sample prompts for an epoch.
        
        Args:
            prompt_dataset: Full prompt dataset
            epoch: Current epoch
            
        Returns:
            Sampled prompts
            
        TODO for Claude Code:
            1. Implement sampling strategy
            2. Consider shuffling with seed for reproducibility
            3. Handle dataset cycling if needed
        """
        # Simple implementation: use all prompts each epoch
        import random
        prompts = prompt_dataset.copy()
        random.seed(epoch)  # Reproducible shuffle
        random.shuffle(prompts)
        return prompts
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Save checkpoint with current state.

        Args:
            epoch: Current epoch number
            metrics: Current metrics
        """
        checkpoint_dir = Path(self.config["pipeline"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        print(f"Saving checkpoint to {checkpoint_path}")

        # Save trainer checkpoint
        self.trainer.save_checkpoint(str(checkpoint_path), metrics)

        # TRL trainer handles checkpointing

        print(f"  Checkpoint saved successfully")
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown all components.
        
        TODO for Claude Code:
            1. Shutdown generator
            2. Shutdown scorer
            3. Save final checkpoint
            4. Clear GPU memory
        """
        print("\nShutting down pipeline...")
        
        if self.generator:
            self.generator.shutdown()
        
        if self.scorer:
            self.scorer.shutdown()
        
        optimize_memory()
        log_gpu_memory("After shutdown")


def main():
    """
    Main entry point for the pipeline.
    
    TODO for Claude Code:
        1. Parse command-line arguments
        2. Load configuration
        3. Load prompt dataset
        4. Initialize and run pipeline
        5. Handle errors and cleanup
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-4B On-Policy Distillation Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompt dataset (text file, one per line)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        config = merge_configs(DEFAULT_CONFIG, config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = DEFAULT_CONFIG
    
    # Load prompts
    with open(args.prompts, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This pipeline requires GPU.")
    
    log_gpu_memory("System")
    
    # Initialize and run pipeline
    pipeline = OnPolicyDistillationPipeline(config)
    pipeline.run(prompts, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
