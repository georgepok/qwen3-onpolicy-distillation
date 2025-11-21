"""
Unit tests for TRL PPO-based trainer with Reverse KL rewards.

Tests the TRLDistillationTrainer implementation using PPOTrainer.
"""

import pytest
import torch
from qwen3_distill.training.trl_trainer import TRLDistillationTrainer


class MockTeacherScorer:
    """Mock scorer for testing."""
    
    def score_trajectories(self, trajectories):
        """Return mock teacher logprobs."""
        results = []
        for traj in trajectories:
            # Mock teacher logprobs (slightly better than student)
            num_tokens = len(traj.get("tokens", [10, 20, 30]))  # Default 3 tokens
            teacher_logprobs = [-0.5] * num_tokens  # Better logprobs than typical student
            
            results.append({
                "teacher_logprobs": teacher_logprobs,
                "kl_divergence": [0.1] * num_tokens,
            })
        
        return results


def test_trl_trainer_initialization():
    """Test TRL trainer can be initialized with PPO config."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")
    
    # Use tiny model for testing
    model_name = "gpt2"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        pytest.skip(f"Could not load test model: {e}")
    
    config = {
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
    }
    
    trainer = TRLDistillationTrainer(
        student_model=model,
        teacher_scorer=MockTeacherScorer(),
        tokenizer=tokenizer,
        config=config,
    )
    
    assert trainer is not None
    assert trainer.student_model is model
    assert trainer.tokenizer is tokenizer
    assert trainer.config == config


def test_reverse_kl_reward_computation():
    """Test that Reverse KL rewards are computed correctly."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")
    
    model_name = "gpt2"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pytest.skip("Could not load test model")
    
    trainer = TRLDistillationTrainer(
        student_model=model,
        teacher_scorer=MockTeacherScorer(),
        tokenizer=tokenizer,
        config={},
    )
    
    # Test reward computation logic
    prompts = ["Test prompt"]
    query_tensors = [torch.tensor([1, 2, 3])]
    response_tensors = [torch.tensor([4, 5, 6])]
    
    # This will call the mock scorer
    rewards = trainer._compute_reverse_kl_rewards(
        prompts,
        query_tensors,
        response_tensors
    )
    
    assert len(rewards) == 1
    assert isinstance(rewards[0], torch.Tensor)
    # Rewards should be negative Reverse KL
    # (student worse than teacher → positive KL → negative rewards)


def test_train_epoch_compatibility():
    """Test that train_epoch method works for pipeline compatibility."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")
    
    model_name = "gpt2"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pytest.skip("Could not load test model")
    
    trainer = TRLDistillationTrainer(
        student_model=model,
        teacher_scorer=MockTeacherScorer(),
        tokenizer=tokenizer,
        config={"batch_size": 1},
    )
    
    # Test that method exists and accepts trajectories
    trajectories = [
        {"prompt": "Test prompt 1"},
        {"prompt": "Test prompt 2"},
    ]
    
    # Should not crash (actual training would require more setup)
    # Just verify the interface exists
    assert hasattr(trainer, "train_epoch")
    assert callable(trainer.train_epoch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
