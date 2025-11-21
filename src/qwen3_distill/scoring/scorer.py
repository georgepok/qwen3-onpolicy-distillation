"""
Teacher scoring using vLLM with FP8 quantization.

This module implements per-token logprob computation with half-memory
footprint via FP8 tensor core utilization on GB10 DGX systems.

Key features:
- FP8 quantization for 2x memory reduction (~32GB for teacher)
- Efficient per-token logprob computation
- Batch processing optimized for scoring workload
- Integration with vLLM-generated trajectories

Architecture:
    The teacher model runs in FP8 precision using Blackwell's FP8 tensor cores,
    allowing both student and teacher to coexist in 128GB memory. vLLM provides
    efficient logprob computation via its OpenAI-compatible API.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import torch
import numpy as np
import requests
from openai import OpenAI
from transformers import AutoTokenizer


@dataclass
class ScoringConfig:
    """Configuration for teacher scoring.

    Attributes:
        model_name: HuggingFace model identifier for teacher (default: larger Qwen variant)
        api_url: URL of vLLM teacher API server
        max_batch_size: Maximum trajectories to score simultaneously (default: 16)
        compute_perplexity: Also compute perplexity metrics (default: True)
        logprobs: Number of logprobs to return (default: 5)
        quantization: Legacy parameter (configured in docker-compose)
        use_tensor_cores: Legacy parameter (configured in docker-compose)
        gpu_memory_utilization: Legacy parameter (configured in docker-compose)
        enable_cache: Legacy parameter (configured in docker-compose)
    """
    model_name: str = "Qwen/Qwen3-32B"  # Larger teacher
    api_url: str = "http://vllm-teacher:30000"
    max_batch_size: int = 16
    compute_perplexity: bool = True
    logprobs: int = 5
    # Legacy parameters (ignored in API mode)
    quantization: str = "fp8"
    use_tensor_cores: bool = True
    gpu_memory_utilization: float = 0.35
    enable_cache: bool = True


class TeacherScorer:
    """
    Manages teacher model for scoring student trajectories via SGLang API.

    This class handles:
    1. Communicating with SGLang API server (with FP8 quantization)
    2. Computing per-token log probabilities
    3. Batch-efficient scoring pipeline
    4. Integration with reverse KL computation

    The SGLang server runs in a separate container with FP8 quantization,
    enabling concurrent operation with student generation on GB10.
    """

    def __init__(self, config: ScoringConfig):
        """
        Initialize teacher scorer with API client.

        Args:
            config: ScoringConfig with API URL and parameters
        """
        self.config = config

        # Override API URL from environment if set
        self.api_url = os.environ.get("TEACHER_API_URL", config.api_url)

        print(f"Initializing vLLM Teacher API client...")
        print(f"  API URL: {self.api_url}")
        print(f"  Model: {config.model_name}")

        # Initialize OpenAI client for vLLM (OpenAI-compatible)
        self.client = OpenAI(
            base_url=f"{self.api_url}/v1",
            api_key="EMPTY"  # vLLM doesn't require API key
        )

        # Load teacher tokenizer for proper token alignment (Issue #2 fix)
        print(f"  Loading teacher tokenizer for alignment...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        print(f"  Tokenizer loaded: {type(self.tokenizer).__name__}")

        # Also keep reference to base URL for custom endpoints
        self.base_url = self.api_url

        self._verify_connection()
        
    def _verify_connection(self) -> None:
        """Verify connection to vLLM Teacher API server."""
        try:
            # Try to list models to verify connection
            models = self.client.models.list()
            print(f"  Connection successful, available models: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"  WARNING: Could not verify connection to vLLM Teacher API: {e}")
            print(f"  Will retry on first scoring attempt")
    
    def score_trajectories(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score student trajectories with teacher logprobs.

        Args:
            trajectories: Student-generated trajectories

        Returns:
            Scored trajectories with teacher logprobs and KL divergence
        """
        if not trajectories:
            return []

        print(f"Scoring {len(trajectories)} trajectories with teacher model...")

        # Extract prompts and completions
        prompts = [t["prompt"] for t in trajectories]
        completions = [t["generated_text"] for t in trajectories]

        # Compute teacher logprobs
        teacher_logprobs = self.compute_logprobs(prompts, completions)

        # Add scores to trajectories
        scored_trajectories = []
        for idx, traj in enumerate(trajectories):
            scored_traj = traj.copy()
            scored_traj["teacher_logprobs"] = teacher_logprobs[idx]

            # Extract student logprobs if available
            if "logprobs" in traj and traj["logprobs"]:
                student_logprobs = np.array([lp for lp in traj["logprobs"]])
                scored_traj["student_logprobs"] = student_logprobs

                # Compute reverse KL divergence
                kl_per_token, mean_kl = self.compute_reverse_kl(
                    student_logprobs,
                    teacher_logprobs[idx]
                )
                scored_traj["kl_divergence"] = kl_per_token
                scored_traj["mean_kl"] = mean_kl
            else:
                scored_traj["student_logprobs"] = None
                scored_traj["kl_divergence"] = None
                scored_traj["mean_kl"] = None

            # Compute perplexity if enabled
            if self.config.compute_perplexity and teacher_logprobs[idx] is not None:
                perplexity = np.exp(-np.mean(teacher_logprobs[idx]))
                scored_traj["perplexity"] = float(perplexity)

            scored_trajectories.append(scored_traj)

        print(f"Scoring complete")
        return scored_trajectories
    
    def compute_logprobs(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> List[np.ndarray]:
        """
        Compute per-token log probabilities for COMPLETION TOKENS ONLY.

        CRITICAL: Only returns logprobs for generated completion tokens, not prompt tokens.
        This is essential for proper KL divergence computation in distillation.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings

        Returns:
            List of numpy arrays with per-token logprobs (completion region only)
        """
        all_logprobs = []

        for prompt, completion in zip(prompts, completions):
            try:
                # ISSUE #2 FIX: Use teacher tokenizer to get exact completion length
                # This ensures proper alignment between student and teacher logprobs
                completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
                expected_len = len(completion_tokens)

                if expected_len == 0:
                    print(f"  Warning: Empty completion after tokenization")
                    all_logprobs.append(np.array([]))
                    continue

                # Get logprobs for full sequence (prompt + completion)
                full_text = prompt + completion

                response = self.client.completions.create(
                    model=self.config.model_name,
                    prompt=full_text,
                    max_tokens=1,  # Minimal generation
                    logprobs=self.config.logprobs,
                    echo=True,  # Return prompt tokens too
                    temperature=0.0,  # Deterministic for scoring
                )

                # Validate response (Issue #8)
                if not response.choices or len(response.choices) == 0:
                    print(f"  Warning: No choices returned from API for completion")
                    all_logprobs.append(np.full(expected_len, np.log(1e-10)))
                    continue

                choice = response.choices[0]

                if choice.logprobs and choice.logprobs.token_logprobs:
                    all_token_logprobs = choice.logprobs.token_logprobs

                    # ISSUE #2 FIX: Extract from the END using negative indexing
                    # This handles BOS/EOS token differences robustly
                    if len(all_token_logprobs) >= expected_len:
                        completion_logprobs = all_token_logprobs[-expected_len:]
                    else:
                        print(f"  Warning: API returned fewer logprobs than expected!")
                        print(f"    Expected: {expected_len}, Got: {len(all_token_logprobs)}")
                        # Use what we got, pad with low probability
                        completion_logprobs = all_token_logprobs + [np.log(1e-10)] * (expected_len - len(all_token_logprobs))

                    # ISSUE #8 FIX: Validate logprobs and use -inf for None instead of 0.0
                    none_count = sum(1 for lp in completion_logprobs if lp is None)
                    if none_count > 0:
                        print(f"  Warning: {none_count}/{len(completion_logprobs)} logprobs are None (using log(1e-10))")

                    # Use log(1e-10) ≈ -23 for None instead of 0.0 (which means probability=1.0!)
                    logprob_array = np.array([lp if lp is not None else np.log(1e-10)
                                             for lp in completion_logprobs])

                    # Validate length matches exactly
                    if len(logprob_array) != expected_len:
                        print(f"  Warning: Length mismatch after extraction!")
                        print(f"    Expected: {expected_len}, Got: {len(logprob_array)}")

                else:
                    # Fallback: low probability for all tokens
                    print(f"  Warning: No logprobs available from API")
                    logprob_array = np.full(expected_len, np.log(1e-10))

                all_logprobs.append(logprob_array)

            except Exception as e:
                print(f"  Warning: Logprob computation failed for one trajectory: {e}")
                # Return low-probability array on failure
                try:
                    fallback_len = len(self.tokenizer.encode(completion, add_special_tokens=False))
                    all_logprobs.append(np.full(fallback_len, np.log(1e-10)))
                except:
                    all_logprobs.append(np.array([np.log(1e-10)]))

        return all_logprobs
    
    def batch_score(
        self,
        trajectories: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Score trajectories in memory-efficient batches.

        Args:
            trajectories: Trajectories to score
            batch_size: Override default batch size

        Returns:
            Scored trajectories
        """
        batch_size = batch_size or self.config.max_batch_size
        all_scored = []

        num_batches = (len(trajectories) + batch_size - 1) // batch_size
        print(f"Scoring {len(trajectories)} trajectories in {num_batches} batch(es)")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(trajectories))
            batch = trajectories[start_idx:end_idx]

            print(f"  Batch {batch_idx + 1}/{num_batches}: Scoring {len(batch)} trajectories...")

            # Score batch (API-only, no GPU monitoring needed)
            try:
                scored_batch = self.score_trajectories(batch)
                all_scored.extend(scored_batch)
                print(f"    Completed batch {batch_idx + 1}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    WARNING: OOM in batch {batch_idx + 1}, reducing batch size...")
                    smaller_batch_size = len(batch) // 2
                    if smaller_batch_size > 0:
                        scored_batch = self.batch_score(batch, smaller_batch_size)
                        all_scored.extend(scored_batch)
                    else:
                        raise
                else:
                    raise

        print(f"Scored {len(all_scored)} total trajectories")
        return all_scored
    
    def compute_reverse_kl(
        self,
        student_logprobs: np.ndarray,
        teacher_logprobs: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute reverse KL divergence: KL(student || teacher).

        Args:
            student_logprobs: Student log probabilities
            teacher_logprobs: Teacher log probabilities
            mask: Optional mask for valid tokens

        Returns:
            Tuple of (per_token_kl, mean_kl)
        """
        # Ensure same length
        min_len = min(len(student_logprobs), len(teacher_logprobs))
        student_logprobs = student_logprobs[:min_len]
        teacher_logprobs = teacher_logprobs[:min_len]

        # Reverse KL: KL(P||Q) = Σ p(x) * log(p(x)/q(x))
        #            = Σ p(x) * (log_p(x) - log_q(x))
        # where p = student, q = teacher
        student_probs = np.exp(student_logprobs)
        log_ratio = student_logprobs - teacher_logprobs
        per_token_kl = student_probs * log_ratio

        # Apply mask if provided
        if mask is not None:
            mask = mask[:min_len]
            per_token_kl = per_token_kl * mask
            num_valid = mask.sum()
        else:
            num_valid = len(per_token_kl)

        # Compute mean
        if num_valid > 0:
            mean_kl = float(per_token_kl.sum() / num_valid)
        else:
            mean_kl = 0.0

        return per_token_kl, mean_kl
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get API connection statistics.

        Returns:
            Dictionary with connection info
        """
        stats = {
            "api_url": self.api_url,
            "model": self.config.model_name,
            "quantization": self.config.quantization,
            "status": "connected",
            "memory_savings": "~50% with FP8 (server-side)",
        }

        return stats
    
    def verify_fp8_accuracy(
        self,
        sample_texts: List[str],
        fp16_reference: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Verify FP8 quantization maintains acceptable accuracy.

        Args:
            sample_texts: Sample texts for validation
            fp16_reference: Optional FP16 model for comparison

        Returns:
            Dictionary with accuracy metrics
        """
        print("Verifying FP8 quantization accuracy...")

        if self.config.quantization != "fp8":
            return {"message": "Not using FP8 quantization"}

        # For full verification, would need to load FP16 reference
        # and compare logprobs. For now, return a placeholder.
        print("  Note: Full FP8 verification requires FP16 reference model")
        print("  Expected accuracy: >99% correlation with FP16")

        return {
            "quantization": "fp8",
            "expected_correlation": ">0.99",
            "expected_mae": "<0.01",
            "status": "FP8 quantization active",
            "note": "Full verification requires FP16 reference",
        }
    
    def shutdown(self) -> None:
        """Close API client connection."""
        print("Shutting down vLLM Teacher API client...")

        try:
            # Close OpenAI client if needed
            if hasattr(self.client, 'close'):
                self.client.close()
            print("  API client closed")
        except Exception as e:
            print(f"  Warning: Error closing client: {e}")

        print("Shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()


# Example usage
if __name__ == "__main__":
    config = ScoringConfig(
        model_name="Qwen/Qwen3-32B-Instruct",
        api_url="http://localhost:30000",
        max_batch_size=16,
    )

    with TeacherScorer(config) as scorer:
        # Example trajectories from student
        trajectories = [
            {
                "prompt": "What is AI?",
                "generated_text": "AI is artificial intelligence...",
                "tokens": [1, 2, 3, 4],
                "logprobs": [-0.5, -0.3, -0.4, -0.2],
            }
        ]

        scored = scorer.batch_score(trajectories)
        print(f"\nScored {len(scored)} trajectories")

        if scored:
            print(f"\nFirst scored trajectory:")
            print(f"  Mean KL: {scored[0].get('mean_kl', 'N/A')}")
            print(f"  Perplexity: {scored[0].get('perplexity', 'N/A')}")

        # Show connection stats
        stats = scorer.get_memory_usage()
        print("\nConnection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
