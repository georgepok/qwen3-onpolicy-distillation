"""
Student trajectory generation using vLLM.

This module implements parallel student instance management for efficient
trajectory generation on Nvidia GB10 DGX systems with 128GB GPU RAM.

Key features:
- Multiple parallel vLLM instances for concurrent generation
- Batch-optimized sampling with temperature control
- GPU-resident inference pipeline
- Dynamic batch size adjustment based on available memory

Architecture:
    The generator maintains multiple vLLM engine instances to maximize
    GPU utilization on the GB10. Since the teacher model will be loaded
    in FP8 (occupying ~32GB), the student can use remaining capacity
    for parallel generation instances.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import time
import torch
from openai import OpenAI
from transformers import AutoTokenizer


@dataclass
class GenerationConfig:
    """Configuration for student trajectory generation.

    Attributes:
        model_name: HuggingFace model identifier for Qwen3-4B
        api_url: URL of vLLM OpenAI-compatible API server
        batch_size: Trajectories per batch (default: 32)
        max_tokens: Maximum generation length (default: 512)
        temperature: Sampling temperature (default: 1.0)
        top_p: Nucleus sampling parameter (default: 0.9)
        logprobs: Number of top logprobs to return (default: 5)
        samples_per_prompt: Number of trajectories to generate per prompt (default: 4)
        num_parallel_instances: Legacy parameter (kept for compatibility)
        gpu_memory_utilization: Legacy parameter (configured in docker-compose)
        tensor_parallel_size: Legacy parameter (configured in docker-compose)
        enable_chunked_prefill: Legacy parameter (configured in docker-compose)
    """
    model_name: str = "Qwen/Qwen3-4B-Instruct"
    api_url: str = "http://vllm-student:8000/v1"
    batch_size: int = 32
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    logprobs: int = 5
    samples_per_prompt: int = 4  # Number of trajectories to generate per prompt
    # Legacy parameters (ignored in API mode)
    num_parallel_instances: int = 1
    gpu_memory_utilization: float = 0.2
    tensor_parallel_size: int = 1
    enable_chunked_prefill: bool = True


class StudentGenerator:
    """
    Manages student model for trajectory generation via vLLM API.

    This class handles:
    1. Communicating with vLLM OpenAI-compatible API server
    2. Formatting prompts and collecting generated trajectories
    3. Memory-efficient batch processing
    4. Extracting logprobs for on-policy learning

    The vLLM server runs in a separate container, allowing clean
    separation of concerns and avoiding dependency conflicts.
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize the student generator with API client.

        Args:
            config: GenerationConfig object with API URL and sampling parameters
        """
        self.config = config

        # Override API URL from environment if set
        api_url = os.environ.get("VLLM_API_URL", config.api_url)
        if not api_url.endswith("/v1"):
            api_url = f"{api_url}/v1"

        print(f"Initializing vLLM API client...")
        print(f"  API URL: {api_url}")
        print(f"  Model: {config.model_name}")

        self.client = OpenAI(
            base_url=api_url,
            api_key="EMPTY"  # vLLM doesn't require API key
        )

        # Load tokenizer for proper token ID extraction
        print(f"  Loading tokenizer for token ID extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        print(f"  Tokenizer loaded: {type(self.tokenizer).__name__}")

        # LoRA adapter tracking for policy synchronization
        self.current_lora_path = None
        self.current_lora_name = None  # Track LoRA adapter name for vLLM API

        self._verify_connection()
        
    def _verify_connection(self) -> None:
        """Verify connection to vLLM API server."""
        try:
            # Try to list models to verify connection
            models = self.client.models.list()
            print(f"  Connection successful, available models: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"  WARNING: Could not verify connection to vLLM API: {e}")
            print(f"  Will retry on first generation attempt")
    
    def generate_trajectories(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate student trajectories for given prompts via API.

        Args:
            prompts: List of prompt strings for generation
            temperature: Override temperature (uses config default if None)
            max_tokens: Override max_tokens (uses config default if None)

        Returns:
            List of trajectory dictionaries with prompt, text, tokens, logprobs
        """
        if not prompts:
            return []

        # Use config defaults if not overridden
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        all_trajectories = []

        # Log which model/LoRA will be used for this generation batch
        if self.current_lora_name:
            print(f"[GENERATION] Using LoRA adapter: {self.current_lora_name}")
            print(f"[GENERATION] Policy synchronized - generating from trained model")
        else:
            print(f"[GENERATION] Using base model (no LoRA)")
            print(f"[GENERATION] Generating from untrained student policy")

        for prompt in prompts:
            try:
                # Prepare request parameters
                # Use LoRA model name if policy has been synchronized, otherwise base model
                # vLLM's OpenAI-compatible API with --enable-lora supports passing LoRA via model parameter
                model_name = self.current_lora_name if self.current_lora_name else self.config.model_name

                request_params = {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": self.config.top_p,
                    "logprobs": self.config.logprobs,
                    "n": self.config.samples_per_prompt,  # Multi-sampling: generate N trajectories per prompt
                }

                # ISSUE #11 FIX: Retry logic with exponential backoff
                MAX_RETRIES = 3
                response = None
                last_error = None

                for retry in range(MAX_RETRIES):
                    try:
                        # Call vLLM OpenAI API
                        response = self.client.completions.create(**request_params)
                        break  # Success - exit retry loop
                    except Exception as e:
                        last_error = e
                        error_str = str(e)

                        # ISSUE #12 FIX: If LoRA model not found, fall back to base model
                        if "404" in error_str and "does not exist" in error_str and self.current_lora_name:
                            print(f"  LoRA model '{self.current_lora_name}' not found in vLLM - falling back to base model")
                            # Update request to use base model
                            request_params["model"] = self.config.model_name
                            # Clear LoRA name to prevent future 404s
                            self.current_lora_name = None
                            self.current_lora_path = None
                            # Retry immediately with base model
                            continue

                        if retry < MAX_RETRIES - 1:
                            wait_time = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
                            print(f"  API error (attempt {retry + 1}/{MAX_RETRIES}): {e}")
                            print(f"  Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            # All retries failed
                            print(f"  ERROR: All {MAX_RETRIES} API attempts failed for prompt: {e}")
                            # Don't add error trajectory - skip this prompt entirely
                            continue

                # If we got here without a response, all retries failed - skip prompt
                if response is None:
                    continue

                # Extract all completions (multi-sampling: N trajectories per prompt)
                for choice in response.choices:
                    # Extract logprobs if available
                    logprobs_data = None
                    if choice.logprobs:
                        # OpenAI format: logprobs.token_logprobs is list of floats
                        logprobs_data = choice.logprobs.token_logprobs

                    # CRITICAL FIX (Issue #1): vLLM API returns token STRINGS, not IDs!
                    # We must tokenize the generated text ourselves to get actual token IDs.
                    # Without this, buffer.py creates zero-filled tensors → model trains on garbage.
                    token_ids = []
                    if choice.text:
                        # Tokenize the actual generated text to get proper token IDs
                        token_ids = self.tokenizer.encode(choice.text, add_special_tokens=False)

                    # Validate token extraction (Issue #5 prevention)
                    if not token_ids and choice.text:
                        print(f"  WARNING: Failed to extract tokens for non-empty text: {choice.text[:50]}...")

                    trajectory = {
                        "prompt": prompt,
                        "generated_text": choice.text,
                        "tokens": token_ids,  # Now actual integer token IDs, not strings!
                        "logprobs": logprobs_data,
                        "finish_reason": choice.finish_reason,
                    }
                    all_trajectories.append(trajectory)

            except Exception as e:
                # Outer exception handler for unexpected errors not caught by retry logic
                print(f"ERROR: Unexpected error during generation: {e}")
                # Skip this prompt - don't add error trajectory
                continue

        return all_trajectories
    
    def generate_batch(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate trajectories in batches with memory management.

        Args:
            prompts: List of prompts to process
            batch_size: Override default batch size if provided

        Returns:
            List of generated trajectories
        """
        batch_size = batch_size or self.config.batch_size
        all_trajectories = []

        # Process in batches
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        print(f"Processing {len(prompts)} prompts in {num_batches} batch(es) of size {batch_size}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]

            print(f"  Batch {batch_idx + 1}/{num_batches}: Generating {len(batch_prompts)} trajectories...")

            # Generate batch (API-only, no GPU monitoring needed)
            try:
                batch_trajectories = self.generate_trajectories(batch_prompts)
                all_trajectories.extend(batch_trajectories)
                print(f"    Completed batch {batch_idx + 1}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    WARNING: OOM in batch {batch_idx + 1}, attempting smaller batch...")
                    # Try with half batch size
                    smaller_batch_size = len(batch_prompts) // 2
                    if smaller_batch_size > 0:
                        batch_trajectories = self.generate_batch(batch_prompts, smaller_batch_size)
                        all_trajectories.extend(batch_trajectories)
                    else:
                        raise
                else:
                    raise

        print(f"Generated {len(all_trajectories)} total trajectories")
        return all_trajectories
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get API connection statistics.

        Returns:
            Dictionary with connection info
        """
        stats = {
            "api_url": self.config.api_url,
            "model": self.config.model_name,
            "status": "connected",
        }

        return stats

    def _convert_to_container_path(self, local_path: str) -> str:
        """
        Convert local checkpoint path to container-accessible path.

        Args:
            local_path: Local filesystem path to LoRA checkpoint

        Returns:
            Container path that vLLM can access

        Note:
            This handles various path formats robustly instead of hardcoded replacement.
        """
        from pathlib import Path

        local_path = Path(local_path).resolve()

        # Get the checkpoint directory name (e.g., "epoch_5_lora")
        checkpoint_name = local_path.name

        # Container path is always /workspace/checkpoints/{name}
        # This is mounted via docker-compose volumes
        container_path = f"/workspace/checkpoints/{checkpoint_name}"

        return container_path

    def reload_with_lora(self, lora_path: str) -> None:
        """
        Reload generator with new LoRA adapter for policy synchronization.

        This method updates the generator to use the latest trained LoRA weights
        from the checkpoint directory. The vLLM server must be configured with
        --enable-lora to support dynamic LoRA loading.

        Args:
            lora_path: Path to LoRA adapter directory

        Note:
            Registers the LoRA adapter with vLLM's management API and sets
            the current LoRA name to be used in subsequent generation requests.
        """
        from pathlib import Path
        import requests

        lora_dir = Path(lora_path)

        if not lora_dir.exists():
            print(f"WARNING: LoRA path does not exist: {lora_path}")
            print(f"  Policy sync skipped - generator will use base model")
            return

        print(f"Reloading generator with LoRA adapter: {lora_path}")

        # ISSUE #7 FIX: Robust path conversion instead of hardcoded replace
        container_lora_path = self._convert_to_container_path(lora_path)

        # Extract LoRA name (directory name)
        lora_name = Path(container_lora_path).name

        self.current_lora_path = container_lora_path
        self.current_lora_name = lora_name

        print(f"  Local path: {lora_path}")
        print(f"  Container path: {self.current_lora_path}")
        print(f"  LoRA name: {self.current_lora_name}")

        # Register LoRA adapter with vLLM server via management API
        try:
            # vLLM's LoRA loading endpoint (if available)
            # Note: This may need adjustment based on vLLM version
            api_base = self.config.api_url.replace('/v1', '')
            load_url = f"{api_base}/v1/load_lora_adapter"

            payload = {
                "lora_name": lora_name,
                "lora_path": container_lora_path
            }

            print(f"  Registering LoRA with vLLM server...")
            response = requests.post(load_url, json=payload, timeout=10)

            if response.status_code == 200:
                print(f"  LoRA adapter registered successfully")
            else:
                print(f"  WARNING: LoRA registration returned status {response.status_code}")
                print(f"  vLLM may not support dynamic LoRA loading - will try using model name directly")
        except Exception as e:
            # If registration fails, still try to use the LoRA by name
            # Some vLLM versions auto-discover LoRAs from the checkpoint directory
            print(f"  NOTE: LoRA registration failed ({e})")
            print(f"  Will attempt to use LoRA by name (vLLM may auto-discover it)")

        print(f"  Next generation will use LoRA: {self.current_lora_name}")

    def shutdown(self) -> None:
        """Close API client connection."""
        print("Shutting down vLLM API client...")

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
    # Example workflow
    config = GenerationConfig(
        model_name="Qwen/Qwen3-4B-Instruct",
        api_url="http://localhost:8000/v1",
        batch_size=32,
        temperature=0.7,
    )

    with StudentGenerator(config) as generator:
        prompts = ["What is machine learning?", "Explain neural networks."]
        trajectories = generator.generate_batch(prompts)
        print(f"\nGenerated {len(trajectories)} trajectories")

        # Print first trajectory
        if trajectories:
            print("\nFirst trajectory:")
            print(f"  Prompt: {trajectories[0]['prompt']}")
            print(f"  Generated: {trajectories[0]['generated_text'][:100]}...")
            if trajectories[0]['tokens']:
                print(f"  Tokens: {len(trajectories[0]['tokens'])}")

        # Show connection stats
        stats = generator.get_memory_stats()
        print("\nConnection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
