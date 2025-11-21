"""
GPU memory utilities for monitoring and optimization.

Provides functions for tracking GPU memory usage, optimizing allocations,
and ensuring efficient memory utilization on GB10 DGX systems.
"""

from typing import Dict, List, Optional
import torch
import gc


def get_gpu_memory_stats(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get GPU memory statistics.
    
    Args:
        device: GPU device index (None for current device)
        
    Returns:
        Dictionary with memory metrics in GB:
            - allocated: Currently allocated memory
            - reserved: Reserved by PyTorch
            - total: Total GPU memory
            - free: Available memory
            - utilization: Percentage of total memory used
            
    TODO for Claude Code:
        1. Query torch.cuda memory stats
        2. Convert bytes to GB
        3. Calculate utilization percentage
        4. Handle multi-GPU scenarios
    """
    if device is None:
        device = torch.cuda.current_device()
    
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "total": 0.0,
            "free": 0.0,
            "utilization": 0.0,
        }
    
    # Get memory stats in bytes
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    
    # Convert to GB
    gb = 1024**3
    stats = {
        "allocated": allocated / gb,
        "reserved": reserved / gb,
        "total": total / gb,
        "free": (total - reserved) / gb,
        "utilization": (reserved / total) * 100,
    }
    
    return stats


def optimize_memory() -> None:
    """
    Optimize GPU memory by clearing cache and running garbage collection.
    
    TODO for Claude Code:
        1. Clear PyTorch CUDA cache
        2. Run Python garbage collection
        3. Reset peak memory stats
        4. Log memory before/after optimization
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()


def log_gpu_memory(prefix: str = "", device: Optional[int] = None) -> None:
    """
    Log current GPU memory statistics.
    
    Args:
        prefix: Prefix for log message
        device: GPU device index
        
    TODO for Claude Code:
        1. Get memory stats
        2. Format message with prefix
        3. Log using configured logger
        4. Include peak memory if relevant
    """
    stats = get_gpu_memory_stats(device)
    
    message = f"{prefix} GPU Memory: " if prefix else "GPU Memory: "
    message += f"{stats['allocated']:.2f}GB allocated, "
    message += f"{stats['reserved']:.2f}GB reserved, "
    message += f"{stats['free']:.2f}GB free "
    message += f"({stats['utilization']:.1f}% utilization)"
    
    print(message)  # TODO: Use proper logger


def check_memory_available(required_gb: float, device: Optional[int] = None) -> bool:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        required_gb: Required memory in GB
        device: GPU device index
        
    Returns:
        True if sufficient memory available
        
    TODO for Claude Code:
        1. Get current free memory
        2. Compare with requirement
        3. Add safety margin (e.g., 10%)
        4. Log warning if insufficient
    """
    stats = get_gpu_memory_stats(device)
    available = stats['free']
    
    # Add 10% safety margin
    required_with_margin = required_gb * 1.1
    
    if available < required_with_margin:
        print(f"Warning: Insufficient GPU memory. Required: {required_with_margin:.2f}GB, Available: {available:.2f}GB")
        return False
    
    return True


def estimate_model_memory(
    model_name: str,
    precision: str = "fp16",
    batch_size: int = 1,
    sequence_length: int = 512
) -> float:
    """
    Estimate GPU memory required for model.
    
    Args:
        model_name: Model identifier
        precision: Precision (fp32, fp16, fp8)
        batch_size: Batch size
        sequence_length: Sequence length
        
    Returns:
        Estimated memory in GB
        
    TODO for Claude Code:
        1. Estimate parameter count from model name
        2. Calculate memory for parameters
        3. Estimate activation memory
        4. Include KV cache memory
        5. Add overhead for framework
    """
    # Rough estimates for Qwen models
    param_counts = {
        "Qwen/Qwen2.5-4B-Instruct": 4e9,
        "Qwen/Qwen2.5-14B-Instruct": 14e9,
    }
    
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
    }
    
    if model_name not in param_counts:
        print(f"Warning: Unknown model {model_name}, using default estimate")
        params = 4e9
    else:
        params = param_counts[model_name]
    
    # Model parameters
    model_memory = (params * bytes_per_param[precision]) / (1024**3)
    
    # Activation memory (rough estimate)
    activation_memory = (batch_size * sequence_length * 4096 * bytes_per_param[precision]) / (1024**3)
    
    # Total with 20% overhead
    total_memory = (model_memory + activation_memory) * 1.2
    
    return total_memory


def get_optimal_batch_size(
    model_memory_gb: float,
    available_memory_gb: float,
    min_batch_size: int = 1,
    max_batch_size: int = 32
) -> int:
    """
    Calculate optimal batch size given memory constraints.
    
    Args:
        model_memory_gb: Base model memory requirement
        available_memory_gb: Available GPU memory
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        
    Returns:
        Optimal batch size
        
    TODO for Claude Code:
        1. Calculate memory per sample
        2. Determine max batch that fits
        3. Round to power of 2 if beneficial
        4. Ensure within min/max bounds
    """
    # Estimate memory per sample (rough heuristic)
    memory_per_sample = model_memory_gb * 0.1
    
    # Calculate max batch size that fits
    max_fit = int((available_memory_gb - model_memory_gb) / memory_per_sample)
    
    # Clamp to bounds
    optimal = max(min_batch_size, min(max_fit, max_batch_size))
    
    return optimal
