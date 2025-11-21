"""
Configuration utilities for the on-policy distillation pipeline.

Provides functions for loading, merging, and validating YAML configuration files.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dataclasses import asdict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration parameters
        
    TODO for Claude Code:
        1. Load YAML file safely
        2. Validate required fields
        3. Apply default values where needed
        4. Handle file not found errors gracefully
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries with override priority.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (takes priority)
        
    Returns:
        Merged configuration dictionary
        
    TODO for Claude Code:
        1. Deep merge nested dictionaries
        2. Handle list concatenation vs replacement
        3. Preserve types during merge
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
        
    TODO for Claude Code:
        1. Convert config to YAML format
        2. Create parent directories if needed
        3. Write with proper formatting
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_from_dataclass(config_obj: Any) -> Dict[str, Any]:
    """
    Convert dataclass config object to dictionary.
    
    Args:
        config_obj: Dataclass configuration object
        
    Returns:
        Dictionary representation
    """
    return asdict(config_obj)


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema.

    Args:
        config: Configuration to validate
        schema: Schema with required fields and types

    Returns:
        True if valid, raises ValueError otherwise
    """
    def validate_field(value: Any, field_schema: Dict[str, Any], field_path: str) -> None:
        """Validate a single field against its schema."""
        # Check if field is required
        if field_schema.get("required", False) and value is None:
            raise ValueError(f"Required field '{field_path}' is missing")

        if value is None:
            return  # Optional field not provided

        # Check type
        expected_type = field_schema.get("type")
        if expected_type:
            if expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"Field '{field_path}' must be string, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"Field '{field_path}' must be number, got {type(value).__name__}")
            elif expected_type == "integer" and not isinstance(value, int):
                raise ValueError(f"Field '{field_path}' must be integer, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Field '{field_path}' must be boolean, got {type(value).__name__}")
            elif expected_type == "dict" and not isinstance(value, dict):
                raise ValueError(f"Field '{field_path}' must be dict, got {type(value).__name__}")

        # Check value range for numbers
        if isinstance(value, (int, float)):
            if "min" in field_schema and value < field_schema["min"]:
                raise ValueError(f"Field '{field_path}' must be >= {field_schema['min']}, got {value}")
            if "max" in field_schema and value > field_schema["max"]:
                raise ValueError(f"Field '{field_path}' must be <= {field_schema['max']}, got {value}")

        # Check allowed values
        if "enum" in field_schema and value not in field_schema["enum"]:
            raise ValueError(f"Field '{field_path}' must be one of {field_schema['enum']}, got {value}")

        # Recursively validate nested dict
        if isinstance(value, dict) and "properties" in field_schema:
            for nested_key, nested_schema in field_schema["properties"].items():
                nested_value = value.get(nested_key)
                validate_field(nested_value, nested_schema, f"{field_path}.{nested_key}")

    # Validate all top-level sections
    for section_name, section_schema in schema.items():
        section_value = config.get(section_name)
        validate_field(section_value, section_schema, section_name)

    return True


# Default configuration template
DEFAULT_CONFIG = {
    "generation": {
        "model_name": "Qwen/Qwen3-4B-Instruct",
        "num_parallel_instances": 1,
        "batch_size": 32,
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.9,
        "samples_per_prompt": 4,  # Multi-sampling: generate N trajectories per prompt
        "gpu_memory_utilization": 0.2,
    },
    "scoring": {
        "model_name": "Qwen/Qwen3-32B-Instruct",
        "quantization": "fp8",
        "max_batch_size": 16,
        "use_tensor_cores": True,
        "gpu_memory_utilization": 0.35,
    },
    "training": {
        "model_name": "Qwen/Qwen3-4B-Instruct",
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "total_steps": 10000,
        "mixed_precision": "bf16",
    },
    "replay": {
        "capacity": 10000,
        "device": "cuda",
        "enable_priority_sampling": True,
        "max_sequence_length": 512,
    },
    "pipeline": {
        "num_epochs": 10,
        "save_interval": 1000,
        "eval_interval": 500,
        "checkpoint_dir": "./checkpoints",
        "log_dir": "./logs",
    },
}


# Configuration schema for validation
CONFIG_SCHEMA = {
    "generation": {
        "type": "dict",
        "required": True,
        "properties": {
            "model_name": {"type": "string", "required": True},
            "num_parallel_instances": {"type": "integer", "required": True, "min": 1, "max": 4},
            "batch_size": {"type": "integer", "required": True, "min": 1},
            "max_tokens": {"type": "integer", "required": True, "min": 1},
            "temperature": {"type": "number", "required": True, "min": 0.0, "max": 2.0},
            "samples_per_prompt": {"type": "integer", "required": False, "min": 1, "max": 16},
            "gpu_memory_utilization": {"type": "number", "required": True, "min": 0.0, "max": 1.0},
        },
    },
    "scoring": {
        "type": "dict",
        "required": True,
        "properties": {
            "model_name": {"type": "string", "required": True},
            "quantization": {"type": "string", "enum": ["fp8", "fp16", "int8"]},
            "max_batch_size": {"type": "integer", "required": True, "min": 1},
            "gpu_memory_utilization": {"type": "number", "required": True, "min": 0.0, "max": 1.0},
        },
    },
    "training": {
        "type": "dict",
        "required": True,
        "properties": {
            "model_name": {"type": "string", "required": True},
            "lora_rank": {"type": "integer", "required": True, "min": 1},
            "lora_alpha": {"type": "integer", "required": True, "min": 1},
            "learning_rate": {"type": "number", "required": True, "min": 0.0},
            "batch_size": {"type": "integer", "required": True, "min": 1},
            "mixed_precision": {"type": "string", "enum": ["bf16", "fp16", "fp32"]},
        },
    },
    "replay": {
        "type": "dict",
        "required": True,
        "properties": {
            "capacity": {"type": "integer", "required": True, "min": 1},
            "device": {"type": "string", "required": True},
            "max_sequence_length": {"type": "integer", "required": True, "min": 1},
        },
    },
    "pipeline": {
        "type": "dict",
        "required": True,
        "properties": {
            "num_epochs": {"type": "integer", "required": True, "min": 1},
            "checkpoint_dir": {"type": "string", "required": True},
            "log_dir": {"type": "string", "required": True},
        },
    },
}
