"""
Model registry and validation utilities.

Provides centralized model management for easy switching between
different teacher-student model pairs without code changes.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a model."""
    name: str
    size_params: float  # In billions
    memory_fp16_gb: float  # Estimated memory in FP16
    memory_fp8_gb: float  # Estimated memory in FP8
    max_seq_length: int
    architecture: str  # e.g., "qwen", "llama", "mistral"


# Model registry with common models
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Qwen3 models
    "Qwen/Qwen3-0.5B-Instruct": ModelSpec(
        name="Qwen/Qwen3-0.5B-Instruct",
        size_params=0.5,
        memory_fp16_gb=2,
        memory_fp8_gb=1,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-1.5B-Instruct": ModelSpec(
        name="Qwen/Qwen3-1.5B-Instruct",
        size_params=1.5,
        memory_fp16_gb=6,
        memory_fp8_gb=3,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-4B-Instruct": ModelSpec(
        name="Qwen/Qwen3-4B-Instruct",
        size_params=4,
        memory_fp16_gb=16,
        memory_fp8_gb=8,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-4B-Instruct-2507": ModelSpec(
        name="Qwen/Qwen3-4B-Instruct-2507",
        size_params=4,
        memory_fp16_gb=16,
        memory_fp8_gb=8,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-7B-Instruct": ModelSpec(
        name="Qwen/Qwen3-7B-Instruct",
        size_params=7,
        memory_fp16_gb=28,
        memory_fp8_gb=14,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-14B-Instruct": ModelSpec(
        name="Qwen/Qwen3-14B-Instruct",
        size_params=14,
        memory_fp16_gb=56,
        memory_fp8_gb=28,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-32B-Instruct": ModelSpec(
        name="Qwen/Qwen3-32B-Instruct",
        size_params=32,
        memory_fp16_gb=90,
        memory_fp8_gb=45,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-32B": ModelSpec(
        name="Qwen/Qwen3-32B",
        size_params=32,
        memory_fp16_gb=90,
        memory_fp8_gb=45,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen3-72B-Instruct": ModelSpec(
        name="Qwen/Qwen3-72B-Instruct",
        size_params=72,
        memory_fp16_gb=180,
        memory_fp8_gb=90,
        max_seq_length=4096,
        architecture="qwen",
    ),

    # Qwen2.5 models (backward compatibility)
    "Qwen/Qwen2.5-0.5B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        size_params=0.5,
        memory_fp16_gb=2,
        memory_fp8_gb=1,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        size_params=1.5,
        memory_fp16_gb=6,
        memory_fp8_gb=3,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-3B-Instruct",
        size_params=3,
        memory_fp16_gb=12,
        memory_fp8_gb=6,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-7B-Instruct",
        size_params=7,
        memory_fp16_gb=28,
        memory_fp8_gb=14,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen2.5-14B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-14B-Instruct",
        size_params=14,
        memory_fp16_gb=56,
        memory_fp8_gb=28,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen2.5-32B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-32B-Instruct",
        size_params=32,
        memory_fp16_gb=90,
        memory_fp8_gb=45,
        max_seq_length=4096,
        architecture="qwen",
    ),
    "Qwen/Qwen2.5-72B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-72B-Instruct",
        size_params=72,
        memory_fp16_gb=180,
        memory_fp8_gb=90,
        max_seq_length=4096,
        architecture="qwen",
    ),
}


def get_model_spec(model_name: str) -> Optional[ModelSpec]:
    """
    Get model specification from registry.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        ModelSpec if found, None otherwise
    """
    return MODEL_REGISTRY.get(model_name)


def validate_model_pair(
    student_model: str,
    teacher_model: str,
    total_gpu_memory_gb: float = 128.0,
    use_fp8_teacher: bool = True
) -> Tuple[bool, str]:
    """
    Validate that a student-teacher model pair is compatible.

    Args:
        student_model: Student model name
        teacher_model: Teacher model name
        total_gpu_memory_gb: Total GPU memory available
        use_fp8_teacher: Whether teacher will use FP8 quantization

    Returns:
        Tuple of (is_valid, message)
    """
    student_spec = get_model_spec(student_model)
    teacher_spec = get_model_spec(teacher_model)

    # Check if models are in registry
    if student_spec is None:
        return False, f"Student model '{student_model}' not found in registry. Add it to MODEL_REGISTRY in utils/models.py"

    if teacher_spec is None:
        return False, f"Teacher model '{teacher_model}' not found in registry. Add it to MODEL_REGISTRY in utils/models.py"

    # ISSUE #3 FIX: Allow cross-architecture distillation with warning
    # Cross-architecture (e.g., Llama student from Qwen teacher) can work,
    # but may require careful attention to tokenizer differences
    architecture_warning = ""
    if student_spec.architecture != teacher_spec.architecture:
        architecture_warning = (
            f"\n  ⚠️  WARNING: Cross-architecture distillation detected!\n"
            f"  Student: {student_spec.architecture}, Teacher: {teacher_spec.architecture}\n"
            f"  This may work but requires attention to:\n"
            f"    - Tokenizer vocabulary differences\n"
            f"    - Potential token alignment issues\n"
            f"    - Architecture-specific behaviors\n"
            f"  Consider using same architecture for best results.\n"
        )

    # Check size relationship (teacher should be larger)
    if teacher_spec.size_params <= student_spec.size_params:
        return False, f"Teacher ({teacher_spec.size_params}B) should be larger than student ({student_spec.size_params}B) for effective distillation."

    # Estimate memory usage
    teacher_memory = teacher_spec.memory_fp8_gb if use_fp8_teacher else teacher_spec.memory_fp16_gb
    student_memory = student_spec.memory_fp16_gb  # Student uses FP16
    training_overhead = 40  # LoRA, optimizer, replay buffer, etc.

    total_estimated = teacher_memory + student_memory + training_overhead

    if total_estimated > total_gpu_memory_gb:
        return False, (
            f"Insufficient GPU memory: {total_estimated:.1f}GB needed > {total_gpu_memory_gb:.1f}GB available.\n"
            f"  Teacher: {teacher_memory:.1f}GB ({'FP8' if use_fp8_teacher else 'FP16'})\n"
            f"  Student: {student_memory:.1f}GB (FP16)\n"
            f"  Training: {training_overhead}GB (LoRA + optimizer + buffer)\n"
            f"Try: smaller models, enable FP8, or increase GPU memory."
        )

    # Success
    utilization = (total_estimated / total_gpu_memory_gb) * 100
    success_message = (
        f"✓ Valid model pair:\n"
        f"  Student: {student_model} ({student_spec.size_params}B)\n"
        f"  Teacher: {teacher_model} ({teacher_spec.size_params}B)\n"
        f"  Memory: {total_estimated:.1f}GB / {total_gpu_memory_gb:.1f}GB ({utilization:.1f}% utilization)"
    )

    # ISSUE #3 FIX: Append architecture warning if cross-architecture
    if architecture_warning:
        success_message += architecture_warning

    return True, success_message


def list_compatible_pairs(
    total_gpu_memory_gb: float = 128.0,
    use_fp8_teacher: bool = True
) -> None:
    """
    Print all compatible student-teacher model pairs for given hardware.

    Args:
        total_gpu_memory_gb: Total GPU memory available
        use_fp8_teacher: Whether teacher will use FP8
    """
    print(f"\nCompatible Model Pairs ({total_gpu_memory_gb}GB GPU, Teacher {'FP8' if use_fp8_teacher else 'FP16'}):")
    print("=" * 80)

    # Get all Qwen models
    qwen_models = [name for name in MODEL_REGISTRY.keys() if "Qwen" in name and "Instruct" in name]
    qwen_models_sorted = sorted(qwen_models, key=lambda x: MODEL_REGISTRY[x].size_params)

    found_pairs = []

    for i, student in enumerate(qwen_models_sorted):
        for teacher in qwen_models_sorted[i+1:]:  # Only consider larger teachers
            is_valid, message = validate_model_pair(
                student, teacher, total_gpu_memory_gb, use_fp8_teacher
            )
            if is_valid:
                student_spec = MODEL_REGISTRY[student]
                teacher_spec = MODEL_REGISTRY[teacher]
                found_pairs.append((
                    student,
                    teacher,
                    student_spec.size_params,
                    teacher_spec.size_params
                ))

    if found_pairs:
        print(f"\nFound {len(found_pairs)} compatible pairs:\n")
        for student, teacher, s_size, t_size in found_pairs:
            print(f"  Student: {student:40s} ({s_size:4.1f}B)")
            print(f"  Teacher: {teacher:40s} ({t_size:4.1f}B)")
            print()
    else:
        print("\nNo compatible pairs found for this hardware configuration.")
        print("Try: increasing GPU memory or enabling FP8 quantization.")


if __name__ == "__main__":
    # Example usage
    print("Model Registry and Validation Utilities")
    print("=" * 80)

    # Test validation
    print("\n1. Validating Qwen3-4B student with Qwen3-32B teacher:")
    is_valid, message = validate_model_pair(
        "Qwen/Qwen3-4B-Instruct",
        "Qwen/Qwen3-32B-Instruct",
        total_gpu_memory_gb=128,
        use_fp8_teacher=True
    )
    print(message)

    # List compatible pairs
    print("\n2. Listing all compatible pairs for 128GB GPU:")
    list_compatible_pairs(total_gpu_memory_gb=128, use_fp8_teacher=True)

    # Test smaller GPU
    print("\n3. Listing compatible pairs for 80GB GPU:")
    list_compatible_pairs(total_gpu_memory_gb=80, use_fp8_teacher=True)
