"""
Student model loader for local training with TRL.

This module loads the student model locally using Unsloth for efficient
LoRA training. The model is used both for:
1. Generation (via TRL's PPOTrainer.generate())
2. Training (via TRL's PPOTrainer optimization)

This is separate from generator.py which uses vLLM API for inference-only scenarios.
"""

from typing import Optional
from dataclasses import dataclass
import torch
# Import unsloth BEFORE transformers to enable optimizations
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class LocalModelConfig:
    """Configuration for local student model loading.

    Attributes:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length (default: 2048)
        load_in_4bit: Use 4-bit quantization (default: True)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling (default: 32)
        lora_dropout: LoRA dropout (default: 0.0)
        target_modules: Modules to apply LoRA (default: None = auto)
    """
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Optional[list] = None


class LocalStudentModel:
    """
    Loads student model locally with LoRA for TRL training.

    This class provides:
    1. Efficient model loading with Unsloth + 4-bit quantization
    2. LoRA adapter configuration for parameter-efficient fine-tuning
    3. Direct PyTorch model access for TRL PPOTrainer

    The loaded model can be passed directly to TRLDistillationTrainer.
    """

    def __init__(self, config: LocalModelConfig):
        """
        Initialize local student model with LoRA.

        Args:
            config: LocalModelConfig with model and LoRA settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_peft_model = False

        print(f"Loading student model locally: {config.model_name}")
        print(f"  Max sequence length: {config.max_seq_length}")
        print(f"  4-bit quantization: {config.load_in_4bit}")
        print(f"  LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")

        self._load_model()

    def _load_model(self):
        """Load model with Unsloth for efficient LoRA training."""
        try:
            # Try Unsloth first (fastest for LoRA training)
            print("  Attempting to load with Unsloth...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                dtype=torch.bfloat16,
            )

            # Add LoRA adapters via Unsloth
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            self.is_peft_model = True
            print("  ✅ Model loaded successfully with Unsloth + LoRA")

        except Exception as e:
            # Fallback to standard transformers + PEFT
            print(f"  Unsloth loading failed: {e}")
            print("  Falling back to transformers + PEFT...")

            self._load_with_transformers()

    def _load_with_transformers(self):
        """Fallback: Load with standard transformers + PEFT."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model with quantization if requested
        model_kwargs = {}
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **model_kwargs
        )

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.is_peft_model = True

        print("  ✅ Model loaded with transformers + PEFT LoRA")

    def get_model(self):
        """Get the loaded PyTorch model for TRL training."""
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer

    def save_lora_adapters(self, output_dir: str):
        """
        Save only the LoRA adapter weights.

        Args:
            output_dir: Directory to save LoRA adapters
        """
        if not self.is_peft_model:
            print(f"WARNING: Model is not a PEFT model, saving full model instead")
            self.model.save_pretrained(output_dir)
        else:
            # Save only LoRA adapters (much smaller)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"  LoRA adapters saved to: {output_dir}")

    def load_lora_checkpoint(self, checkpoint_dir: str):
        """
        Load LoRA weights from checkpoint.

        Args:
            checkpoint_dir: Directory containing LoRA checkpoint
        """
        print(f"Loading LoRA checkpoint: {checkpoint_dir}")

        if self.is_peft_model:
            # Load adapters into existing PEFT model
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model.base_model,
                checkpoint_dir,
            )
            print("  ✅ LoRA checkpoint loaded")
        else:
            print("  WARNING: Cannot load LoRA into non-PEFT model")

    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Trainable params: {trainable_params:,} / {all_params:,} "
                  f"({100 * trainable_params / all_params:.2f}%)")


# Example usage
if __name__ == "__main__":
    config = LocalModelConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=2048,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
    )

    loader = LocalStudentModel(config)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    print("\nModel loaded successfully!")
    loader.print_trainable_parameters()

    # Test generation
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest generation: {generated_text}")
