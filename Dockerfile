# Dockerfile for Qwen3-4B On-Policy Distillation
# Optimized for NVIDIA GB10 DGX Spark systems with 128GB GPU RAM

# Use official NVIDIA vLLM container specifically built for GB10 GPU
# This container includes vLLM pre-built and optimized for Blackwell architecture
FROM nvcr.io/nvidia/vllm:25.09-py3

# Set working directory
WORKDIR /workspace/qwen3-distill

# Install additional project-specific dependencies
# vLLM is already included in the base image - DO NOT reinstall it
# Use --no-deps to avoid breaking vLLM's dependencies
RUN pip install --no-cache-dir --no-deps \
    sglang>=0.2.0 \
    unsloth>=2024.1 \
    einops>=0.7.0 && \
    pip install --no-cache-dir bitsandbytes>=0.41.0

# Copy project files
COPY . /workspace/qwen3-distill

# Install project in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/qwen3-distill/checkpoints \
    /workspace/qwen3-distill/logs \
    /workspace/qwen3-distill/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]
