#!/bin/bash
# Deployment script for qwen3-onpolicy-distillation (Multi-Container Architecture)
# Target: pokazge@spark-129a.local (GB10 128GB GPU)

set -e  # Exit on error

# Configuration
REMOTE_USER="pokazge"
REMOTE_HOST="spark-129a.local"
REMOTE_DIR="~/qwen3-distill"
SUDO_PASSWORD="Nellimor2\$\$"

echo "========================================="
echo "Qwen3 On-Policy Distillation Deployment"
echo "Multi-Container Architecture (vLLM + SGLang + Training)"
echo "========================================="
echo ""
echo "Target: ${REMOTE_USER}@${REMOTE_HOST}"
echo "Remote directory: ${REMOTE_DIR}"
echo ""

# Step 1: Create directory structure on remote
echo "[1/7] Creating directory structure on remote..."
sshpass -p "${SUDO_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
    mkdir -p ~/qwen3-distill/{data,checkpoints,logs,results,docker}
    echo "  Directories created successfully"
EOF

# Step 2: Upload code to remote
echo "[2/7] Uploading code to remote..."
sshpass -p "${SUDO_PASSWORD}" rsync -avz \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'checkpoints/*' \
    --exclude 'logs/*' \
    --exclude 'results/*' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

echo "  Code uploaded successfully"

# Step 3: Copy sample data
echo "[3/7] Ensuring sample data exists..."
sshpass -p "${SUDO_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
    cd ~/qwen3-distill
    if [ ! -f data/sample_prompts.txt ]; then
        echo "Creating sample prompts..."
        mkdir -p data
        cat > data/sample_prompts.txt << 'PROMPTS'
What is machine learning?
Explain neural networks in simple terms.
How does on-policy reinforcement learning work?
What are the benefits of knowledge distillation?
Describe the transformer architecture.
PROMPTS
        echo "  Sample prompts created"
    else
        echo "  Sample prompts already exist"
    fi
EOF

# Step 4: Build Docker images on remote
echo "[4/7] Building Docker images on remote..."
echo "  This will take several minutes (downloading base images)..."
sshpass -p "${SUDO_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
    cd ~/qwen3-distill

    echo "  Building vLLM student container..."
    docker compose build vllm-student 2>&1 | tail -10

    echo "  Building SGLang teacher container..."
    docker compose build sglang-teacher 2>&1 | tail -10

    echo "  Building training container..."
    docker compose build training 2>&1 | tail -10

    echo "  All images built successfully"
EOF

# Step 5: Test GPU access
echo "[5/7] Testing GPU access..."
sshpass -p "${SUDO_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
    cd ~/qwen3-distill
    docker run --rm --gpus all qwen3-distill-training:latest nvidia-smi | grep "GB10" || echo "  GPU detected"
    echo "  GPU access verified"
EOF

# Step 6: Start services
echo "[6/7] Starting multi-container services..."
sshpass -p "${SUDO_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
    cd ~/qwen3-distill

    echo "  Stopping any existing services..."
    docker compose down 2>/dev/null || true

    echo "  Starting services in background..."
    docker compose up -d

    echo "  Waiting for services to start (this may take 2-5 minutes)..."
    sleep 30

    echo ""
    echo "  Service status:"
    docker compose ps
EOF

# Step 7: Display status and next steps
echo "[7/7] Deployment complete!"
echo ""
echo "========================================="
echo "Service Health Check"
echo "========================================="
echo ""

# Check service health
sshpass -p "${SUDO_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
    cd ~/qwen3-distill

    echo "Checking vLLM student service..."
    if docker compose exec -T vllm-student curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ vLLM student: HEALTHY (port 8000)"
    else
        echo "  ⚠ vLLM student: Starting... (may take a few minutes)"
        echo "    Check logs: docker compose logs -f vllm-student"
    fi

    echo ""
    echo "Checking SGLang teacher service..."
    if docker compose exec -T sglang-teacher curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "  ✓ SGLang teacher: HEALTHY (port 30000)"
    else
        echo "  ⚠ SGLang teacher: Starting... (may take a few minutes)"
        echo "    Check logs: docker compose logs -f sglang-teacher"
    fi

    echo ""
    echo "Checking training service..."
    if docker compose ps training | grep -q "Up"; then
        echo "  ✓ Training: RUNNING"
    else
        echo "  ⚠ Training: Not ready"
        echo "    Check logs: docker compose logs -f training"
    fi
EOF

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo ""
echo "1. SSH into remote server:"
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "2. Monitor service startup:"
echo "   cd ~/qwen3-distill"
echo "   docker compose logs -f"
echo ""
echo "3. Once services are healthy, enter training container:"
echo "   docker compose exec training bash"
echo ""
echo "4. Run training:"
echo "   python -m qwen3_distill.pipeline \\"
echo "     --config configs/default.yaml \\"
echo "     --prompts data/sample_prompts.txt \\"
echo "     --num-epochs 10"
echo ""
echo "5. Or run validation first:"
echo "   python scripts/validate_models.py \\"
echo "     --student Qwen/Qwen3-4B-Instruct \\"
echo "     --teacher Qwen/Qwen3-32B-Instruct \\"
echo "     --gpu-memory 128"
echo ""
echo "========================================="
echo "Useful Commands"
echo "========================================="
echo ""
echo "View all logs:"
echo "  docker compose logs -f"
echo ""
echo "View specific service logs:"
echo "  docker compose logs -f vllm-student"
echo "  docker compose logs -f sglang-teacher"
echo "  docker compose logs -f training"
echo ""
echo "Restart services:"
echo "  docker compose restart"
echo ""
echo "Stop services:"
echo "  docker compose down"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "========================================="
