#!/bin/bash
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# Log the setup for debugging
echo "Starting AI Service with Blackwell Shim:"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

source ai_service/venv/bin/activate
uvicorn ai_service.app:app --host 0.0.0.0 --port 8001
