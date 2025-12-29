#!/bin/bash
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:/app/ai_service/MuseTalk
cd /app/ai_service/MuseTalk

echo "Starting Distillation Training..."
accelerate launch /app/ai_service/distill/train_distill.py \
    --teacher_model ./models/musetalkV15 \
    --student_model nota-ai/bk-sdm-tiny \
    --data_root ./dataset/HDTF \
    --resolution 256 \
    --batch_size 4 \
    --train_steps 50000 > train_distill_persistent.log 2>&1
