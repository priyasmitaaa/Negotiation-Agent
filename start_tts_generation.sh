#!/bin/bash
# Robust TTS Generation Script
cd /home/paritosh/priyasmita/dialogue-dataset/Qwen3-tts

# Remove corrupted progress file if exists
if [ -f "tts_outputs/tts_progress.json" ]; then
    python3 -m json.tool tts_outputs/tts_progress.json > /dev/null 2>&1 || {
        echo "Corrupted progress file detected, backing up and removing..."
        mv tts_outputs/tts_progress.json tts_outputs/tts_progress.json.backup.$(date +%s)
    }
fi

# Start GPU 0 with nohup and logging
echo "Starting GPU 0 process..."
nohup python3 -u parallel_tts.py --gpu 0 --preprocessed > logs/gpu0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
GPU0_PID=$!
echo "GPU 0 PID: $GPU0_PID"

# Start GPU 1 with nohup and logging
echo "Starting GPU 1 process..."
nohup python3 -u parallel_tts.py --gpu 1 --preprocessed > logs/gpu1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
GPU1_PID=$!
echo "GPU 1 PID: $GPU1_PID"

# Save PIDs
echo $GPU0_PID > logs/gpu0.pid
echo $GPU1_PID > logs/gpu1.pid

echo "Both processes started!"
echo "Monitor with: tail -f logs/gpu*.log"
echo "Check PIDs: cat logs/gpu*.pid"
