#!/bin/bash

# Check if enough arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: ./train.sh <GPU_ID> <NOTE>"
    echo "Example: ./train.sh 0 note"
    exit 1
fi

# Read input arguments
GPU_ID=$1       # First argument: GPU ID
NOTE=$2         # Second argument: Custom note
RUNNAME=$3

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Define log file name with timestamp and note
LOG_FILE="logs/main_${NOTE}_$(date +'%Y-%m-%d_%H-%M-%S').log"

# Create logs directory if not exists
mkdir -p logs

# Write a header note to the log file
echo "===== RUNNING main.py =====" > "$LOG_FILE"
echo "Note: $NOTE" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" >> "$LOG_FILE"
echo "============================" >> "$LOG_FILE"

# Run main.py in the background, redirect stdout and stderr to the log file
nohup bash -lc "source /data/home/yihan/ssm_hydrology/venv/bin/activate && exec -a pecan_${RUNNAME} python -u main.py" >> "$LOG_FILE" 2>&1 &

# Get the process ID (PID)
PID=$!

# Print message with PID
echo "main.py is running in the background on GPU $CUDA_VISIBLE_DEVICES with note: '$NOTE' (PID: $PID)"
echo "Log file: $LOG_FILE"
