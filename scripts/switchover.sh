#!/bin/bash
# Wait for a running stage PID to exit, then kill its (SIGSTOPped) run_all
# parent and launch a replacement script detached on the given GPU.
STAGE_PID=$1; PARENT_PID=$2; GPU=$3; SCRIPT=$4; LOG=$5
while kill -0 $STAGE_PID 2>/dev/null; do sleep 60; done
kill -9 $PARENT_PID 2>/dev/null
sleep 2
cd /lambda/nfs/aaai/aaai
env PATH=/home/ubuntu/venvs/rc/bin:$PATH HF_HOME=/home/ubuntu/hf \
  HF_TOKEN=$(cat /home/ubuntu/.hf_token) TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup setsid bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
disown
echo "SWITCHED: gpu$GPU -> $SCRIPT (old stage $STAGE_PID done, parent $PARENT_PID killed)"
