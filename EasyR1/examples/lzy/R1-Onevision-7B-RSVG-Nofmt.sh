#!/usr/bin/env bash
# RSVG(JSONL) + exp(IoU(boxes))+中心距离奖励；单机多卡
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

# 进入工程目录
cd /g0001sr/lzy/EasyR1

# 激活 conda（bash 下）
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate base
fi

# 路径
MODEL_PATH=/g0001sr/lzy/R1-Onevision-7B
RSVG_JSONL_DIR=/g0001sr/lzy/DIOR_RSVG_jsonl_NEW_shortcut/
FORMAT_PROMPT=/g0001sr/lzy/EasyR1/examples/format_prompt/IOU.jinja
REWARD_PY=/g0001sr/lzy/EasyR1/examples/reward_function/IOU_No_Fmt.py
EXP_NAME=922R1_Onevision_7B_Nofmt

# （可选）离线 & 稳定性
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# 启动
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=${RSVG_JSONL_DIR}/train.jsonl \
  data.val_files=${RSVG_JSONL_DIR}/val.jsonl \
  data.image_dir=null \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.format_prompt=${FORMAT_PROMPT} \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.rollout.tensor_parallel_size=1 \
  worker.reward.reward_type=batch \
  worker.reward.reward_function=${REWARD_PY}:compute_score \
  trainer.experiment_name=${EXP_NAME} \
  trainer.n_gpus_per_node=4 \
  worker.actor.global_batch_size=128 \
  worker.actor.micro_batch_size_per_device_for_experience=16 \
  data.val_batch_size=128\
  trainer.total_epochs=30\
  trainer.save_limit=-1\
  trainer.save_freq=40
