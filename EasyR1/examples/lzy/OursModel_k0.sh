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

ray stop
# 路径
MODEL_PATH=/g0001sr/lzy/model/Qwen2.5-VL-7B-Instruct-rmm1004-full-sft-2e
RSVG_JSONL_DIR=/g0001sr/lzy/datasets/all_for_train_k/k_0
FORMAT_PROMPT=/g0001sr/lzy/EasyR1/examples/format_prompt/Ours.jinja
REWARD_PY=/g0001sr/lzy/EasyR1/examples/reward_function/oursIOU_k0.py
EXP_NAME=1014Ours_K0
#override_chat_template=/g0001sr/lzy/model/Qwen2.5-VL-7B-Instruct-rmm1004-full-sft-2e/chat_template.jinja

  #data.override_chat_template=${override_chat_template}\
# （可选）离线 & 稳定性
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export VLLM_USE_V1=1 && ray start --head
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
  worker.rollout.tensor_parallel_size=4 \
  worker.reward.reward_type=batch \
  worker.reward.reward_function=${REWARD_PY}:compute_score \
  trainer.experiment_name=${EXP_NAME} \
  trainer.n_gpus_per_node=8 \
  worker.actor.global_batch_size=512 \
  worker.actor.micro_batch_size_per_device_for_experience=16 \
  data.val_batch_size=512\
  data.rollout_batch_size=512\
  trainer.total_epochs=20\
  trainer.save_limit=7\
  trainer.save_freq=20
