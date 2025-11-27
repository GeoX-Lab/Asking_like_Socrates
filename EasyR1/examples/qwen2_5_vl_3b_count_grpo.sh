#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/g0001sr/lzy/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/g0001sr/lzy/clevr_count_70k_json/train.jsonl \
    data.val_files=/g0001sr/lzy/clevr_count_70k_json/val.jsonl \
    data.image_dir=null \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=2
