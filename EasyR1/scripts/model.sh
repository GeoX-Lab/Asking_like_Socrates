#!/usr/bin/env bash
set -e

# 用法:
#   ./merge.sh <local_dir> [hf_repo]
#
# 参数:
#   local_dir  必填，模型分片所在目录
#   hf_repo    选填，Hugging Face 仓库名 (如 username/my-model)
LOCAL_DIR=/g0001sr/lzy/EasyR1/checkpoints/easy_r1/109R1_Onevision_7B_ONLYIOU/global_step_240/actor
#LOCAL_DIR=/g0001sr/lzy/EasyR1/checkpoints/easy_r1/922_R1_Onevision_7B/global_step_560/actor
HF_REPO=${2:-false}

if [ -z "$LOCAL_DIR" ]; then
  echo "用法: $0 <local_dir> [hf_repo]"
  exit 1
fi

python3 model_merger.py \
  --local_dir "$LOCAL_DIR" \
  --hf_upload_path "$HF_REPO"
