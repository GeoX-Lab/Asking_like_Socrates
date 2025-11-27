#!/bin/bash
# 用法: bash run_rsvg_json.sh
# 循环运行 k = 0, 2.5, 5

# 如果需要 base 环境，请按需启用：
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate base

for K in 0 2.5 5
do
    echo "=============================="
    echo ">>> Running RSVG2json_with_k.py with k = $K"
    echo "=============================="
    python3 /g0001sr/lzy/EasyR1/scripts/RSVQA2json2.py --k "$K"
    echo ">>> Finished run for k = $K"
    echo
done
