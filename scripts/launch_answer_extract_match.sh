#!/usr/bin/env bash

set -x

extract_model=/mnt/pfs/jinfeng_team/SFT/wanqian/yq9/models/math-policy/qwen_14b_answer_extract_gpt4_clean_english_4096_0116
for i in `seq 0 7`;
do
  CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.api_server --port 100$i --model ${extract_model} \
    --trust-remote-code --disable-log-stats --enforce-eager -tp 1 --gpu-memory-utilization 0.4 &>/dev/null &
done

sleep 300

match_model=/mnt/pfs/jinfeng_team/SFT/wanqian/yq9/models/math-policy/qwen_14b_timu1_finegrained_label_instruct_0109
for i in `seq 0 7`;
do
  CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.api_server --port 200$i --model ${match_model} \
    --trust-remote-code --disable-log-stats --enforce-eager -tp 1 --gpu-memory-utilization 0.4 &>/dev/null &
done

wait
