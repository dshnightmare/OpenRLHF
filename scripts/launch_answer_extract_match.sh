#!/usr/bin/env bash

set -x

openai_compatible=0

if [[ $openai_compatible -eq 0 ]]; then
  openai_infix=".openai"
  model_name_arg="--served-model-name ${run_mode}-model"
else
  openai_infix=""
  model_name_arg=""
fi

extract_model=/mnt/pfs/jinfeng_team/SFT/wanqian/yq9/models/math-policy/qwen_14b_answer_extract_gpt4_clean_english_4096_0116
for i in `seq 0 7`;
do
  CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints${openai_infix}.api_server --port 100$i --model extract_model \
    --trust-remote-code --disable-log-stats --enforce-eager -tp 1 $model_name_arg --gpu-memory-utilization 0.4 &>/dev/null &
done

match_model=/mnt/pfs/jinfeng_team/SFT/wanqian/yq9/models/math-policy/qwen_14b_timu1_finegrained_label_instruct_0109
for i in `seq 0 7`;
do
  CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints${openai_infix}.api_server --port 200$i --model extract_model \
    --trust-remote-code --disable-log-stats --enforce-eager -tp 1 $model_name_arg --gpu-memory-utilization 0.4 &>/dev/null &
done

wait
