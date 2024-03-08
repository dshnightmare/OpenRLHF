set -x 

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain /root/workspace/openrlhf/examples/scripts/ckpt/7b_llama/sft/byte/3EP \
    --save_path ./ckpt/7b_llama/rlhf/byte \
    --save_steps 50 \
    --ckpt_path ./ckpt/7b_llama/rlhf/byte/ckpt \
    --max_ckpt_num 10 \
    --logging_steps 1 \
    --eval_steps 50 \
    --num_episodes 21 \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 512 \
    --l2 0.1 \
    --max_epochs 2 \
    --prompt_max_len 384 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 1e-5 \
    --init_kl_coef 0.04 \
    --prompt_data /root/datasets/gsm8k/main/train-00000-of-00001.parquet \
    --eval_data /root/datasets/gsm8k/main/test-00000-of-00001.parquet \
    --input_key question \
    --output_key answer \
    --max_samples 80000 \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing
    --use_wandb 78586494a87f02f1479f6e43ebbfcabda19e2ce2
EOF
     
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     #  --reward_pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
