set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --max_len 2048 \
    --dataset /root/datasets/gsm8k_byte/ \
    --input_key question \
    --output_key answer_cot \
    --dataset_probs 1.0 \
    --train_batch_size 64 \
    --micro_train_batch_size 4 \
    --max_samples 5000000 \
    --pretrain /root/models/Llama-2-7b-hf/ \
    --save_path ./ckpt/7b_llama/sft/byte/40EP \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 40 \
    --bf16 \
    --flash_attn \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --use_wandb 78586494a87f02f1479f6e43ebbfcabda19e2ce2
EOF
    # --wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
