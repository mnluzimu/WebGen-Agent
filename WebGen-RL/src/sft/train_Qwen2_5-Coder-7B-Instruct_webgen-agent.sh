
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

conda activate env/webgen-rl
cd $DIR/../..  # cd to WebGen-RL root

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0

model_name=Qwen2_5-Coder-7B-Instruct_webgen-agent_sft

OMP_NUM_THREADS=1 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 $DIR/train.py \
--ddp_timeout 3600 \
--processor qwen_agent \
--model_cfg models/Qwen2.5-Coder-1.5B-Instruct \
--train_file data/webgen-agent_train_sft.jsonl \
--output_dir outs/$model_name \
--logging_dir outs/$model_name \
--remove_unused_columns False \
--dataloader_num_workers 16 \
--max_len 32768 \
--max_steps -1 \
--num_train_epochs 1 \
--save_strategy "epoch" \
--warmup_ratio 0.1 \
--logging_steps 1 \
--learning_rate 4e-5 \
--lr_scheduler_type cosine \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--seed 3407 \
--deepspeed $DIR/config/deepspeed.json \
--bf16 \
--do_train \
--save_safetensors \
--gradient_checkpointing \
--run_name $model_name \
--save_total_limit 3 \
--save_only_model \
--report_to tensorboard