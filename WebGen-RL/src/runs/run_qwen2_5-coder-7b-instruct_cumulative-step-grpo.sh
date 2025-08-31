DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../..  # cd to WebGen-RL root

export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS=14400000   # 4 h watchdog
export NCCL_ASYNC_ERROR_HANDLING=1                    # propagate failures

# 2 – Shrink memory footprint (GPU **and** CPU)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64  # less fragmentation
# and increase NCCL timeout

export DATA_DIR='data/webgen-agent_train_step-grpo'
BASE_MODEL='outs/Qwen2_5-Coder-7B-Instruct_webgen-agent_sft'
CKPT_PATH='outs'

PROJECT_NAME='MultiTurn-Experiment'
EXPERIMENT_NAME="Qwen2_5-Coder-7B-Instruct_cumulative-step-grpo"

KL_LOSS_COEF=0.001
ENTROPY_COEFF=0
KL_LOSS_TYPE=low_var_kl
N_AGENT=5
N_TURNS=5
TEMP=1.0
TOPP=0.95
USE_KL_LOSS=False
LR=1e-6
CLIP_LOW=0.2
CLIP_HIGH=0.2
GRAD_CLIP=0.5
BATCH_SIZE=16
TP_SIZE=4

if [ "${RANK}" != "0" ]; then
    sleep 10
    ray start --address ${MASTER_ADDR}:6379 --num-gpus 8
    sleep inf
else
    ray start --node-ip-address ${MASTER_ADDR} --num-gpus 8 --head
    sleep 30

    python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/validation.parquet \
        data.train_batch_size=$BATCH_SIZE \
        data.dataloader_num_workers=0 \
        data.val_batch_size=1 \
        data.max_prompt_length=23767 \
        data.max_response_length=9000 \
        actor_rollout_ref.rollout.webgen.max_prompt_length=23767 \
        actor_rollout_ref.rollout.webgen.max_response_length=9000 \
        actor_rollout_ref.rollout.webgen.max_start_length=23767 \
        actor_rollout_ref.rollout.webgen.max_obs_length=5000 \
        algorithm.adv_estimator=cumulative_step_grpo \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
        actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
        actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
        actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
        actor_rollout_ref.rollout.n_trajectories=$N_AGENT \
        actor_rollout_ref.rollout.max_iterations=$N_TURNS \
        actor_rollout_ref.rollout.name=async \
        actor_rollout_ref.rollout.enable_memory_saver=True \
        actor_rollout_ref.rollout.project_name=$PROJECT_NAME \
        actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
        actor_rollout_ref.rollout.task_type='webgen' \
        reward_model.reward_manager=webgen \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.sampling_params.temperature=$TEMP \
        actor_rollout_ref.rollout.sampling_params.top_p=$TOPP \
        actor_rollout_ref.actor.masking=True \
        trainer.logger=['tensorboard'] \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=1 \
        trainer.test_freq=-1 \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=5 \
        trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
        trainer.max_actor_ckpt_to_keep=2 \
        trainer.max_critic_ckpt_to_keep=2 \
        actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
        2>&1
fi