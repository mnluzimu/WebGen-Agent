__conda_setup="$('/usr/local/lib/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/lib/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/lib/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/lib/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /mnt/cache/luzimu/verl/.env/verl
cd /mnt/cache/luzimu/verl

model_path=/mnt/cache/luzimu/code_agent/outs/MultiTurn-Experiment/Qwen2_5-Coder-7B-Instruct_711-06071005_checkpoint-16_step-GRPO_pen-rep-tmp1_WebGen-Instruct_2_501/global_step_2/actor
hf_model_path=/mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_new_711-06071005/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_new_711-06071005_checkpoint-16

python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path ${hf_model_path} \
    --local_dir $model_path \
    --target_dir ${model_path}_hf \