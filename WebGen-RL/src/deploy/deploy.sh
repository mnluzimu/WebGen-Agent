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

conda activate /mnt/cache/luzimu/DeepSeek-R1/.env/vllm

vllm serve /mnt/cache/luzimu/code_agent/outs/Aguvis-7B-720P \
    --dtype auto \
    --host 0.0.0.0 \
    --port 8000 \
    --limit_mm_per_prompt image=4 \
    --max_model_len 8784 \
    --gpu_memory_utilization 0.8
