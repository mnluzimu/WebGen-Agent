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

conda activate /mnt/cache/luzimu/code_agent/.env/qwenvl

vllm serve /mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct \
    --port 28456 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=5,video=5 \
    --gpu_memory_utilization 0.8 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 4 \