DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../..

model_name=deepseek-chat
fb_model_name=deepseek-chat

until python src/infer_batch.py \
    --model ${model_name} \
    --vlm_model Qwen/Qwen2.5-VL-32B-Instruct \
    --fb_model ${fb_model_name} \
    --data-path data/webgen-bench/test.jsonl \
    --workspace-root workspaces_root \
    --log-root service_logs \
    --max-iter 20 \
    --num-workers 4 \
    --eval-tag select_best \
    --error-limit 5 \
    --max-tokens -1 \
    --max-completion-tokens -1 \
    --temperature 0.5
do
    echo "Run failed (exit code $?). Retrying in 10 s…"
    sleep 10
done
