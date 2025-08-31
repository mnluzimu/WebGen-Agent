python src/infer_single.py \
    --model deepseek-chat \
    --vlm_model Qwen/Qwen2.5-VL-32B-Instruct \
    --instruction "Please implement a wheel of fortune website where users can spin the wheel to win prizes. The website should have functionalities for spinning the wheel, displaying prizes, and recording user winning records. Users should be able to spin the wheel, view the prize list, view their own winning records. Use light gray as the default background and dark red for component styling." \
    --workspace-dir workspaces_root/test \
    --log-dir service_logs/test \
    --max-iter 20 \
    --overwrite \
    --error-limit 5