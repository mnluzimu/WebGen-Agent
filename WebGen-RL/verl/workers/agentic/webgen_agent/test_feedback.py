from .feedback import get_feedback
from .feedback.file_management import prompt_to_messages
from .feedback.generate_gui_agent_instruction import generate_gui_agent_instruction
from transformers import AutoTokenizer
import json
import time
with open("/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug_rollout_2025-06-12T03-45-35-112/execute_pred_2025-06-12T04-47-45-231.json", "r", encoding="utf-8") as f:
    inputs_responses = json.load(f)

vlm_model = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct"
model = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
instructions = []
gui_instructions = []
new_inputs_str = []
new_responses_str = []
for input_str, response_str in zip(inputs_responses["inputs_str"], inputs_responses["responses_str"]):
    messages = prompt_to_messages(input_str + response_str)
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant" and "<boltAction type=\"screenshot_validated\"/>" in messages[i]["content"]:
            new_messages = messages[:i+1]
            new_input_str = tokenizer.apply_chat_template(new_messages[:-1], tokenize=False, add_generation_prompt=False)
            new_response_str = tokenizer.apply_chat_template(new_messages[-1:], tokenize=False, add_generation_prompt=False)
            new_inputs_str.append(new_input_str)
            new_responses_str.append(new_response_str)
            instruction = messages[1]["content"]
            gui_instruction = generate_gui_agent_instruction(instruction, vlm_model)
            instructions.append(instruction)
            gui_instructions.append(gui_instruction)
            break

print(f"Total inputs: {len(new_inputs_str)}")

start = time.time()
results = get_feedback(new_inputs_str, new_responses_str, gui_instructions, instructions, vlm_model)
exe_time = time.time() - start

for instruction, gui_instruction, result in zip(instructions, gui_instructions, results):
    result["instruction"] = instruction
    result["gui_instruction"] = gui_instruction
print(exe_time)
with open("/mnt/cache/luzimu/code_agent/WebGen-RL/src/tests/feedback_webvoyager1.jsonl", "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")