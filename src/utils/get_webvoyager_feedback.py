import os
import base64

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from webvoyager import run_single_task
from .llm_fb_generation import llm_fb_generation

import json
import re
from typing import Dict


def parse_screenshot_output(output_str: str) -> Dict[str, object]:
    # 1) Extract JSON block (handles optional Markdown/code fences)
    match = re.search(r"\{.*\}", output_str, flags=re.DOTALL)
    if not match:
        print("No JSON object found in the input string.")

    json_str = match.group(0)

    data = {}
    # 2) Parse JSON into Python dict
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}")

    # 3) Basic validation (optional but useful)
    # required_keys = {"test_passed", "improvement_suggestions"}
    # missing = required_keys - data.keys()
    # if missing:
    #     print(f"Missing required key(s): {', '.join(missing)}")

    return data


def convert_to_string(messages):
    result = []
    for message in messages:
        if message["role"] == "user":
            result.append("[webpage screenshot]")
        elif message["role"] == "assistant":
            result.append(message["content"])
    return "\n\n".join(result)


# summary_prompt = """**The GUI agent trajectory:**

# GUI-Agent Testing Instruction:

# {gui_instruction}

# Trajectory:

# {result}

# Based on the above GUI-agent testing process, generate a testing summary. The summary should include an analysis of whether each functional or appearance requirement was successfully met. The testing instructions specify which functional and appearance requirements the website must satisfy. List any unfulfilled functional or appearance requirements revealed by the testing and suggest how to implement them. If the testing is smooth and no problems arise, just output that the testing was successful and do not make any suggestions. IMPORTANT: You do not need to provide suggestions about enhancing the testing process. Focus on how to improve the website."""

summary_prompt = """You are given a GUI-agent testing trajectory.

**The GUI agent testing trajectory:**

GUI-Agent Testing Instruction:

{gui_instruction}

Trajectory:

{result}

**Task**

1. Examine the trajectory for any failed actions that indicates a problem in the website design.
2. Decide whether the GUI-agent testing trajectory reveals any flow in the website implementation.

   * If **yes**, set `"test_passed": true`, and leave `"improvement_suggestions"` empty.
   * If **no**, set `"test_passed": false`, and write a concise but thorough `"improvement_suggestions"` that covers the suggested improvements targeting the problems revealed by the testing result.
3. Evaluate the results of the GUI-agent test run and assign **one integer grade from 1 to 5**:
   * 1: The vast majority of tested functions fail or behave incorrectly.
   * 2: Many functions fail; only a few behave as expected.
   * 3: About half of the functions work as expected; success is mixed.
   * 4: Most functions work as expected; only minor issues remain.
   * 5: All tested functions work exactly as expected; no issues observed.
   * assign the grade to `"grade"`.
**Output format (valid JSON)**

```json
{{
  "test_passed": <boolean>,
  "improvement_suggestions": "<string>",
  "grade": <int>
}}
```

You can first make a short analysis of two or three sentences, then output this JSON object.
"""


def generate_summary(result, instruction, model, max_tokens, max_completion_tokens):
    prompt = summary_prompt.format(gui_instruction=instruction, result=result)
    messages = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": prompt},
    ]
    output = llm_fb_generation(messages, model, max_tokens=max_tokens, max_completion_tokens=max_completion_tokens)
    return output


def get_webvoyager_feedback(idx, output_dir, instruction, url, vlm_model, model, max_tokens, max_completion_tokens):
    task = {"web_name": idx, 
    "id": idx, 
    "ques": instruction, 
    "web": url}


    args_dict = {
        "api_model": vlm_model,
        "headless": True,
        "max_iter": 15,
        "max_attached_imgs": 3,
        "temperature": 1,
        "fix_box_color": True,
        "seed": 42,
        "output_dir": output_dir,
        "download_dir": os.path.join(output_dir, "download"),
        "window_width": 1024,
        "window_height": 768,
        "text_only": False,
        "save_accessibility_tree": False
    }

    messages = run_single_task(task, args_dict)
    result = convert_to_string(messages)
    if messages[-1]["role"] == "assistant" and "YES" in messages[-1]["content"]:
        summary = generate_summary(result, instruction, model, max_tokens, max_completion_tokens)
        feedback_json = parse_screenshot_output(summary)
        feedback_json["test_passed"] = True
        feedback_json["improvement_suggestions"] = ""
    else:
        summary = generate_summary(result, instruction, model, max_tokens, max_completion_tokens)
        feedback_json = parse_screenshot_output(summary)
    return summary, feedback_json


if __name__ == "__main__":
    # idx = "000001_step3"
    # output_dir = "/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/debug_webvoyager/feedback/000001"
    instruction = "Verify white background and navy buttons. Search and summarize stock information, generate customized stock reports by inputting stock codes or names and selecting report formats and content. Check that reports include basic stock information, market trends, and financial data."
    # url = "http://localhost:3298/"
    # vlm_model = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct"
    model = "deepseek-v3-250324"
    # result = get_webvoyager_feedback(idx, output_dir, instruction, url, vlm_model, model)
    # print(result)

    import json

    in_file = "/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/debug_agentv1_2/000002/task2025-05-21T14-36-33-065/interact_messages.json"
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data
    result = convert_to_string(messages)
    max_tokens = 32768
    summary = generate_summary(result, instruction, model, max_tokens, max_completion_tokens=max_tokens)
    print(summary)