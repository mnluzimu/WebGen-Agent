import os
import base64

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from prompts import system_prompt
from utils import vlm_generation

import json
import re
from typing import Dict


def parse_json_output(output_str: str) -> Dict[str, object]:
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
    # required_keys = {"is_error", "error_message", "screenshot_description"}
    # missing = required_keys - data.keys()
    # if missing:
    #     print(f"Missing required key(s): {', '.join(missing)}")

    return data


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


screenshot_prompt = """You are given a single website screenshot as input.

**Task**

1. Examine the screenshot closely for any rendering or runtime errors (e.g., "404 Not Found", stack traces, missing styles, blank areas).
2. Decide whether the screenshot *shows a rendering or runtime error*.

   * If **yes**, set `"is_error": true`, extract or paraphrase the visible error message(s) into `"error_message"`, and leave `"screenshot_description"` empty.
   * If **no**, set `"is_error": false`, leave `"error_message"` as an empty string (`""`), and write a concise but thorough `"screenshot_description"` that covers:

     * Overall layout (e.g., header/sidebar/footer, grid, flex, single-column).
     * Key UI components (navigation bar, buttons, forms, images, cards, tables, modals, etc.).
     * Color scheme and visual style (dominant colors, light/dark theme, gradients, shadows).
     * Visible content and text (headings, labels, sample data).
     * Notable design details (icons, spacing, font style) that help someone understand what the page looks like.
3. Suggest ways to improve the appearance of the website, for example:

   * Separate incorrectly overlapping components.
   * Adjust layout to avoid large blank areas.
   * Adjust text or background color to avoid text color being too similar to the background color.
   * If no improvement is necessary, leave `"suggestions"` as an empty string (`""`); otherwise, briefly list the suggestion(s) in `"suggestions"`.
4. Grade the 
**Output format (valid JSON)**

```json
{
  "is_error": <boolean>,
  "error_message": "<string>",
  "screenshot_description": "<string>",
  "suggestions": "<string>"
}
```

Return **only** this JSON object—no additional commentary, markdown, or code fences.
"""


def get_screenshot_description(image_path, model):
    base64_image = encode_image(image_path)
    messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content":[
                {
                    "type": "text",
                    "text": screenshot_prompt
                },
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                ]
            }
        ]
    #/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct
    result = vlm_generation(messages, model)
    data = parse_json_output(result)

    return data


appearance_grade_prompt = """
Instruction:
You are tasked with evaluating the functional design of a webpage. Grade the webpage's appearance on a scale of 0 to 5 (5 being highest), considering the following criteria:

  - Successful Rendering: Are there any components in the page or is it completely blank? Does the webpage render correctly without visual errors? Are colors, fonts, and components displayed as specified?
  - Content Relevance: Does the design align with the website’s purpose and user requirements? Are elements (e.g., search bars, report formats) logically placed and functional?
  - Layout Harmony: Is the arrangement of components (text, images, buttons) balanced, intuitive, and clutter-free?
  - Modernness & Beauty: Does the design follow contemporary trends (e.g., minimalism, responsive layouts)? Are colors, typography, and visual hierarchy aesthetically pleasing?

Grading Scale:

  - 0 (Blank Page): The screenshot is completely blank or does not contain any visible content. It may only have a background color or display an error message.
  - 1 (Poor): Major rendering issues (e.g., broken layouts, incorrect colors). Content is irrelevant or missing. Layout is chaotic. Design is outdated or visually unappealing.
  - 2 (Below Average): Partial rendering with noticeable errors. Content is partially relevant but poorly organized. Layout lacks consistency. Design is basic or uninspired.
  - 3 (Average): Mostly rendered correctly with minor flaws. Content is relevant but lacks polish. Layout is functional but unremarkable. Design is clean but lacks modern flair.
  - 4 (Good): Rendered well with no major errors. Content is relevant and logically organized. Layout is harmonious and user-friendly. Design is modern and visually appealing.
  - 5 (Excellent): Flawless rendering. Content is highly relevant, intuitive, and tailored to user needs. Layout is polished, responsive, and innovative. Design is cutting-edge, beautiful, and memorable.

Task:
Review the provided screenshot(s) of the webpage. Provide a concise analysis of a few sentences and then assign a grade (1–5) based on your analysis. Highlight strengths, weaknesses, and how well the design adheres to the specifications.

Your Response Format:

```json
{{
  "analysis": "<string>",
  "grade": <int>
}}
```

Your Response:
"""


def get_screenshot_grade(image_path, model, instruction=None):
    base64_image = encode_image(image_path)
    prompt = appearance_grade_prompt
    messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content":[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                ]
            }
        ]
    #/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct
    result = vlm_generation(messages, model)
    data = parse_json_output(result)

    return data, data.get("grade", 0)


if __name__ == "__main__":
    image_path = "/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/WebGenAgentV3_WebGen-Bench_Qwen2_5-Coder-7B-Instruct_711-06071005_checkpoint-16_step-GRPO_pen-rep-tmp1_WebGen-Instruct_2_501_global_step_5_iter20_select_best/000011/screenshot_step4_2025-07-07T12-48-53-423.png"
    model = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct"
    instruction = "Please implement a consulting business website that provides leadership and business training services. The website should have an e-commerce component, allowing customers to purchase services directly from the site. The website should include features such as workshops, online classes, podcasts, and blogs. Users should be able to browse and purchase services, and interact with the website. Set all pages to have a light cyan background and cadet blue components."
    data, grade = get_screenshot_grade(image_path, model, instruction)
    print(data)
    print(grade)