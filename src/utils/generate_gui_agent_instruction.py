import re
from typing import Optional

import os

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from prompts import function_test_prompt
from utils import llm_fb_generation


def extract_gui_agent_instruction(html: str) -> Optional[str]:
    """
    Return the inner text of a <boltAction type="gui_agent_test"> … </boltAction> tag.

    Parameters
    ----------
    html : str
        The HTML/XML snippet containing a boltAction element.

    Returns
    -------
    Optional[str]
        The trimmed instruction string if found, otherwise None.
    """
    pattern = (
        r'<boltAction\b[^>]*\btype=["\']gui_agent_test["\'][^>]*>'  # opening tag with correct type
        r'(.*?)'                                                    # non‑greedy capture of inner text
        r'</boltAction>'                                            # closing tag
    )
    match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def generate_gui_agent_instruction(instruction: str, model: str, max_tokens: int, max_completion_tokens: int) -> Optional[str]:
    prompt = function_test_prompt.format(instruction=instruction)

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
    result = llm_fb_generation(messages, model, max_tokens=max_tokens, max_completion_tokens=max_completion_tokens)
    result = extract_gui_agent_instruction(result)
    result = result + "\n\nIf prompted for a username, password, or email in the process of testing, enter \"admin,\" \"admin123456\", and \"admin@example.com\", respectively.\n\nAnswer with one of the following:\n- YES: if the testing instruction was fully achieved during your interactions.\n- NO: if the testing instruction was not achieved at all.\nProvide your final answer based on your testing experience."
    return result


if __name__ == "__main__":
    instruction = "Please implement a website for generating stock reports to provide stock information and analysis. The website should have the functionality to search and summarize stock information, and generate customized stock reports based on user requirements. Users should be able to input stock codes or names, select report formats and content, and the website will automatically generate the corresponding reports. The reports should include basic stock information, market trends, financial data, and more. Set the background color to white and the component color to navy."
    model = "deepseek-v3-250324"
    print(generate_gui_agent_instruction(instruction, model))