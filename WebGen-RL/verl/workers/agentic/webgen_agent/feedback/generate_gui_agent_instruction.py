import re
from typing import Optional

import os

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from .vlm_generation import vlm_generation


function_test_prompt = """Based on the original website development instruction, you should identify the requirements and create an instruction for a web-navigation GUI agent to test the generated website. The following is an example of triggering the GUI agent testing based on the original instruction:

<example>
Original instruction: Please implement a self-driving tour website that provides self-driving tour products and services. The website should have functionalities for browsing self-driving tour routes, booking self-driving tour hotels, and self-help self-driving tour packages. Users should be able to browse different types of self-driving tour routes, book hotels and packages, and query self-driving club information. The website should also provide search and filtering functions to help users quickly find the self-driving tour products they need. Define background as cream; define components with dark teal.

<boltAction type="gui_agent_test">Verify cream background and dark‑teal buttons. Browse different types of self-driving tour routes, book hotels and packages, and query self-driving club information. Search and filter for self-driving tour products.</boltAction>
</example>

The following is the original website developemnt instruction:

<instruction>{instruction}</instruction>

Trigger the GUI agent testing based on the original instruction in a way similar to the example. Do not generate additional comments.
"""


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


def generate_gui_agent_instruction(instruction: str, model: str) -> Optional[str]:
    prompt = function_test_prompt.format(instruction=instruction)

    retry_num = 5
    for i in range(retry_num):
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
        result = vlm_generation(messages, model)
        result = extract_gui_agent_instruction(result)
        if result is not None:
            break

    result = result + "\n\nIf prompted for a username, password, or email in the process of testing, enter \"admin,\" \"admin123456\", and \"admin@example.com\", respectively.\n\nAnswer with one of the following:\n- YES: if the testing instruction was fully achieved during your interactions.\n- NO: if the testing instruction was not achieved at all.\nProvide your final answer based on your testing experience."
    return result


if __name__ == "__main__":
    instruction = "Please implement a website for generating stock reports to provide stock information and analysis. The website should have the functionality to search and summarize stock information, and generate customized stock reports based on user requirements. Users should be able to input stock codes or names, select report formats and content, and the website will automatically generate the corresponding reports. The reports should include basic stock information, market trends, financial data, and more. Set the background color to white and the component color to navy."
    model = "deepseek-v3-250324"
    print(generate_gui_agent_instruction(instruction, model))