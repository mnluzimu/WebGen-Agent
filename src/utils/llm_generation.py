import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from prompts.system import system_prompt

client = OpenAI(api_key=os.environ["OPENAILIKE_API_KEY"], 
                base_url=os.environ["OPENAILIKE_BASE_URL"])


def llm_generation(messages, model, max_tokens=-1, max_completion_tokens=-1, temperature=0.5):
    if temperature > 0:
        if max_tokens > 0:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif max_completion_tokens > 0:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
        else:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
    else:
        if max_tokens > 0:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
        elif max_completion_tokens > 0:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
            )
        else:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

    return chat_response.choices[0].message.content