import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAILIKE_VLM_API_KEY"], 
                base_url=os.environ["OPENAILIKE_VLM_BASE_URL"])


def vlm_generation(messages, model):
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return chat_response.choices[0].message.content