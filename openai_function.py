import openai
from openai import OpenAI
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

model_name = "gpt-4.1-nano-2025-04-14"
token_limit = 10000

def configure_openai():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    openai.api_key = api_key
    print("OpenAI API key loaded successfully.")
    return OpenAI(api_key=api_key)

def generate_chat_completion(messages, model=model_name, max_tokens=token_limit):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens
        )

        if response.choices:
            return response
        return None

    except openai.APIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None