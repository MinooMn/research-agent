from openai import OpenAI
from config import GROQ_API_KEY, GROQ_MODEL_DEFAULT
import time

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def generate(prompt: str, model: str = GROQ_MODEL_DEFAULT, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2**attempt  # exponential backoff: 1, 2, 4, 8, 16s
                print(
                    f"Rate limited, retrying in {wait}s... (attempt {attempt+1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise
