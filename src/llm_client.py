from openai import OpenAI
from config import GROQ_API_KEY, GROQ_MODEL_DEFAULT

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def generate(prompt: str, model: str = GROQ_MODEL_DEFAULT) -> str:
    """Send a prompt to the LLM, return the full response as a string."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(generate("Explain how transformers work in two sentences."))
