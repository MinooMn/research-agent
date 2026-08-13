import os

# config.py raises at import time if GROQ_API_KEY is unset. The unit tests
# under tests/unit never make a real API call, but importing src modules
# (e.g. src.agents.critic -> src.llm_client -> config) still needs a value
# present, so set a dummy one before any test module gets collected.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
