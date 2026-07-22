import re
import json


def clean_json(model_output: str) -> dict:
    # 1. Try to extract content inside ```json ... ``` or ``` ... ```
    markdown_regex = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_regex, model_output)

    if match:
        json_string = match.group(1).strip()
    else:
        # 2. Fallback: If no backticks, find the first '{' and last '}'
        # This strips out any leading/trailing conversational text
        start_idx = model_output.find("{")
        end_idx = model_output.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_string = model_output[start_idx : end_idx + 1].strip()
        else:
            json_string = model_output.strip()

    # 3. Parse the cleaned string
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Handle cases where the JSON itself is structurally broken
        print(f"Failed to parse JSON: {e}")
        print(json_string)
        raise
