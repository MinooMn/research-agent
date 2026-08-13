import json

import pytest

from src.llm_json import clean_json


def test_clean_json_strips_markdown_fences():
    raw = 'Sure, here is the plan:\n```json\n{"sub_questions": ["a", "b"]}\n```\n'
    assert clean_json(raw) == {"sub_questions": ["a", "b"]}


def test_clean_json_falls_back_to_brace_slicing_without_fences():
    raw = 'Here is the result: {"verdict": "supported", "reasoning": "ok"} hope that helps!'
    assert clean_json(raw) == {"verdict": "supported", "reasoning": "ok"}


def test_clean_json_raises_on_structurally_broken_json():
    raw = '```json\n{"verdict": "supported",\n```'
    with pytest.raises(json.JSONDecodeError):
        clean_json(raw)
