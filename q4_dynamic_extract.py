import os
import json
import re
from typing import Any, Dict

import dateparser
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel, Field


load_dotenv()

router = APIRouter(tags=["Q4 - Dynamic Extraction"])


class DynamicExtractRequest(BaseModel):
    text: str
    requested_schema: Dict[str, str] = Field(alias="schema")


def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    raise HTTPException(
        status_code=500,
        detail="Groq did not return valid JSON",
    )


def convert_date(value: Any):
    if value is None:
        return None

    parsed = dateparser.parse(str(value))

    if parsed is None:
        return None

    return parsed.strftime("%Y-%m-%d")


def convert_boolean(value: Any):
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "yes", "y", "1", "active", "enabled"}:
        return True

    if text in {"false", "no", "n", "0", "inactive", "disabled"}:
        return False

    return None


def convert_integer(value: Any):
    if value is None or isinstance(value, bool):
        return None

    try:
        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        cleaned = str(value).replace(",", "").strip()
        match = re.search(r"-?\d+", cleaned)

        if not match:
            return None

        return int(match.group(0))

    except (TypeError, ValueError):
        return None


def convert_float(value: Any):
    if value is None or isinstance(value, bool):
        return None

    try:
        if isinstance(value, (int, float)):
            return float(value)

        cleaned = str(value).replace(",", "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)

        if not match:
            return None

        return float(match.group(0))

    except (TypeError, ValueError):
        return None


def convert_string_array(value: Any):
    if value is None:
        return None

    if isinstance(value, list):
        return [str(item).strip() for item in value]

    text = str(value).strip()

    if not text:
        return None

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def convert_integer_array(value: Any):
    if value is None:
        return None

    values = value if isinstance(value, list) else str(value).split(",")

    result = []

    for item in values:
        converted = convert_integer(item)

        if converted is not None:
            result.append(converted)

    return result if result else None


def convert_value(value: Any, type_name: str):
    type_name = type_name.strip().lower()

    if value is None:
        return None

    if type_name == "string":
        cleaned = str(value).strip()
        return cleaned if cleaned else None

    if type_name == "integer":
        return convert_integer(value)

    if type_name == "float":
        return convert_float(value)

    if type_name == "boolean":
        return convert_boolean(value)

    if type_name == "date":
        return convert_date(value)

    if type_name == "array[string]":
        return convert_string_array(value)

    if type_name == "array[integer]":
        return convert_integer_array(value)

    return None


@router.post("/dynamic-extract")
def dynamic_extract(request: DynamicExtractRequest):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing",
        )

    client = Groq(api_key=api_key)

    schema_json = json.dumps(
        request.requested_schema,
        indent=2,
    )

    prompt = f"""
Extract information from the provided text according to the dynamic schema.

Text:
{request.text}

Requested schema:
{schema_json}

Return only one valid JSON object.

Important rules:

1. Return exactly the keys present in the requested schema.
2. Do not add any extra keys.
3. Do not omit any requested keys.
4. Use null when a value cannot be found.
5. string values must be JSON strings.
6. integer values must be JSON integers.
7. float values must be JSON numbers.
8. boolean values must be true or false.
9. date values must use YYYY-MM-DD.
10. array[string] must be an array of strings.
11. array[integer] must be an array of integers.
12. Do not return Markdown or explanations.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured data according to "
                        "a runtime schema. Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0,
            max_completion_tokens=700,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Groq request failed: {str(exc)}",
        )

    content = response.choices[0].message.content or ""
    extracted = parse_json_response(content)

    # Build the response only from requested schema keys.
    # This guarantees no extra keys and no missing keys.
    result = {}

    for field_name, type_name in request.requested_schema.items():
        raw_value = extracted.get(field_name)

        result[field_name] = convert_value(
            raw_value,
            type_name,
        )

    return result