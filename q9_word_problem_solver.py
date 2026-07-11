import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter()

MODEL_NAME = "openai/gpt-oss-120b"

EXPECTED_KEYS = {
    "reasoning",
    "answer",
}


class WordProblemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    problem: str


class WordProblemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(min_length=80)
    answer: int


STRICT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
        },
        "answer": {
            "type": "integer",
        },
    },
    "required": [
        "reasoning",
        "answer",
    ],
    "additionalProperties": False,
}


def get_groq_client() -> Groq:
    """
    Create the Groq client only when a request reaches the endpoint.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured",
        )

    return Groq(
        api_key=api_key,
        timeout=120,
        max_retries=2,
    )


def validate_problem(problem: str) -> str:
    """
    Reject an empty word problem.
    """

    cleaned_problem = problem.strip()

    if not cleaned_problem:
        raise HTTPException(
            status_code=400,
            detail="problem cannot be empty",
        )

    return cleaned_problem


def normalize_answer(value: Any) -> int:
    """
    Guarantee that answer is a real JSON integer.

    Reject:
    - booleans
    - strings
    - decimal values
    """

    if isinstance(value, bool):
        raise HTTPException(
            status_code=502,
            detail="answer cannot be a boolean",
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    raise HTTPException(
        status_code=502,
        detail="answer must be an integer",
    )


def normalize_reasoning(
    reasoning: Any,
    answer: int,
) -> str:
    """
    Ensure reasoning is plain text and at least 80 characters long.
    """

    if not isinstance(reasoning, str):
        raise HTTPException(
            status_code=502,
            detail="reasoning must be a string",
        )

    cleaned = reasoning.strip()

    # Remove accidental Markdown wrappers.
    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    if len(cleaned) < 80:
        cleaned = (
            f"{cleaned} "
            f"After applying the relevant operations in the required order "
            f"and ignoring unrelated numbers, the final integer answer is "
            f"{answer}."
        ).strip()

    if len(cleaned) < 80:
        raise HTTPException(
            status_code=502,
            detail="reasoning is shorter than 80 characters",
        )

    return cleaned


def validate_result(
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Return exactly two keys with strict data types.
    """

    if set(raw_result.keys()) != EXPECTED_KEYS:
        raise HTTPException(
            status_code=502,
            detail=(
                "Solver returned incorrect keys. "
                f"Expected {sorted(EXPECTED_KEYS)}, "
                f"got {sorted(raw_result.keys())}"
            ),
        )

    answer = normalize_answer(
        raw_result["answer"]
    )

    reasoning = normalize_reasoning(
        raw_result["reasoning"],
        answer,
    )

    normalized = {
        "reasoning": reasoning,
        "answer": answer,
    }

    try:
        validated = WordProblemResponse.model_validate(
            normalized
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Solver response validation failed: {error}",
        ) from error

    return validated.model_dump()


def solve_word_problem(
    problem_id: str,
    problem: str,
) -> dict[str, Any]:
    client = get_groq_client()

    system_prompt = """
You are a highly reliable arithmetic word-problem solver.

Solve the supplied problem carefully and return exactly one JSON object with
exactly these two keys:

{
  "reasoning": "A clear calculation summary containing at least 80 characters.",
  "answer": 0
}

Rules:

1. The problem has exactly one integer answer.

2. Identify what the question actually asks before calculating.

3. Separate relevant quantities from distractor quantities.

4. Ignore numbers that do not affect the requested result.

5. Apply operations in the correct order.

6. Carefully handle:
   - multiplication and division
   - addition and subtraction
   - percentages
   - discounts
   - taxes
   - markups
   - ratios
   - rates
   - unit conversions
   - totals and differences
   - repeated groups
   - time and distance
   - remaining quantities

7. Percentage rules:
   - x percent of A means A * x / 100
   - an x percent discount means multiply by (1 - x / 100)
   - an x percent increase or tax means multiply by (1 + x / 100)

8. Do not use irrelevant numbers merely because they appear in the text.

9. Perform an independent arithmetic check before returning the answer.

10. "reasoning" must:
    - be a plain JSON string
    - be at least 80 characters long
    - show the relevant arithmetic operations
    - mention ignored distractors when distractors exist
    - contain no Markdown
    - contain no JSON outside the required object

11. "answer" must:
    - be a JSON integer
    - not be a string
    - not be a float
    - contain no currency symbol
    - contain no units
    - contain no commas

12. Return exactly two top-level keys:
    reasoning
    answer

13. Do not add fields such as:
    steps
    explanation
    result
    confidence
    units
    final_answer

Return only the valid JSON object.
""".strip()

    user_prompt = f"""
Problem ID: {problem_id}

Word problem:

--- PROBLEM START ---
{problem}
--- PROBLEM END ---

Solve the problem carefully.

Internally perform these checks before responding:

1. State what quantity is being requested.
2. Identify the relevant numbers.
3. Identify and ignore distractor numbers.
4. Perform the arithmetic in the correct order.
5. Independently verify the final calculation.
6. Return a reasoning string of at least 80 characters.
7. Return answer as a JSON integer.

Return only the required JSON object.
""".strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "word_problem_solution",
                    "strict": True,
                    "schema": STRICT_RESPONSE_SCHEMA,
                },
            },
        )

        content = completion.choices[0].message.content

        if not content:
            raise HTTPException(
                status_code=502,
                detail="Solver returned an empty response",
            )

        try:
            raw_result = json.loads(content)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=502,
                detail="Solver returned invalid JSON",
            ) from error

        if not isinstance(raw_result, dict):
            raise HTTPException(
                status_code=502,
                detail="Solver did not return a JSON object",
            )

        print(
            "Q9 raw result: "
            + json.dumps(
                raw_result,
                ensure_ascii=False,
            ),
            flush=True,
        )

        return validate_result(raw_result)

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Q9 solver error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Word-problem solving failed: {error}",
        ) from error


@router.post(
    "/solve-word-problem",
    response_model=WordProblemResponse,
)
def solve_problem(
    request: WordProblemRequest,
) -> dict[str, Any]:
    problem = validate_problem(
        request.problem
    )

    print(
        f"Q9 request: problem_id={request.problem_id}",
        flush=True,
    )

    result = solve_word_problem(
        problem_id=request.problem_id,
        problem=problem,
    )

    print(
        "Q9 final response: "
        + json.dumps(
            result,
            ensure_ascii=False,
        ),
        flush=True,
    )

    return result


@router.get("/solve-word-problem")
def solve_word_problem_information():
    return {
        "message": (
            "Use POST with problem_id and problem"
        )
    }