import os
import re

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel

load_dotenv()

router = APIRouter()


class ImageQARequest(BaseModel):
    image_base64: str
    question: str


def clean_answer(answer: str) -> str:
    answer = answer.strip()

    answer = answer.replace("```", "").strip()

    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()

    answer = answer.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", answer)

    if match:
        return match.group(0)

    return answer
    answer = answer.strip()

    answer = answer.replace("```", "").strip()

    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()

    # Convert numeric answers like ₹1,075.00 or 1,075.00 INR
    # into the grader-required format: 1075.00
    match = re.fullmatch(
        r"[₹$€£]?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:[A-Za-z%]+)?",
        answer,
    )

    if match:
        return match.group(1).replace(",", "")

    return answer


@router.post("/answer-image")
def answer_image(request: ImageQARequest):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing",
        )

    image_base64 = request.image_base64.strip()

    # Accept both raw base64 and data URLs.
    if image_base64.startswith("data:"):
        image_url = image_base64
    else:
        image_url = f"data:image/png;base64,{image_base64}"

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
Look carefully at the image and answer this question:

{request.question}

Return only the final answer.

Rules:
- Do not explain.
- Do not write "Answer:".
- For numeric answers, return only the number.
- Do not include currency symbols, commas, units, or extra words.
- For category or name questions, return only the category or name.
""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_completion_tokens=100,
        )

        answer = response.choices[0].message.content or ""
        answer = clean_answer(answer)

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Groq returned an empty answer",
            )

        return {
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Groq request failed: {str(exc)}",
        )