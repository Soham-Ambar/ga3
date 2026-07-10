import os
import json
import re
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel


load_dotenv()

router = APIRouter(tags=["Q3 - Invoice Extraction"])


class InvoiceRequest(BaseModel):
    invoice_text: str


def parse_json_response(text: str) -> dict:
    text = text.strip()

    # Remove Markdown code fences if the model adds them.
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


def clean_number(value) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    text = text.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return None

    return float(match.group(0))


def clean_date(value) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    # Already ISO
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        pass

    formats = [
        "%d %B %Y",
        "%d %b %Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
    ]

    # Remove ordinal suffixes: 3rd March 2026 -> 3 March 2026
    text = re.sub(
        r"(\d+)(st|nd|rd|th)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    for date_format in formats:
        try:
            parsed = datetime.strptime(text, date_format)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def clean_text(value) -> Optional[str]:
    if value is None:
        return None

    cleaned = str(value).strip()

    if not cleaned:
        return None

    return cleaned


def detect_currency(invoice_text: str) -> Optional[str]:
    text = invoice_text.lower()

    if "inr" in text or "rs." in text or "rs " in text or "₹" in invoice_text:
        return "INR"

    if "usd" in text or "$" in invoice_text:
        return "USD"

    if "eur" in text or "€" in invoice_text:
        return "EUR"

    if "gbp" in text or "£" in invoice_text:
        return "GBP"

    return None


@router.post("/extract")
def extract_invoice(request: InvoiceRequest):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing",
        )

    client = Groq(api_key=api_key)

    prompt = f"""
Extract invoice information from the text below.

Return only one valid JSON object with exactly these six keys:

{{
  "invoice_no": string or null,
  "date": string in YYYY-MM-DD format or null,
  "vendor": string or null,
  "amount": number or null,
  "tax": number or null,
  "currency": string or null
}}

Important rules:

1. invoice_no is the invoice number, reference number, or invoice ID.
2. date is the invoice issue date, not payment due date.
3. vendor is the seller, supplier, or company issuing the invoice.
4. amount is the subtotal before tax.
5. tax is only the tax amount, such as GST, VAT, IGST, CGST, or sales tax.
6. Do not use the grand total as amount.
7. currency should be a code such as INR, USD, EUR, or GBP.
8. Use null when a field cannot be found.
9. Do not return explanations or Markdown.

Invoice text:

{request.invoice_text}
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured invoice data. "
                        "Return valid JSON only."
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
            max_completion_tokens=500,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Groq request failed: {str(exc)}",
        )

    content = response.choices[0].message.content or ""
    data = parse_json_response(content)

    result = {
        "invoice_no": clean_text(data.get("invoice_no")),
        "date": clean_date(data.get("date")),
        "vendor": clean_text(data.get("vendor")),
        "amount": clean_number(data.get("amount")),
        "tax": clean_number(data.get("tax")),
        "currency": (
            clean_text(data.get("currency"))
            or detect_currency(request.invoice_text)
        ),
    }

    if result["currency"]:
        result["currency"] = result["currency"].upper()

    return result