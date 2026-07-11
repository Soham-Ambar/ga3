import json
import os
import re
from typing import Any, Literal

import dateparser
from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter()


EXPECTED_KEYS = {
    "vendor",
    "currency",
    "total_amount",
    "invoice_date",
    "due_in_days",
    "is_paid",
    "priority",
    "contact_email",
    "line_items",
    "item_count",
}


class InvoiceRequest(BaseModel):
    document_id: str
    text: str

    # Accept incoming JSON key "schema" without creating
    # the Pydantic warning about shadowing BaseModel.schema.
    requested_schema: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(
        populate_by_name=True,
    )


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str
    quantity: int
    unit_price: int


class InvoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    currency: Literal["USD", "EUR", "GBP", "INR", "JPY"]
    total_amount: int
    invoice_date: str
    due_in_days: int
    is_paid: bool
    priority: Literal["low", "normal", "high", "urgent"]
    contact_email: str
    line_items: list[LineItem]
    item_count: int


STRICT_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {
            "type": "string",
        },
        "currency": {
            "type": "string",
            "enum": [
                "USD",
                "EUR",
                "GBP",
                "INR",
                "JPY",
            ],
        },
        "total_amount": {
            "type": "integer",
        },
        "invoice_date": {
            "type": "string",
        },
        "due_in_days": {
            "type": "integer",
        },
        "is_paid": {
            "type": "boolean",
        },
        "priority": {
            "type": "string",
            "enum": [
                "low",
                "normal",
                "high",
                "urgent",
            ],
        },
        "contact_email": {
            "type": "string",
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                    },
                    "quantity": {
                        "type": "integer",
                    },
                    "unit_price": {
                        "type": "integer",
                    },
                },
                "required": [
                    "sku",
                    "quantity",
                    "unit_price",
                ],
                "additionalProperties": False,
            },
        },
        "item_count": {
            "type": "integer",
        },
    },
    "required": [
        "vendor",
        "currency",
        "total_amount",
        "invoice_date",
        "due_in_days",
        "is_paid",
        "priority",
        "contact_email",
        "line_items",
        "item_count",
    ],
    "additionalProperties": False,
}


def get_groq_client() -> Groq:
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


def clean_integer(
    value: Any,
    field_name: str,
) -> int:
    """
    Convert a generated value into a real integer.

    Supported examples:
    12480
    12480.0
    "12,480"
    "1,24,800"
    "$12480"
    "₹1,24,800"
    """

    if isinstance(value, bool):
        raise HTTPException(
            status_code=500,
            detail=f"{field_name} cannot be boolean",
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

        raise HTTPException(
            status_code=500,
            detail=f"{field_name} is not a whole number",
        )

    if isinstance(value, str):
        cleaned = value.strip()

        for character in [
            ",",
            "₹",
            "$",
            "€",
            "£",
            "¥",
        ]:
            cleaned = cleaned.replace(character, "")

        cleaned = cleaned.strip()

        try:
            number = float(cleaned)
        except ValueError as error:
            raise HTTPException(
                status_code=500,
                detail=f"{field_name} is not an integer",
            ) from error

        if not number.is_integer():
            raise HTTPException(
                status_code=500,
                detail=f"{field_name} is not a whole number",
            )

        return int(number)

    raise HTTPException(
        status_code=500,
        detail=f"{field_name} has an invalid type",
    )


def normalize_vendor(value: str) -> str:
    """
    Remove punctuation that merely ends the invoice sentence while
    preserving punctuation that is normally part of a legal company name.
    """

    vendor = value.strip()

    legal_suffixes = (
        "Inc.",
        "Ltd.",
        "Co.",
        "Corp.",
        "Pvt.",
        "L.L.C.",
        "S.A.",
        "S.p.A.",
    )

    if vendor.endswith(legal_suffixes):
        return vendor

    # Remove punctuation added only because the company name
    # appeared at the end of a sentence.
    vendor = re.sub(
        r"[.,;:!?]+$",
        "",
        vendor,
    ).strip()

    return vendor


def normalize_date(value: str) -> str:
    """
    Return invoice date in YYYY-MM-DD format.
    """

    value = value.strip()

    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        value,
    ):
        return value

    parsed = dateparser.parse(
        value,
        settings={
            "DATE_ORDER": "DMY",
            "STRICT_PARSING": False,
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )

    if parsed is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not normalize invoice date: "
                f"{value}"
            ),
        )

    return parsed.strftime("%Y-%m-%d")


def normalize_currency(value: str) -> str:
    currency = value.strip().upper()

    aliases = {
        "$": "USD",
        "US$": "USD",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
        "US DOLLAR": "USD",
        "US DOLLARS": "USD",
        "€": "EUR",
        "EURO": "EUR",
        "EUROS": "EUR",
        "£": "GBP",
        "POUND": "GBP",
        "POUNDS": "GBP",
        "POUND STERLING": "GBP",
        "POUNDS STERLING": "GBP",
        "₹": "INR",
        "RUPEE": "INR",
        "RUPEES": "INR",
        "INDIAN RUPEE": "INR",
        "INDIAN RUPEES": "INR",
        "¥": "JPY",
        "YEN": "JPY",
        "JAPANESE YEN": "JPY",
    }

    currency = aliases.get(
        currency,
        currency,
    )

    allowed = {
        "USD",
        "EUR",
        "GBP",
        "INR",
        "JPY",
    }

    if currency not in allowed:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported currency: {value}",
        )

    return currency


def normalize_priority(value: str) -> str:
    priority = value.strip().lower()

    aliases = {
        "routine": "low",
        "minor": "low",
        "standard": "normal",
        "regular": "normal",
        "medium": "normal",
        "important": "high",
        "critical": "urgent",
        "immediate": "urgent",
        "asap": "urgent",
    }

    priority = aliases.get(
        priority,
        priority,
    )

    allowed = {
        "low",
        "normal",
        "high",
        "urgent",
    }

    if priority not in allowed:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported priority: {value}",
        )

    return priority


def normalize_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        true_values = {
            "true",
            "yes",
            "paid",
            "paid in full",
            "settled",
            "complete",
            "completed",
            "payment received",
        }

        false_values = {
            "false",
            "no",
            "unpaid",
            "awaiting payment",
            "pending",
            "pending payment",
            "outstanding",
        }

        if normalized in true_values:
            return True

        if normalized in false_values:
            return False

    raise HTTPException(
        status_code=500,
        detail="is_paid is not a valid boolean",
    )


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_line_items(
    raw_items: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise HTTPException(
            status_code=500,
            detail="line_items must be an array",
        )

    normalized_items = []

    expected_item_keys = {
        "sku",
        "quantity",
        "unit_price",
    }

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"line_items[{index}] "
                    "must be an object"
                ),
            )

        if set(item.keys()) != expected_item_keys:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"line_items[{index}] has incorrect keys: "
                    f"{sorted(item.keys())}"
                ),
            )

        normalized_items.append(
            {
                "sku": str(item["sku"]).strip(),
                "quantity": clean_integer(
                    item["quantity"],
                    f"line_items[{index}].quantity",
                ),
                "unit_price": clean_integer(
                    item["unit_price"],
                    f"line_items[{index}].unit_price",
                ),
            }
        )

    return normalized_items


def normalize_result(
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Force exact keys, types, casing and item count.
    """

    if set(raw_result.keys()) != EXPECTED_KEYS:
        raise HTTPException(
            status_code=500,
            detail=(
                "LLM returned incorrect top-level keys. "
                f"Expected {sorted(EXPECTED_KEYS)}, "
                f"got {sorted(raw_result.keys())}"
            ),
        )

    line_items = normalize_line_items(
        raw_result["line_items"]
    )

    normalized = {
        "vendor": normalize_vendor(
            str(raw_result["vendor"])
        ),
        "currency": normalize_currency(
            str(raw_result["currency"])
        ),
        "total_amount": clean_integer(
            raw_result["total_amount"],
            "total_amount",
        ),
        "invoice_date": normalize_date(
            str(raw_result["invoice_date"])
        ),
        "due_in_days": clean_integer(
            raw_result["due_in_days"],
            "due_in_days",
        ),
        "is_paid": normalize_boolean(
            raw_result["is_paid"]
        ),
        "priority": normalize_priority(
            str(raw_result["priority"])
        ),
        "contact_email": normalize_email(
            str(raw_result["contact_email"])
        ),
        "line_items": line_items,

        # Always calculate item_count from the final
        # ordered line-item array.
        "item_count": len(line_items),
    }

    try:
        validated = InvoiceResponse.model_validate(
            normalized
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Invoice validation failed: "
                f"{error}"
            ),
        ) from error

    return validated.model_dump()


def extract_invoice(
    document_id: str,
    document_text: str,
    grader_schema: dict[str, Any],
) -> dict[str, Any]:
    client = get_groq_client()

    system_prompt = """
You are a precise invoice extraction engine.

Extract information from messy invoice text and return JSON matching the
provided schema exactly.

Follow these rules strictly:

1. vendor

   Return only the biller's proper company name.

   Do not include sentence-ending punctuation that follows the company name.

   Example:

   "Invoice issued by Vertex Cloud Systems."

   must produce:

   "Vertex Cloud Systems"

   Preserve punctuation only when it is genuinely part of the legal company
   name, such as "Acme, Inc." or "Example Co.".

   Preserve spelling, capitalization and legal suffixes.

2. currency

   Return only one ISO 4217 code:

   USD, EUR, GBP, INR or JPY.

   Currency examples:

   dollars, US dollars, $ -> USD
   euros, euro, € -> EUR
   pounds sterling, pounds, £ -> GBP
   rupees, Indian rupees, ₹ -> INR
   yen, Japanese yen, ¥ -> JPY

3. total_amount

   Return an integer in the main currency unit.

   Remove currency symbols and separators.

   Correctly understand:

   12,480 -> 12480
   1,24,800 -> 124800
   12K -> 12000
   12.5K -> 12500
   2M -> 2000000
   "twelve thousand four hundred eighty" -> 12480
   "one lakh twenty-four thousand eight hundred" -> 124800

4. invoice_date

   Convert every date format to YYYY-MM-DD.

   Understand numeric dates, month names and ordinal dates.

   Use the actual invoice date, not the due date or payment date.

5. due_in_days

   Return an integer number of days.

   Examples:

   Net 30 -> 30
   payable within 45 days -> 45
   due in two weeks -> 14
   due in three weeks -> 21
   due in one month -> 30
   due immediately -> 0
   due on receipt -> 0

6. is_paid

   Return true for wording such as:

   paid
   paid in full
   settled
   payment received

   Return false for wording such as:

   unpaid
   awaiting payment
   outstanding
   pending payment

7. priority

   Return exactly one lowercase value:

   low
   normal
   high
   urgent

   Preserve explicitly stated priority.

   Interpret critical, immediate or ASAP as urgent.
   Interpret important as high.
   Interpret routine as low.
   Use normal when no special priority is indicated.

8. contact_email

   Return the invoice contact or billing email.

   Convert it to lowercase.

9. line_items

   Extract each line item in the same order it appears.

   Each item must contain exactly:

   sku
   quantity
   unit_price

   quantity and unit_price must be integers.

   Do not add shipping, tax, discounts or totals as line items unless the
   document explicitly presents them as SKU line items.

10. item_count

    Return the number of extracted line items.

11. Ignore unrelated numbers such as:

    telephone numbers
    postal codes
    account identifiers
    shipment distances
    tax registration numbers
    document page numbers
    years appearing outside the invoice date

12. Return exactly the required keys.

    Do not add explanations, markdown or extra fields.
""".strip()

    user_prompt = f"""
Document ID:

{document_id}

Invoice document:

--- DOCUMENT START ---
{document_text}
--- DOCUMENT END ---

The grader supplied this JSON Schema:

{json.dumps(grader_schema, ensure_ascii=False)}

Extract the invoice carefully.

Return only the required JSON object.
""".strip()

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
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
                    "name": "invoice_extraction",
                    "strict": True,
                    "schema": STRICT_INVOICE_SCHEMA,
                },
            },
        )

        content = completion.choices[0].message.content

        if not content:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Invoice model returned "
                    "an empty response"
                ),
            )

        raw_result = json.loads(content)

        if not isinstance(raw_result, dict):
            raise HTTPException(
                status_code=502,
                detail=(
                    "Invoice model did not "
                    "return an object"
                ),
            )

        print(
            "Q7 raw result: "
            + json.dumps(
                raw_result,
                ensure_ascii=False,
            ),
            flush=True,
        )

        return normalize_result(raw_result)

    except HTTPException:
        raise

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=502,
            detail="Invoice model returned invalid JSON",
        ) from error

    except Exception as error:
        print(
            "Q7 extraction error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Invoice extraction failed: {error}",
        ) from error


@router.post(
    "/invoice-intelligence",
    response_model=InvoiceResponse,
)
def invoice_intelligence(
    request: InvoiceRequest,
) -> dict[str, Any]:
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Invoice text cannot be empty",
        )

    print(
        f"Q7 request: document_id={request.document_id}",
        flush=True,
    )

    result = extract_invoice(
        document_id=request.document_id,
        document_text=request.text,
        grader_schema=request.requested_schema,
    )

    print(
        "Q7 final response: "
        + json.dumps(
            result,
            ensure_ascii=False,
        ),
        flush=True,
    )

    return result


@router.get("/invoice-intelligence")
def invoice_intelligence_information():
    return {
        "message": (
            "Use POST with document_id, text and schema"
        )
    }