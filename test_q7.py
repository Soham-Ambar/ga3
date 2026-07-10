import json
import os

import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/invoice-intelligence",
)


payload = {
    "document_id": "doc0",
    "text": (
        "INVOICE from Acme Industrial Supply. "
        "Invoice date: March 3, 2024. "
        "Currency: US dollars. "
        "Total due: twelve thousand four hundred eighty dollars. "
        "Terms: Net 30. Awaiting payment. "
        "Priority: HIGH. Contact AP@ACME.COM. "
        "Items: WIDGET-204, quantity 12, unit price $40; "
        "BOLT-118, quantity 200, unit price $5."
    ),
    "schema": {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "currency": {"type": "string"},
            "total_amount": {"type": "integer"},
            "invoice_date": {"type": "string"},
            "due_in_days": {"type": "integer"},
            "is_paid": {"type": "boolean"},
            "priority": {"type": "string"},
            "contact_email": {"type": "string"},
            "line_items": {"type": "array"},
            "item_count": {"type": "integer"},
        },
    },
}


response = requests.post(
    API_URL,
    json=payload,
    timeout=180,
)

print("Calling:", API_URL)
print("Status:", response.status_code)

try:
    print(
        json.dumps(
            response.json(),
            indent=2,
            ensure_ascii=False,
        )
    )
except ValueError:
    print(response.text)