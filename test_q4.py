import json
import os

import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/dynamic-extract",
)


payload = {
    "text": (
        "Rahul bought 3 notebooks for Rs. 240 "
        "on 12 June 2026 from Alpha Store."
    ),
    "schema": {
        "customer_name": "string",
        "quantity": "integer",
        "amount": "float",
        "purchase_date": "date",
        "store": "string"
    }
}


print("Calling:", API_URL)

response = requests.post(
    API_URL,
    json=payload,
    timeout=120,
)

print("Status:", response.status_code)

try:
    print(json.dumps(response.json(), indent=2))
except ValueError:
    print(response.text)