import json
import os

import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/extract",
)


invoice_text = """
INVOICE

Invoice No: INV-2026-0041
Date: 15 March 2026
Vendor: TechParts Pvt Ltd
Bill To: IITM Procurement Dept

Items:
USB-C Hub (x2) ........ Rs. 1,299.00
HDMI Cable (x3) ....... Rs. 450.00

Subtotal: Rs. 2,199.00
GST (18%): Rs. 395.82
TOTAL: Rs. 2,594.82
Currency: INR
"""


payload = {
    "invoice_text": invoice_text
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