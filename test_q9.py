import json
import os

import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/solve-word-problem",
)


payload = {
    "problem_id": "p0",
    "problem": (
        "A workshop orders 150 tiles at 8 dollars each. "
        "Any order of more than 50 units earns a 25% bulk discount. "
        "After the discount, a 5% tax is added. "
        "The workshop is 12 kilometers from the supplier and sells "
        "7 other product lines. What is the final amount paid?"
    ),
}


print("Calling:", API_URL)

try:
    response = requests.post(
        API_URL,
        json=payload,
        timeout=180,
    )
except requests.RequestException as error:
    print("Request failed:", error)
    raise SystemExit(1)


print("Status:", response.status_code)

try:
    result = response.json()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if response.status_code == 200:
        keys = set(result.keys())

        print("Exact keys:", keys == {"reasoning", "answer"})
        print(
            "Reasoning length:",
            len(result.get("reasoning", "")),
        )
        print(
            "Answer is integer:",
            (
                isinstance(result.get("answer"), int)
                and not isinstance(result.get("answer"), bool)
            ),
        )

except ValueError:
    print(response.text)