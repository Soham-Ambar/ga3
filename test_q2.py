import os
import base64
import json
from pathlib import Path

import requests


IMAGE_PATH = Path("test_image.jpg")

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/answer-image",
)


def main():
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH.resolve()}")
        return

    image_base64 = base64.b64encode(
        IMAGE_PATH.read_bytes()
    ).decode("utf-8")

    payload = {
        "image_base64": image_base64,
        "question": "What is the total?",
    }

    print("Calling:", API_URL)

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        print("Request failed:", exc)
        return

    print("Status:", response.status_code)

    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()