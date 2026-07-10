import base64
import json
import os
import sys

import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/audio-stats",
)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python test_q6.py path-to-audio-file")
        raise SystemExit(1)

    audio_path = sys.argv[1]

    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}")
        raise SystemExit(1)

    with open(audio_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(
            audio_file.read()
        ).decode("utf-8")

    payload = {
        "audio_id": "local-test",
        "audio_base64": audio_base64,
    }

    print(f"Calling: {API_URL}")

    response = requests.post(
        API_URL,
        json=payload,
        timeout=180,
    )

    print(f"Status: {response.status_code}")

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


if __name__ == "__main__":
    main()