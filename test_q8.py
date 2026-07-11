import json
import os

import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8765/semantic-search",
)


payload = {
    "query_id": "q0",
    "query": (
        "How do I automatically scale the number of pods "
        "when CPU usage rises?"
    ),
    "candidates": [
        (
            "A valley fold creases the paper so the crease "
            "points toward you."
        ),
        (
            "Horizontal Pod Autoscaler adds or removes pods "
            "based on observed CPU or custom metrics."
        ),
        (
            "Water the tomato plants thoroughly during hot weather."
        ),
        (
            "Kubernetes namespaces isolate groups of cluster resources."
        ),
        (
            "A CPU-based autoscaling policy increases application "
            "replicas when processor utilization crosses a threshold."
        ),
        (
            "The recipe requires flour, butter, sugar, and two eggs."
        ),
        (
            "Use a Deployment to define the desired pod template."
        ),
        (
            "Autoscaling enables workloads to expand capacity as "
            "resource demand increases."
        ),
        (
            "A violin has four strings and is played with a bow."
        ),
        (
            "DNS translates domain names into network addresses."
        ),
        (
            "A paperback book usually has a flexible paper cover."
        ),
        (
            "Container images package an application and its dependencies."
        ),
        (
            "Configure the HorizontalPodAutoscaler resource with a "
            "CPU utilization target and minimum and maximum replicas."
        ),
        (
            "A compass points toward magnetic north."
        ),
        (
            "A database index can accelerate lookup operations."
        ),
        (
            "Cloud storage keeps files on remote infrastructure."
        ),
        (
            "A bicycle chain transfers force to the rear wheel."
        ),
        (
            "TLS encrypts traffic between a client and server."
        ),
    ],
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
    print(
        json.dumps(
            response.json(),
            indent=2,
            ensure_ascii=False,
        )
    )
except ValueError:
    print(response.text)