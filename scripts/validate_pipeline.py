from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime


def run(*args: str) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
    return completed.stdout.strip()


def current_serving_pod() -> str:
    pods = json.loads(
        run(
            "kubectl",
            "-n",
            "mlops",
            "get",
            "pods",
            "-l",
            "app=mlops-serving",
            "-o",
            "json",
        )
    )
    running = [pod for pod in pods.get("items", []) if pod.get("status", {}).get("phase") == "Running"]
    if not running:
        raise RuntimeError("no running serving pod found")

    def sort_key(pod: dict) -> tuple[datetime, str]:
        ts = pod.get("metadata", {}).get("creationTimestamp", "")
        created = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.min
        return created, pod.get("metadata", {}).get("name", "")

    return max(running, key=sort_key)["metadata"]["name"]


def exec_in_serving(*args: str) -> str:
    pod_name = current_serving_pod()
    return run(
        "kubectl",
        "-n",
        "mlops",
        "exec",
        f"pod/{pod_name}",
        "--",
        *args,
    )


def main() -> int:
    print("HEALTHZ")
    print(exec_in_serving(
        "python3",
        "-c",
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/healthz').read().decode())",
    ))

    print("PREDICT")
    predict_code = (
        "import json, urllib.request; "
        "req = urllib.request.Request("
        "'http://127.0.0.1:8080/predict', "
        "data=json.dumps({'user_id': 1001}).encode(), "
        "headers={'Content-Type': 'application/json'}, "
        "method='POST'"
        "); "
        "print(urllib.request.urlopen(req).read().decode())"
    )
    print(exec_in_serving("python3", "-c", predict_code))

    print("FAIRNESS")
    print(exec_in_serving(
        "python3",
        "-c",
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/fairness').read().decode())",
    ))

    print("METRICS")
    metrics = exec_in_serving(
        "python3",
        "-c",
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/metrics').read().decode())",
    )
    print(metrics[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
