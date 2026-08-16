from __future__ import annotations

import json
import subprocess
import time
import sys
from contextlib import contextmanager
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


@contextmanager
def port_forward_service(service_name: str, local_port: int, remote_port: int):
    process = subprocess.Popen(
        [
            "kubectl",
            "-n",
            "mlops",
            "port-forward",
            f"svc/{service_name}",
            f"{local_port}:{remote_port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            probe = subprocess.run(
                ["python3", "-c", f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{local_port}/healthz').read()"],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"port-forward to {service_name} did not become ready")
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    with port_forward_service("mlops-serving", 18080, 8080):
        print("HEALTHZ")
        print(
            run(
                "python3",
                "-c",
                "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:18080/healthz').read().decode())",
            )
        )

        print("PREDICT")
        predict_code = (
            "import json, urllib.request; "
            "req = urllib.request.Request("
            "'http://127.0.0.1:18080/predict', "
            "data=json.dumps({'user_id': 1001}).encode(), "
            "headers={'Content-Type': 'application/json'}, "
            "method='POST'"
            "); "
            "print(urllib.request.urlopen(req).read().decode())"
        )
        print(run("python3", "-c", predict_code))

        print("FAIRNESS")
        print(
            run(
                "python3",
                "-c",
                "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:18080/fairness').read().decode())",
            )
        )

        print("METRICS")
        metrics = run(
            "python3",
            "-c",
            "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:18080/metrics').read().decode())",
        )
        print(metrics[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
