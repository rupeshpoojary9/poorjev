"""Minimal client for TypeSafe's Jev API (api.typesafe.ai/v1/systemone)."""
import json
import os
import time
import urllib.request
import urllib.error

API_URL = "https://api.typesafe.ai/v1/systemone"


def _load_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1]
    raise RuntimeError("TYPESAFE_API_KEY not set (env var or .env file)")


class JevClient:
    def __init__(self, model="jev-latest", max_retries=5):
        self.key = _load_key()
        self.model = model
        self.max_retries = max_retries

    def predict(self, state, questions):
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode()
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                err_body = e.read().decode(errors="replace")
                if e.code in (429, 529) and attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Jev API error {e.code}: {err_body}") from None
        raise RuntimeError("Jev API: exhausted retries")
