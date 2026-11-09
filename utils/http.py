"""Tiny stdlib HTTP JSON client with retry/backoff (keeps dependencies at zero)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional


class HTTPError(RuntimeError):
    pass


def post_json(url: str, payload: dict, headers: Optional[Dict[str, str]] = None,
              timeout: float = 90.0, retries: int = 3, backoff: float = 2.0,
              sleep: Callable[[float], None] = time.sleep) -> dict:
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # 429 / 5xx are worth retrying
            detail = e.read().decode("utf-8", "replace")[:300]
            last = HTTPError(f"HTTP {e.code}: {detail}")
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = HTTPError(f"{type(e).__name__}: {e}")
        if attempt < retries:
            sleep(backoff * (2 ** attempt))
    raise last or HTTPError("request failed")
