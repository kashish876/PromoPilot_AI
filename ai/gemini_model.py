"""Google Gemini backend (free tier works). Key comes from GEMINI_API_KEY."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from ai.base_model import BaseLLM, EpisodeRequest, extract_json
from ai.prompts import SYSTEM_PROMPT, build_user_prompt
from utils.http import post_json

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiModel(BaseLLM):
    provider = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.2, transport: Callable[..., dict] = post_json, **_: Any):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self._post = transport          # injectable -> unit-testable without network

    def evaluate_episode(self, request: EpisodeRequest) -> List[Dict[str, Any]]:
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": build_user_prompt(request)}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        resp = self._post(ENDPOINT.format(model=self.model), payload, {"x-goog-api-key": self.api_key})
        try:
            text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"unexpected Gemini response shape: {str(resp)[:200]}") from e
        return extract_json(text)
