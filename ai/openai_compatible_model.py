"""Generic OpenAI-style /chat/completions backend. This is the 'plug in Kuku's API'
adapter: it works with OpenAI, OpenRouter, Groq, Together, Ollama, vLLM, LM Studio or
any gateway exposing the same protocol - configure base URL, key and model name only."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from ai.base_model import BaseLLM, EpisodeRequest, extract_json
from ai.prompts import SYSTEM_PROMPT, build_user_prompt
from utils.http import post_json


class OpenAICompatibleModel(BaseLLM):
    provider = "openai_compatible"

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.2, transport: Callable[..., dict] = post_json, **_: Any):
        self.base_url = (base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY", "")
        self.model = model or os.getenv("OPENAI_COMPAT_MODEL", "")
        if not (self.base_url and self.model):
            raise ValueError("OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_MODEL must be set")
        self.temperature = temperature
        self._post = transport

    def evaluate_episode(self, request: EpisodeRequest) -> List[Dict[str, Any]]:
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": build_user_prompt(request)}]}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self._post(f"{self.base_url}/chat/completions", payload, headers)
        try:
            text = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"unexpected chat-completions response shape: {str(resp)[:200]}") from e
        return extract_json(text)
