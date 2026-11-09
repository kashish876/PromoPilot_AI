"""Model selection is CONFIGURATION. Add a backend by registering a class here."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from ai.base_model import BaseLLM
from ai.gemini_model import GeminiModel
from ai.openai_compatible_model import OpenAICompatibleModel
from ai.rule_based_model import RuleBasedModel
from config import model_settings_from_env

REGISTRY: Dict[str, Type[BaseLLM]] = {
    "rule_based": RuleBasedModel,
    "gemini": GeminiModel,
    "openai_compatible": OpenAICompatibleModel,
}
ALIASES = {"mock": "rule_based", "rules": "rule_based", "free": "rule_based",
           "google": "gemini", "openai": "openai_compatible", "kuku": "openai_compatible"}


def register_model(name: str, cls: Type[BaseLLM]) -> None:
    """Plug in a new backend at runtime (e.g. Kuku's own API) without touching anything else."""
    REGISTRY[name] = cls


def available_models() -> list:
    return ["auto"] + sorted(REGISTRY)


def create_model(provider: Optional[str] = None, **overrides: Any) -> BaseLLM:
    s = {**model_settings_from_env(), **{k: v for k, v in overrides.items() if v not in (None, "")}}
    name = (provider or s["provider"] or "auto").lower()
    name = ALIASES.get(name, name)
    if name == "auto":
        name = "gemini" if s.get("gemini_api_key") else ("openai_compatible" if s.get("openai_base_url") and s.get("openai_model") else "rule_based")
    if name not in REGISTRY:
        raise ValueError(f"unknown model '{name}'. Available: {available_models()}")
    cls = REGISTRY[name]
    if name == "gemini":
        return cls(api_key=s["gemini_api_key"], model=s["gemini_model"])
    if name == "openai_compatible":
        return cls(base_url=s["openai_base_url"], api_key=s["openai_api_key"], model=s["openai_model"])
    return cls(**{k: v for k, v in overrides.items() if k not in ("provider",)}) if name not in ("rule_based",) else cls()
