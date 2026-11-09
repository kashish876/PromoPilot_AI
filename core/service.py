"""Single entry point used by BOTH the CLI and the Streamlit UI, so they cannot diverge."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ai.base_model import BaseLLM
from ai.model_factory import create_model
from config import PipelineConfig
from core.loader import load_scripts
from core.pipeline import ProgressFn, PromoPipeline, RunResult
from utils.logging_setup import get_logger


def run_promopilot(sources: Iterable[Any], model: Optional[str] = None, model_overrides: Optional[Dict[str, Any]] = None,
                   cfg: Optional[PipelineConfig] = None, only_episodes: Optional[Sequence[int]] = None,
                   progress: Optional[ProgressFn] = None, log_file: Optional[str] = None,
                   console: bool = False, model_instance: Optional[BaseLLM] = None) -> RunResult:
    sink: List[str] = []
    logger = get_logger("promopilot", Path(log_file) if log_file else None, sink, console)
    scripts, skipped = load_scripts(list(sources))
    logger.info("input discovery: %d screenplay file(s) found, %d skipped", len(scripts), len(skipped))
    llm = model_instance or create_model(model, **(model_overrides or {}))
    logger.info("model: %s / %s", llm.describe()["provider"], llm.describe()["model"])
    pipe = PromoPipeline(llm, cfg or PipelineConfig.from_env(), logger=logger, progress=progress)
    result = pipe.run(scripts, only_episodes=only_episodes, skipped_notes=skipped)
    result.logs = sink
    return result
