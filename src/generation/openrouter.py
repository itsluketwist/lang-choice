"""Model runner for API-accessed models via OpenRouter."""

import os
from typing import Any

import openai

from src.generation.schemas import InferenceConfig, ModelConfig, ResponseData


# openrouter uses the openai-compatible api
_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterRunner:
    """Model runner for API-accessed models via OpenRouter.

    Uses the OpenAI-compatible API at https://openrouter.ai/api/v1.
    Requires OPENROUTER_API_KEY in the environment.

    reasoning_format on ModelConfig controls how the response is parsed:
      native     — reasoning is in choice.message.reasoning (DeepSeek R1, Qwen3 thinking)
      think_tags — reasoning is inline <think>...</think> in content (some open models)
      hidden     — reasoning happens server-side, not returned (OpenAI o-series)
      none       — non-reasoning model; treat full content as response
    """

    def __init__(self) -> None:
        """Initialise the OpenRouter client using OPENROUTER_API_KEY from env."""
        self._client = openai.OpenAI(
            base_url=_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ) -> ResponseData:
        """Call the OpenRouter API and return a ResponseData for a single sample.

        Resolves sampling parameters from the inference config, falling back to
        per-model defaults. top_k is silently ignored — not supported by the API.
        Returns a ResponseData with reasoning and response separated.
        """
        # resolve sampling params — inference config takes priority, then model defaults
        temperature = inference_config.temperature
        if temperature is None:
            temperature = model_config.defaults.get("temperature", 1.0)

        top_p = inference_config.top_p
        if top_p is None:
            top_p = model_config.defaults.get("top_p", 1.0)

        max_tokens = model_config.defaults.get("max_tokens", 32768)

        # build the api call kwargs
        call_kwargs: dict[str, Any] = {
            "model": model_config.model_path,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # pass any model-specific params (e.g. enable_thinking=True for Qwen3)
        if model_config.extra_params:
            call_kwargs["extra_body"] = model_config.extra_params

        api_response = self._client.chat.completions.create(**call_kwargs)

        choice = api_response.choices[0]
        # openrouter normalises reasoning into choice.message.reasoning for all
        # models that expose it — no need to handle provider-specific formats
        reasoning: str | None = getattr(choice.message, "reasoning", None) or None
        response: str = choice.message.content or ""

        # reasoning_tokens is only present for some models (e.g. OpenAI o-series)
        reasoning_tokens = getattr(api_response.usage, "reasoning_tokens", None)
        response_tokens = getattr(api_response.usage, "completion_tokens", None)

        return ResponseData(
            response=response,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            response_tokens=response_tokens,
        )
