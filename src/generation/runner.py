"""Model runner for direct provider API access via llm_cgr."""

import random
import time

from llm_cgr.llm.clients import PROVIDER_MAP

from src.generation.schemas import InferenceConfig, ModelConfig, ResponseData


# retry settings for transient API failures (network errors, rate limits, etc.)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt with added jitter


class ModelRunner:
    """Model runner that calls provider APIs directly via llm_cgr.

    Uses llm_cgr's PROVIDER_MAP to select the right client for each provider
    (e.g. Anthropic_LLM for "anthropic", DeepSeek_LLM for "deepseek"). The
    provider key in ModelConfig must match a key in PROVIDER_MAP.

    Transient failures (network errors, rate limits) are retried up to
    _MAX_RETRIES times with exponential backoff. Hard errors are raised immediately.
    """

    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        """Initialise the llm_cgr client for the model's provider.

        Passes enable_reasoning to clients that support it (e.g. DeepSeek_LLM).
        Clients that do not accept the flag (e.g. Anthropic_LLM) are initialised
        without it — reasoning will return None for those models.
        """
        client_class = PROVIDER_MAP[model_config.provider]
        try:
            # not all clients accept enable_reasoning; try with it first
            self._client = client_class(
                enable_reasoning=model_config.enable_reasoning,
            )
        except TypeError:
            self._client = client_class()

    def generate(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ) -> ResponseData:
        """Call the provider API and return a ResponseData for a single sample.

        Resolves sampling parameters from the inference config, falling back to
        per-model defaults. Retries on transient failures with exponential backoff.

        Returns a ResponseData with response and reasoning (reasoning is None for
        models that do not expose chain-of-thought).
        """
        # resolve sampling params — inference config takes priority, then model defaults
        temperature = inference_config.temperature
        if temperature is None:
            temperature = model_config.defaults.get("temperature", 1.0)

        top_p = inference_config.top_p
        if top_p is None:
            top_p = model_config.defaults.get("top_p", 1.0)

        max_tokens = model_config.defaults.get("max_tokens", 32768)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                # _get_response() is called directly because llm_cgr's public
                # generate() only accepts a single user string, but we need to
                # pass multi-turn messages (optional context + task prompt)
                response, reasoning = self._client._get_response(
                    model=model_config.model_path,
                    input=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                return ResponseData(
                    response=response,
                    reasoning=reasoning or None,
                )

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    # exponential backoff with jitter: 2s, 4s, 8s (±1s noise)
                    delay = _RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)

        raise RuntimeError(
            f"API call failed after {_MAX_RETRIES} attempts: {last_error}",
        ) from last_error
