"""
LLM client module — Groq API integration.

Provides the LLM client for calling Groq's ultra-fast inference API
with retry logic, rate limit handling, and error management.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from src.config import LLMConfig

logger = logging.getLogger("zomato_recommender.recommendation.llm_client")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the LLM API call times out."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM API returns a rate limit error (429)."""
    pass


class LLMAuthError(LLMError):
    """Raised when the API key is missing or invalid."""
    pass


# ---------------------------------------------------------------------------
# Abstract Base Client
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        Args:
            user_prompt: The main user-facing prompt.
            system_prompt: System-level instructions.

        Returns:
            Raw text response from the LLM.

        Raises:
            LLMError: On any API failure.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this client is configured and ready to use."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


# ---------------------------------------------------------------------------
# Groq Client
# ---------------------------------------------------------------------------
class GroqClient(BaseLLMClient):
    """Groq API client — ultra-fast LLM inference."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or LLMConfig.groq_api_key()
        self._client = None

    def is_available(self) -> bool:
        return self._api_key is not None

    def _get_client(self):
        """Lazy-initialize the Groq client."""
        if self._client is None:
            if not self._api_key:
                raise LLMAuthError(
                    "Groq API key not set. Add GROQ_API_KEY to your .env file."
                )
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """Call Groq API and return the response text."""
        try:
            client = self._get_client()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content
            if not text:
                raise LLMError("Groq returned an empty response.")

            logger.debug(f"Groq response length: {len(text)} chars")
            return text

        except LLMAuthError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "quota" in error_str:
                raise LLMRateLimitError(f"Groq rate limit: {e}") from e
            if "timeout" in error_str or "deadline" in error_str:
                raise LLMTimeoutError(f"Groq timeout: {e}") from e
            if "401" in str(e) or "invalid" in error_str and "key" in error_str:
                raise LLMAuthError(f"Groq auth error: {e}") from e
            raise LLMError(f"Groq error: {e}") from e


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------
def call_with_retry(
    client: BaseLLMClient,
    user_prompt: str,
    system_prompt: str = "",
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """
    Call an LLM client with exponential backoff retry logic.

    Retries on timeout and rate limit errors. Does NOT retry on auth errors.

    Args:
        client: The LLM client to use.
        user_prompt: User prompt text.
        system_prompt: System prompt text.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).

    Returns:
        Raw text response from the LLM.

    Raises:
        LLMError: If all retries are exhausted.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"LLM call attempt {attempt}/{max_retries} via {client}")
            response = client.generate(user_prompt, system_prompt)
            logger.info(f"LLM call succeeded on attempt {attempt}.")
            return response

        except LLMAuthError:
            # Don't retry auth errors
            raise

        except (LLMRateLimitError, LLMTimeoutError) as e:
            last_error = e
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Attempt {attempt} failed ({type(e).__name__}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            if attempt < max_retries:
                time.sleep(delay)

        except LLMError as e:
            last_error = e
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Attempt {attempt} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            if attempt < max_retries:
                time.sleep(delay)

    raise LLMError(
        f"All {max_retries} attempts failed. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_llm_client(
    model: str | None = None,
) -> BaseLLMClient:
    """
    Create the Groq LLM client.

    Args:
        model: Model name override. Defaults to config value.

    Returns:
        Configured GroqClient instance.

    Raises:
        LLMAuthError: If no Groq API key is available.
    """
    model = model or LLMConfig.MODEL
    temperature = LLMConfig.TEMPERATURE
    max_tokens = LLMConfig.MAX_TOKENS

    client = GroqClient(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if client.is_available():
        return client

    raise LLMAuthError(
        "Groq API key not configured. Set GROQ_API_KEY in your .env file.\n"
        "Get a free key at: https://console.groq.com/keys"
    )
