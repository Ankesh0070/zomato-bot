"""
Configuration management for the Zomato AI Restaurant Recommendation System.

Loads settings from environment variables (.env file) and exposes them
as typed, validated properties. Centralizes all configuration in one place.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Try .env.example as a fallback for first-time setup guidance
    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        logging.warning(
            ".env file not found. Copy .env.example to .env and add your API keys.\n"
            f"  cp {env_example} {ENV_FILE}"
        )

# Load Streamlit secrets into environment variables as fallback
try:
    import streamlit as st
    for key in st.secrets:
        if key not in os.environ:
            os.environ[key] = str(st.secrets[key])
        if key.upper() not in os.environ:
            os.environ[key.upper()] = str(st.secrets[key])
except Exception:
    pass


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
class LLMConfig:
    """Configuration for the LLM provider (Groq)."""

    @staticmethod
    def groq_api_key() -> str | None:
        """Groq API key. Returns None if not set."""
        key = os.getenv("GROQ_API_KEY", "")
        if key and key != "your_groq_api_key_here":
            return key
        return None

    PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------
class DatasetConfig:
    """Configuration for the Hugging Face Zomato dataset."""

    HF_DATASET_NAME: str = os.getenv(
        "HF_DATASET_NAME",
        "ManikaSaini/zomato-restaurant-recommendation",
    )
    CACHE_DIR: Path = Path(
        os.getenv("DATASET_CACHE_DIR", str(PROJECT_ROOT / "data"))
    )


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------
class AppConfig:
    """General application settings."""

    TOP_K_CANDIDATES: int = int(os.getenv("TOP_K_CANDIDATES", "20"))
    MAX_RECOMMENDATIONS: int = int(os.getenv("MAX_RECOMMENDATIONS", "5"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PROJECT_ROOT: Path = PROJECT_ROOT


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("zomato_recommender")
    logger.setLevel(getattr(logging, AppConfig.LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        # Structured JSON formatter for logs (easy ingestion by log aggregators)
        import json as _json

        class _StructuredFormatter(logging.Formatter):
            def format(self, record):
                # ISO-like timestamp
                timestamp = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
                payload = {
                    "timestamp": timestamp,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "func": record.funcName,
                    "lineno": record.lineno,
                }
                # Include any extra record attributes if serializable
                for k, v in record.__dict__.items():
                    if k not in {
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                    }:
                        try:
                            _json.dumps(v)
                            payload[k] = v
                        except Exception:
                            payload[k] = str(v)
                return _json.dumps(payload, ensure_ascii=True)

        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Startup Validation
# ---------------------------------------------------------------------------
def validate_config() -> list[str]:
    """
    Validate that essential configuration is present.
    Returns a list of warning messages (empty if all is well).
    """
    warnings: list[str] = []

    if LLMConfig.groq_api_key() is None:
        warnings.append(
            "No Groq API key configured. Set GROQ_API_KEY in .env\n"
            "  Get a free key at: https://console.groq.com/keys"
        )

    if not DatasetConfig.CACHE_DIR.exists():
        try:
            DatasetConfig.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warnings.append(f"Cannot create dataset cache directory: {e}")

    return warnings


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Zomato AI Recommender — Configuration Check")
    print("=" * 60)
    print(f"  Project Root   : {PROJECT_ROOT}")
    print(f"  Groq Key       : {'[OK] Set' if LLMConfig.groq_api_key() else '[X] Not set'}")
    print(f"  LLM Provider   : {LLMConfig.PROVIDER}")
    print(f"  LLM Model      : {LLMConfig.MODEL}")
    print(f"  Temperature    : {LLMConfig.TEMPERATURE}")
    print(f"  Dataset        : {DatasetConfig.HF_DATASET_NAME}")
    print(f"  Cache Dir      : {DatasetConfig.CACHE_DIR}")
    print(f"  Top-K          : {AppConfig.TOP_K_CANDIDATES}")
    print(f"  Max Recs       : {AppConfig.MAX_RECOMMENDATIONS}")
    print(f"  Log Level      : {AppConfig.LOG_LEVEL}")
    print("=" * 60)

    issues = validate_config()
    if issues:
        print("\n  Warnings:")
        for w in issues:
            print(f"  - {w}")
    else:
        print("\n  All configuration looks good!")
