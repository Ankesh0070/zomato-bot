"""
Data ingestion module — download and cache the Zomato dataset.

Downloads the Zomato restaurant dataset from Hugging Face and caches it
locally as a Parquet file for fast subsequent loads.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.config import DatasetConfig

logger = logging.getLogger("zomato_recommender.data.ingestion")

# Cache file name
CACHE_FILENAME = "zomato_dataset.parquet"


def _get_cache_path() -> Path:
    """Return the full path to the cached Parquet file."""
    cache_dir = DatasetConfig.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / CACHE_FILENAME


def download_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download the Zomato dataset from Hugging Face.

    If a cached Parquet file exists locally, loads from cache unless
    force_refresh is True. On download failure, falls back to cache.

    Args:
        force_refresh: If True, re-download even if cache exists.

    Returns:
        Raw Pandas DataFrame with the Zomato dataset.

    Raises:
        RuntimeError: If download fails and no cache is available.
    """
    cache_path = _get_cache_path()

    # Try loading from cache first (unless force refresh)
    if not force_refresh and cache_path.exists():
        logger.info(f"Loading dataset from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(df):,} restaurants from cache.")
            return df
        except Exception as e:
            logger.warning(f"Cache file is corrupt, re-downloading: {e}")

    # Download from Hugging Face
    logger.info(f"Downloading dataset from Hugging Face: {DatasetConfig.HF_DATASET_NAME}")
    df = _download_from_huggingface()

    # Cache the downloaded data
    _save_to_cache(df, cache_path)

    return df


def _download_from_huggingface(max_retries: int = 3) -> pd.DataFrame:
    """
    Download dataset from Hugging Face with retry logic.

    Args:
        max_retries: Maximum number of download attempts.

    Returns:
        Raw Pandas DataFrame.

    Raises:
        RuntimeError: If all download attempts fail and no cache exists.
    """
    # Lazy import to avoid slow startup if cached
    from datasets import load_dataset

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_retries}...")
            dataset = load_dataset(
                DatasetConfig.HF_DATASET_NAME,
                split="train",
            )
            df = dataset.to_pandas()
            logger.info(f"Successfully downloaded {len(df):,} records.")
            return df

        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
            logger.warning(
                f"Download attempt {attempt} failed: {e}. "
                f"Retrying in {wait_time}s..."
            )
            if attempt < max_retries:
                time.sleep(wait_time)

    # All retries exhausted — try cache as last resort
    cache_path = _get_cache_path()
    if cache_path.exists():
        logger.warning(
            "All download attempts failed. Falling back to cached data."
        )
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    raise RuntimeError(
        f"Failed to download dataset after {max_retries} attempts. "
        f"Last error: {last_error}\n"
        f"Please check your internet connection and try again."
    )


def _save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save DataFrame to local Parquet cache."""
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Dataset cached to: {cache_path}")
    except Exception as e:
        logger.warning(f"Could not cache dataset: {e}")


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return summary statistics about the loaded dataset.

    Useful for debugging and UI display.

    Args:
        df: The raw or processed DataFrame.

    Returns:
        Dictionary with dataset summary info.
    """
    info = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "null_counts": df.isnull().sum().to_dict(),
    }

    return info


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.config import setup_logging
    setup_logging()

    print("Downloading Zomato dataset...")
    df = download_dataset()
    print(f"\n✅ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"\nColumns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    info = get_dataset_info(df)
    print(f"\nMemory usage: {info['memory_usage_mb']} MB")
    print(f"Null counts: {info['null_counts']}")
