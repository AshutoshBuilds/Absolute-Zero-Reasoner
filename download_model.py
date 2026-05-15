import os
import argparse
import logging
import re
import math

from huggingface_hub import snapshot_download

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    RichHandler = None
    _RICH_AVAILABLE = False

logger = logging.getLogger(__name__)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _apply_cpu_cap(cpu_cap_percent: float) -> int:
    """Apply a soft CPU cap for background/model download workers."""
    cpu_count = os.cpu_count() or 1
    cap = max(1.0, min(100.0, float(cpu_cap_percent)))
    max_threads = max(1, math.floor(cpu_count * cap / 100.0))
    max_threads = min(cpu_count, max_threads)

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(max_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(max_threads)))
    except Exception:
        pass

    return max_threads


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text) if isinstance(text, str) else str(text)


def configure_logging(use_rich: bool = True) -> None:
    logger = logging.getLogger()
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if use_rich and _RICH_AVAILABLE:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)

        class _AnsiSafeFormatter(logging.Formatter):
            def format(self, record):
                if isinstance(record.msg, str):
                    record.msg = _strip_ansi(record.msg)
                if record.args:
                    record.args = tuple(_strip_ansi(arg) if isinstance(arg, str) else arg for arg in record.args)
                return super().format(record)

        handler.setFormatter(_AnsiSafeFormatter("%(name)s - %(levelname)s - %(message)s"))
        if Console is not None:
            Console(highlight=False, force_terminal=True)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to local cache.")
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-E4B",
        help="Model repository id to download.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Optional custom target directory; defaults to `models/<model-name>`.",
    )
    parser.add_argument(
        "--rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rich terminal rendering",
    )
    parser.add_argument(
        "--cpu-cap",
        type=float,
        default=20.0,
        help="CPU cap percentage (0-100) for this process",
    )
    args = parser.parse_args()

    configure_logging(use_rich=args.rich)
    _apply_cpu_cap(args.cpu_cap)
    model_id = args.model_id
    model_folder_name = os.path.basename(model_id.replace("\\", "/"))
    target_directory = args.target_dir or os.path.join(os.getcwd(), "models", model_folder_name)

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        logger.info(f"Created directory: {target_directory}")

    logger.info("Starting download of %s to %s...", model_id, target_directory)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_directory
        )
        logger.info("Successfully downloaded %s to %s", model_id, target_directory)
    except Exception as e:
        logger.error("An error occurred during download: %r", e)
        logger.error(
            "Please check the model ID, your internet connection, and permissions for the target directory."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
