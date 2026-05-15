"""Local-first Hugging Face benchmark dataset loading for AZR evaluation.

Snapshots live under ``AZR_BENCHMARK_DATA_ROOT`` (default ``benchmark_data`` under
the repo root). When a split exists from ``save_to_disk``, ``load_from_disk`` is
used so eval avoids Hub traffic.

Environment:
- ``AZR_BENCHMARK_ALLOW_ONLINE`` — default ``0``: missing local snapshots raise a
  clear error. Set ``1`` to allow ``load_dataset`` and optional auto-cache.
- Legacy: if ``AZR_BENCHMARK_ALLOW_ONLINE`` is unset and
  ``AZR_BENCHMARK_ALLOW_ONLINE_LOAD`` is set, the legacy variable is honored.
- ``AZR_BENCHMARK_OFFLINE`` — when ``1``, sets ``HF_DATASETS_OFFLINE`` and
  ``HF_HUB_OFFLINE`` (call ``apply_benchmark_offline_env()`` from benchmark entrypoints).
- ``AZR_BENCHMARK_HUB_REVISION`` — optional revision passed to Hub ``load_dataset``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

def _get_repo_root(current_path: Path) -> Path:
    p = current_path
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return current_path

REPO_ROOT = _get_repo_root(Path(__file__).resolve().parent)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on", "y"):
        return True
    if v in ("0", "false", "no", "off", "n"):
        return False
    return default


def resolve_benchmark_data_root() -> Path:
    """Root directory for on-disk benchmark snapshots."""
    raw = (os.environ.get("AZR_BENCHMARK_DATA_ROOT") or "benchmark_data").strip()
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def allow_online_fetch() -> bool:
    """True if Hub download is allowed when a local snapshot is missing."""
    explicit = os.environ.get("AZR_BENCHMARK_ALLOW_ONLINE")
    if explicit is not None and str(explicit).strip() != "":
        return _parse_bool(explicit, False)
    legacy = os.environ.get("AZR_BENCHMARK_ALLOW_ONLINE_LOAD")
    if legacy is not None and str(legacy).strip() != "":
        return _parse_bool(legacy, False)
    return False


def benchmark_offline_env_requested() -> bool:
    return _parse_bool(os.environ.get("AZR_BENCHMARK_OFFLINE"), False)


def hub_revision_env() -> Optional[str]:
    r = (os.environ.get("AZR_BENCHMARK_HUB_REVISION") or "").strip()
    return r or None


def apply_benchmark_offline_env() -> None:
    if benchmark_offline_env_requested():
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"


def canonical_local_split_dir(root: Path, dataset_name: str, config: Optional[str], split: str) -> Path:
    """Directory for one split (``save_to_disk``), e.g. ``openai_humaneval/test``."""
    slug = dataset_name.replace("/", "__")
    parts: List[str] = [slug]
    if config:
        parts.append(str(config))
    parts.append(str(split))
    return root.joinpath(*parts)


def _looks_like_saved_dataset(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "state.json").exists() or (path / "dataset_info.json").exists()


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def log_manifest_for_path(
    root: Path,
    local_split_dir: Path,
    dataset_name: str,
    config: Optional[str],
    split: str,
    logger: Optional[logging.Logger],
) -> None:
    log = logger or logging.getLogger(__name__)
    mp = _manifest_path(root)
    if not mp.is_file():
        return
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Could not read benchmark manifest %s: %s", mp, e)
        return
    entries = data.get("datasets") or data.get("entries") or []
    try:
        rel = str(local_split_dir.resolve().relative_to(root.resolve()))
    except Exception:
        rel = str(local_split_dir)
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        if ent.get("split_dir") == rel or ent.get("path") == rel:
            log.info(
                "Benchmark manifest: dataset=%s config=%s split=%s revision=%s prefetched=%s",
                ent.get("dataset") or dataset_name,
                ent.get("config", config),
                ent.get("split") or split,
                ent.get("revision", "unknown"),
                ent.get("prefetched_at", "unknown"),
            )
            return
    log.info("Benchmark: loaded local split %s (no manifest match)", rel)


def hub_dataset_revision(dataset_name: str) -> Optional[str]:
    """Resolve current Hub git SHA for a dataset repo (manifest pinning)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(dataset_name, repo_type="dataset")
        sha = getattr(info, "sha", None)
        return str(sha) if sha else None
    except Exception:
        return None


def load_azr_benchmark_split(
    dataset_name: str,
    split: str,
    config: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Load a split: local ``save_to_disk`` first, else Hub per ``allow_online_fetch``."""
    from datasets import load_dataset, load_from_disk

    log = logger or logging.getLogger(__name__)
    rev = hub_revision_env()
    root = resolve_benchmark_data_root()
    local_dir = canonical_local_split_dir(root, dataset_name, config, split)

    if _looks_like_saved_dataset(local_dir):
        log.info(
            "Benchmark: load_from_disk %s (dataset=%s config=%s split=%s)",
            local_dir,
            dataset_name,
            config,
            split,
        )
        log_manifest_for_path(root, local_dir, dataset_name, config, split, log)
        return load_from_disk(str(local_dir))

    if not allow_online_fetch():
        raise FileNotFoundError(
            f"Benchmark snapshot not found at {local_dir}. "
            f"Run: python scripts/prefetch_benchmark_datasets.py "
            f"(or set AZR_BENCHMARK_DATA_ROOT). "
            f"To download from the Hub, set AZR_BENCHMARK_ALLOW_ONLINE=1."
        )

    log.warning(
        "Benchmark: downloading from Hub dataset=%s config=%s split=%s revision=%s",
        dataset_name,
        config,
        split,
        rev or "(default)",
    )
    if config:
        ds = load_dataset(dataset_name, config, split=split, revision=rev)
    else:
        ds = load_dataset(dataset_name, split=split, revision=rev)

    try:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(local_dir))
        log.info("Benchmark: cached Hub dataset to %s", local_dir)
        _append_runtime_manifest(root, dataset_name, config, split, local_dir, log)
    except Exception as e:
        log.warning("Benchmark: could not cache to disk (%s): %s", local_dir, e)

    return ds


def _append_runtime_manifest(
    root: Path,
    dataset_name: str,
    config: Optional[str],
    split: str,
    local_dir: Path,
    log: logging.Logger,
) -> None:
    try:
        rel = str(local_dir.resolve().relative_to(root.resolve()))
    except Exception:
        rel = str(local_dir)
    rev_pin = hub_revision_env()
    entry = {
        "dataset": dataset_name,
        "config": config,
        "split": split,
        "split_dir": rel,
        "revision": rev_pin,
        "source": "hub_runtime_cache",
        "prefetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _merge_manifest_entry(root, entry, log)


def _merge_manifest_entry(root: Path, entry: Dict[str, Any], log: logging.Logger) -> None:
    mp = _manifest_path(root)
    data: Dict[str, Any] = {"version": 1, "datasets": []}
    if mp.is_file():
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            if "datasets" not in data:
                data["datasets"] = data.get("entries", [])
        except Exception:
            pass
    rows: List[Dict[str, Any]] = list(data.get("datasets") or [])
    key = (entry.get("dataset"), entry.get("config"), entry.get("split"))
    kept = [r for r in rows if (r.get("dataset"), r.get("config"), r.get("split")) != key]
    kept.append(entry)
    data["datasets"] = kept
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        root.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Could not update manifest %s: %s", mp, e)


def write_prefetch_manifest(root: Path, entries: List[Dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mp = _manifest_path(root)
    payload = {
        "version": 1,
        "prefetched_at": datetime.now(timezone.utc).isoformat(),
        "datasets": entries,
    }
    mp.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Optional named helpers
def load_openai_humaneval_test(**kwargs):
    return load_azr_benchmark_split("openai_humaneval", "test", None, **kwargs)


def load_mbpp_sanitized_test(**kwargs):
    return load_azr_benchmark_split("mbpp", "test", "sanitized", **kwargs)


def load_gsm8k_main_test(**kwargs):
    return load_azr_benchmark_split("gsm8k", "test", "main", **kwargs)


def load_competition_math_test(**kwargs):
    return load_azr_benchmark_split("hendrycks/competition_math", "test", None, **kwargs)
