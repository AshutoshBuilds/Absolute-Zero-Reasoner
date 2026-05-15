#!/usr/bin/env python3
"""One-shot download of benchmark splits used by evaluate_benchmarks.py into local disk.

Respects ``AZR_BENCHMARK_DATA_ROOT`` (default ``benchmark_data`` under repo root).
Writes ``save_to_disk`` layouts and ``manifest.json`` with Hub revision pins.

Optional: ``AZR_BENCHMARK_PREFETCH_ONLINE=0`` skips all downloads (exit 0).

Unset ``AZR_BENCHMARK_OFFLINE`` when running this script (Hub access required).

MATH snapshot: ``hendrycks/competition_math`` is written under the same on-disk path as
eval uses, but rows are downloaded from the public ``EleutherAI/hendrycks_math`` mirror
(concatenated ``test`` splits) because the original Hub repo is gated.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import concatenate_datasets, load_dataset  # noqa: E402

from hf_benchmark_data import (  # noqa: E402
    benchmark_offline_env_requested,
    canonical_local_split_dir,
    hub_dataset_revision,
    hub_revision_env,
    resolve_benchmark_data_root,
    write_prefetch_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prefetch_benchmarks")


def _truthy_prefetch(value: str | None, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


BENCHMARK_SPECS = [
    {"dataset": "openai_humaneval", "config": None, "split": "test"},
    {"dataset": "mbpp", "config": "sanitized", "split": "test"},
    {"dataset": "gsm8k", "config": "main", "split": "test"},
    {"dataset": "hendrycks/competition_math", "config": None, "split": "test"},
]


def _hub_load_one(dataset: str, config: str | None, split: str, revision: str | None):
    if config:
        return load_dataset(dataset, config, split=split, revision=revision)
    return load_dataset(dataset, split=split, revision=revision)


# ``hendrycks/competition_math`` is gated on the Hub (403 without accepted terms).
# Public mirror: same ``problem`` / ``solution`` / ``level`` / ``type`` fields per row.
_MATH_MIRROR = "EleutherAI/hendrycks_math"
_MATH_MIRROR_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _load_competition_math_test_via_mirror(revision: str | None):
    chunks = []
    for subcfg in _MATH_MIRROR_CONFIGS:
        chunks.append(
            load_dataset(_MATH_MIRROR, subcfg, split="test", revision=revision),
        )
    return concatenate_datasets(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch AZR benchmark datasets to disk")
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Override AZR_BENCHMARK_DATA_ROOT for this run",
    )
    args = parser.parse_args()
    if args.root:
        os.environ["AZR_BENCHMARK_DATA_ROOT"] = args.root

    if benchmark_offline_env_requested():
        logger.error("AZR_BENCHMARK_OFFLINE=1 blocks Hub access. Unset it to prefetch.")
        return 2

    if not _truthy_prefetch(os.environ.get("AZR_BENCHMARK_PREFETCH_ONLINE"), True):
        logger.info("AZR_BENCHMARK_PREFETCH_ONLINE is off; skipping prefetch.")
        return 0

    root = resolve_benchmark_data_root()
    root.mkdir(parents=True, exist_ok=True)
    logger.info("Writing benchmark snapshots under %s", root)

    rev = hub_revision_env()
    manifest_rows: list[dict] = []
    for spec in BENCHMARK_SPECS:
        ds_name = spec["dataset"]
        cfg = spec["config"]
        split = spec["split"]
        out_dir = canonical_local_split_dir(root, ds_name, cfg, split)
        hub_rev = (
            hub_dataset_revision(_MATH_MIRROR)
            if ds_name == "hendrycks/competition_math" and cfg is None
            else hub_dataset_revision(ds_name)
        )
        logger.info(
            "Downloading %s (config=%s split=%s) env_revision=%s hub_sha=%s",
            ds_name,
            cfg,
            split,
            rev or "(default)",
            hub_rev or "unknown",
        )
        if ds_name == "hendrycks/competition_math" and cfg is None and split == "test":
            logger.info(
                "Using public mirror %s (concat test splits) for local snapshot path %s",
                _MATH_MIRROR,
                out_dir,
            )
            ds = _load_competition_math_test_via_mirror(rev)
        else:
            ds = _hub_load_one(ds_name, cfg, split, rev)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ds.save_to_disk(str(out_dir))
        logger.info("Saved to %s", out_dir)
        try:
            rel = str(out_dir.resolve().relative_to(root.resolve()))
        except Exception:
            rel = str(out_dir)
        manifest_rows.append(
            {
                "dataset": ds_name,
                "config": cfg,
                "split": split,
                "split_dir": rel,
                "revision": hub_rev,
                "hub_revision_env": rev,
                "source": "prefetch_script",
                "prefetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    write_prefetch_manifest(root, manifest_rows)
    logger.info("Wrote manifest: %s", root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
