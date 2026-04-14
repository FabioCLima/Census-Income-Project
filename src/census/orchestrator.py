"""CLI orchestrator for the Census Income ML pipeline.

This module provides one command-line entrypoint (argparse-based) to run the
main project stages individually or end-to-end:

- ``prepare-data``: raw CSV -> cleaned CSV
- ``train``: cross-validation selection + model artifact persistence
- ``slice``: slice-level evaluation and ``model/slice_output.txt`` generation
- ``all``: run data preparation, training, and slice evaluation in sequence
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from loguru import logger

from census.configure_logging import configure_logging
from census.data_loader import CATEGORICAL_FEATURES, clean_raw_input, save_cleaned_data
from census.slicing import DEFAULT_SLICE_FEATURES, run_slice_pipeline
from census.train_model import run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs" / "orchestrator_log"
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "data_raw" / "census.csv"


def prepare_data_pipeline(
    *,
    raw_path: Path | str = DEFAULT_RAW_PATH,
    cleaned_path: Path | str | None = None,
) -> Path:
    """Execute raw->cleaned data preparation and persist the cleaned dataset."""
    source = Path(raw_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{source}'. "
            "Provide --raw-path or place census.csv under data/data_raw/."
        )

    logger.info("prepare_data_pipeline | loading raw CSV from {}", source)
    raw_df = pd.read_csv(source)
    cleaned_df = clean_raw_input(raw_df)
    output = save_cleaned_data(cleaned_df, cleaned_path)
    logger.info("prepare_data_pipeline complete | cleaned_path={}", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create and return the argparse parser for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        prog="census-orchestrator",
        description="Orchestrate the Census Income ML pipeline stages.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser(
        "prepare-data",
        help="Run raw->cleaned data preparation.",
    )
    p_prepare.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to raw census CSV (default: data/data_raw/census.csv).",
    )
    p_prepare.add_argument(
        "--cleaned-path",
        type=Path,
        default=None,
        help="Optional output path for cleaned CSV.",
    )

    subparsers.add_parser(
        "train",
        help="Run model training pipeline and persist artifacts.",
    )

    p_slice = subparsers.add_parser(
        "slice",
        help="Run slice-level evaluation and write model/slice_output.txt.",
    )
    p_slice.add_argument(
        "--slice-features",
        nargs="+",
        default=list(DEFAULT_SLICE_FEATURES),
        help=(
            "Categorical features for slice evaluation "
            f"(available: {', '.join(CATEGORICAL_FEATURES)})."
        ),
    )

    p_all = subparsers.add_parser(
        "all",
        help="Run prepare-data, train, and slice stages in sequence.",
    )
    p_all.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to raw census CSV (default: data/data_raw/census.csv).",
    )
    p_all.add_argument(
        "--cleaned-path",
        type=Path,
        default=None,
        help="Optional output path for cleaned CSV.",
    )
    p_all.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip raw->cleaned preparation and use existing cleaned data.",
    )
    p_all.add_argument(
        "--slice-features",
        nargs="+",
        default=list(DEFAULT_SLICE_FEATURES),
        help=(
            "Categorical features for slice evaluation "
            f"(available: {', '.join(CATEGORICAL_FEATURES)})."
        ),
    )

    return parser


def _normalise_slice_features(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and deduplicate slice features preserving input order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for argparse-driven orchestration."""
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(LOG_DIR, "orchestrator")

    if args.command == "prepare-data":
        prepare_data_pipeline(raw_path=args.raw_path, cleaned_path=args.cleaned_path)
        return 0

    if args.command == "train":
        run_training_pipeline()
        return 0

    if args.command == "slice":
        slice_features = _normalise_slice_features(args.slice_features)
        run_slice_pipeline(slice_features=slice_features)
        return 0

    if args.command == "all":
        if not args.skip_data:
            prepare_data_pipeline(
                raw_path=args.raw_path,
                cleaned_path=args.cleaned_path,
            )
        run_training_pipeline()
        slice_features = _normalise_slice_features(args.slice_features)
        run_slice_pipeline(slice_features=slice_features)
        logger.info("Pipeline orchestration complete | command=all")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
