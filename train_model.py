"""Project training entrypoint.

Runs the end-to-end training workflow implemented in ``census.train_model``:
data loading, splitting, cross-validation model selection, and artifact
persistence (pipeline, estimator, and categorical encoder).
"""

from census.configure_logging import configure_logging
from census.train_model import run_training_pipeline


def main() -> None:
    configure_logging("logs/training_log", "training")
    run_training_pipeline()


if __name__ == "__main__":
    main()
