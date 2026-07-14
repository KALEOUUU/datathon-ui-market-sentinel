import gc
import logging

from src.download_data import ensure_external_data
from src.features.build_feature_frame import build_feature_frame
from src.processing.build_dataset import build_dataset
from src.processing.eda import run_eda
from src.processing.label_governance import audit_and_prepare_annotation_package


def run() -> None:
    steps = (
        ("verify/download external data", ensure_external_data),
        ("build dataset", build_dataset),
        ("prepare labels", audit_and_prepare_annotation_package),
        ("build features", build_feature_frame),
        ("run EDA", run_eda),
    )
    for name, step in steps:
        logging.getLogger(__name__).info("Starting: %s", name)
        step()
        gc.collect()
    logging.getLogger(__name__).info("Steps 1-4 pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
