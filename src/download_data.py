"""Reproducibly restore external inputs that are intentionally excluded from Git."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sh1zuka/indonesia-news-dataset-2024?datasetVersionNumber=2"
)
KAGGLE_MEMBER = "final_merge_dataset.csv"
KAGGLE_PATH = PROJECT_ROOT / "data/external/kaggle_news/final_merge_dataset.csv"
KAGGLE_SHA256 = "1fdab8112d69449768035f70eca6c6e90ca139f85644b7cb17648af77a111d8f"

HOAX_REVISION = "e6b3fd75ae0e2d7379418ebbe9263fc0214eecf0"
HOAX_FILENAME = "train-00000-of-00001-c843229dc636c69c.parquet"
HOAX_URL = (
    "https://huggingface.co/datasets/Rifky/indonesian-hoax-news/resolve/"
    f"{HOAX_REVISION}/data/{HOAX_FILENAME}"
)
HOAX_PATH = PROJECT_ROOT / f"data/external/indonesian_hoax_news/data/{HOAX_FILENAME}"
HOAX_SHA256 = "fbaacbf3c03376698b210d55c8e3814ee63edc1e998ea8f88c13f28ce6619fd0"

BUNDLED_INPUTS = {
    PROJECT_ROOT / "data/external/ihsg/ihsg_jkse_2024-07-31_2025-10-23.parquet": (
        "b2eea1e7bfc640a41aebd47a3228d33e93228cc5b00d24e094c98654cde4dcdd"
    ),
    PROJECT_ROOT / "data/external/inset_lexicon/positive.tsv": (
        "4d0dc2a6b2fe88a438fe5a61663f45c5cf60ebaeb019d4ca4d31c3315121c2ae"
    ),
    PROJECT_ROOT / "data/external/inset_lexicon/negative.tsv": (
        "b8d4b231077967d07ae07014b9dcea255301c2b17a2452c215e3d810a4ba3be4"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_existing(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path}: expected={expected_sha256}, actual={actual_sha256}. "
            "Delete the corrupted file and run the download command again."
        )
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path
    logger.info("Verified existing input: %s", display_path)
    return True


def _download(url: str, temporary_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "underdogs-datathon-data-bootstrap/1.0"})
    with urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _download_file(url: str, destination: Path, expected_sha256: str) -> None:
    if _verify_existing(destination, expected_sha256):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.download")
    temporary_path.unlink(missing_ok=True)
    logger.info("Downloading %s", destination)
    try:
        _download(url, temporary_path)
        _verify_existing(temporary_path, expected_sha256)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_zip_member(
    url: str,
    member: str,
    destination: Path,
    expected_sha256: str,
) -> None:
    if _verify_existing(destination, expected_sha256):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path = destination.parent / ".dataset-download.zip"
    temporary_path = destination.with_suffix(f"{destination.suffix}.download")
    archive_path.unlink(missing_ok=True)
    temporary_path.unlink(missing_ok=True)
    logger.info("Downloading %s", destination)
    try:
        _download(url, archive_path)
        with ZipFile(archive_path) as archive:
            if member not in archive.namelist():
                raise ValueError(f"Expected {member!r} in downloaded archive; found {archive.namelist()!r}.")
            with archive.open(member) as source, temporary_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        _verify_existing(temporary_path, expected_sha256)
        temporary_path.replace(destination)
    finally:
        archive_path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)


def ensure_external_data() -> dict[str, str]:
    """Verify bundled inputs and download only the two large, omitted datasets."""
    for path, checksum in BUNDLED_INPUTS.items():
        if not _verify_existing(path, checksum):
            raise FileNotFoundError(f"Bundled external input is missing from the repository: {path}")
    _download_zip_member(KAGGLE_URL, KAGGLE_MEMBER, KAGGLE_PATH, KAGGLE_SHA256)
    _download_file(HOAX_URL, HOAX_PATH, HOAX_SHA256)
    logger.info("All external inputs are ready.")
    return {
        "kaggle_news": str(KAGGLE_PATH),
        "indonesian_hoax_news": str(HOAX_PATH),
        "ihsg": str(next(path for path in BUNDLED_INPUTS if path.parent.name == "ihsg")),
        "inset": str(PROJECT_ROOT / "data/external/inset_lexicon"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_external_data()
