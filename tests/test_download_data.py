import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.download_data import _download_file, _download_zip_member


class TestExternalDataBootstrap(unittest.TestCase):
    def test_download_file_verifies_checksum_and_is_idempotent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "output/data.bin"
            source.write_bytes(b"verified-content")
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()

            _download_file(source.as_uri(), destination, checksum)
            _download_file(source.as_uri(), destination, checksum)

            self.assertEqual(destination.read_bytes(), b"verified-content")
            self.assertFalse(destination.with_suffix(".bin.download").exists())

    def test_download_zip_extracts_only_expected_member(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "source.zip"
            destination = root / "output/dataset.csv"
            content = b"title,content\nA,B\n"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("dataset.csv", content)
                archive.writestr("ignored.txt", b"ignored")
            checksum = hashlib.sha256(content).hexdigest()

            _download_zip_member(archive_path.as_uri(), "dataset.csv", destination, checksum)

            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse((destination.parent / "ignored.txt").exists())


if __name__ == "__main__":
    unittest.main()
