"""Download and extract the Zomato delivery operations dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from delivery_time import config

logger = logging.getLogger(__name__)

_ZIP_CONTENT_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",
    "binary/octet-stream",
}


class FetchError(RuntimeError):
    """Raised when the dataset cannot be downloaded, extracted, or validated."""


@dataclass(frozen=True)
class DownloadInfo:
    """Metadata for a downloaded archive."""

    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CsvInfo:
    """Metadata for a validated CSV file."""

    path: Path
    size_bytes: int
    row_count: int
    columns: tuple[str, ...]


def resolve_auth() -> httpx.BasicAuth | None:
    """Return Kaggle basic auth when credentials are configured."""
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    key = (os.getenv("KAGGLE_KEY") or "").strip()

    if not username and not key:
        return None
    if not username or not key:
        raise FetchError(
            "Set both KAGGLE_USERNAME and KAGGLE_KEY, or neither. "
            "The Zomato dataset currently downloads anonymously."
        )
    return httpx.BasicAuth(username, key)


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(config.CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def inspect_archive(path: Path) -> DownloadInfo:
    """Inspect an existing archive and return its size and checksum."""
    if not path.is_file():
        raise FetchError(f"Archive not found: {path}")
    if not zipfile.is_zipfile(path):
        raise FetchError(f"Existing archive is not a valid ZIP file: {path}")

    size, sha256 = _sha256(path)
    return DownloadInfo(path=path, size_bytes=size, sha256=sha256)


def download_archive(
    client: httpx.Client,
    url: str,
    destination: Path,
) -> DownloadInfo:
    """Stream a ZIP archive to disk and return its metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    digest = hashlib.sha256()
    size = 0

    try:
        with client.stream("GET", url) as response:
            if response.status_code in {401, 403}:
                raise FetchError(
                    "Kaggle rejected the download request. The dataset may require "
                    "authentication; set KAGGLE_USERNAME and KAGGLE_KEY in .env."
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise FetchError(
                    f"Kaggle download failed with HTTP {response.status_code}."
                ) from exc

            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in _ZIP_CONTENT_TYPES:
                raise FetchError(
                    "Expected a ZIP download from Kaggle, but received content type "
                    f"{content_type or 'unknown'!r}. Kaggle may have returned a login "
                    "or error page."
                )

            with part_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)

        if not zipfile.is_zipfile(part_path):
            raise FetchError("Downloaded file is not a valid ZIP archive.")

        os.replace(part_path, destination)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    logger.debug("Downloaded %s bytes to %s", size, destination)
    return DownloadInfo(path=destination, size_bytes=size, sha256=digest.hexdigest())


def extract_csv(
    archive_path: Path,
    destination: Path,
    *,
    force: bool = False,
) -> Path:
    """Extract the dataset CSV to a fixed, safe destination path."""
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            csv_members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".csv")
            ]
            if not csv_members:
                raise FetchError(f"No CSV file found in archive: {archive_path}")

            preferred = [
                member
                for member in csv_members
                if Path(member.filename).name == config.EXPECTED_CSV_FILENAME
            ]
            member = preferred[0] if preferred else csv_members[0]
            if len(csv_members) > 1:
                logger.warning(
                    "Archive contains multiple CSV files; extracting %s",
                    member.filename,
                )

            with archive.open(member) as source, part_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=config.CHUNK_SIZE)

        os.replace(part_path, destination)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    logger.debug("Extracted %s to %s", member.filename, destination)
    return destination


def validate_csv(csv_path: Path) -> CsvInfo:
    """Validate the CSV header and count its data rows."""
    if not csv_path.is_file():
        raise FetchError(f"CSV file not found: {csv_path}")

    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise FetchError(f"CSV file is empty: {csv_path}") from exc
            columns = tuple(header)
            row_count = sum(1 for row in reader if any(field.strip() for field in row))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise FetchError(f"CSV could not be parsed as UTF-8: {csv_path}") from exc

    if config.TARGET_COLUMN not in columns:
        raise FetchError(
            f"Target column {config.TARGET_COLUMN!r} not found in {csv_path}. "
            f"Found columns: {', '.join(columns)}"
        )

    if columns != config.EXPECTED_COLUMNS:
        missing = [name for name in config.EXPECTED_COLUMNS if name not in columns]
        extra = [name for name in columns if name not in config.EXPECTED_COLUMNS]
        if missing:
            logger.warning("CSV is missing expected columns: %s", ", ".join(missing))
        if extra:
            logger.warning("CSV has unexpected columns: %s", ", ".join(extra))
        if not missing and not extra:
            logger.warning("CSV columns are in a different order than expected.")

    return CsvInfo(
        path=csv_path,
        size_bytes=csv_path.stat().st_size,
        row_count=row_count,
        columns=columns,
    )


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_manifest(
    *,
    output_dir: Path,
    archive: DownloadInfo,
    csv_info: CsvInfo,
) -> Path:
    """Write a JSON manifest describing the fetched dataset."""
    manifest_path = output_dir / config.MANIFEST_FILENAME
    part_path = manifest_path.with_name(f"{manifest_path.name}.part")
    downloaded_at = datetime.fromtimestamp(archive.path.stat().st_mtime, tz=UTC)
    payload: dict[str, Any] = {
        "dataset": config.DATASET_SLUG,
        "source_url": config.DATASET_DOWNLOAD_URL,
        "downloaded_at_utc": downloaded_at.isoformat(),
        "target_column": config.TARGET_COLUMN,
        "archive": {
            "path": _project_relative(archive.path),
            "bytes": archive.size_bytes,
            "sha256": archive.sha256,
        },
        "csv": {
            "path": _project_relative(csv_info.path),
            "bytes": csv_info.size_bytes,
            "row_count": csv_info.row_count,
            "columns": list(csv_info.columns),
        },
    }

    try:
        part_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(part_path, manifest_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download and extract the Zomato delivery operations dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help=f"Directory for the archive and CSV (default: {config.RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite the archive and CSV.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {config.DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _log_dataset_summary(csv_info: CsvInfo) -> None:
    logger.info("Dataset ready: %s", csv_info.path)
    logger.info(
        "Rows: %d | Columns: %d | Target: %s",
        csv_info.row_count,
        len(csv_info.columns),
        config.TARGET_COLUMN,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Download, extract, and validate the dataset."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    output_dir: Path = args.output_dir
    archive_path = output_dir / config.ARCHIVE_FILENAME
    csv_path = output_dir / config.EXPECTED_CSV_FILENAME

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        if csv_path.exists() and not args.force:
            csv_info = validate_csv(csv_path)
            if archive_path.exists():
                try:
                    archive = inspect_archive(archive_path)
                except FetchError as exc:
                    logger.warning("Existing archive is invalid (%s); skipping manifest.", exc)
                else:
                    manifest_path = write_manifest(
                        output_dir=output_dir,
                        archive=archive,
                        csv_info=csv_info,
                    )
                    logger.info("Manifest: %s", manifest_path)
            else:
                logger.warning(
                    "Archive not found; using existing CSV without a manifest: %s",
                    csv_path,
                )
            _log_dataset_summary(csv_info)
            return 0

        auth = resolve_auth()
        if auth is None:
            logger.info("Downloading %s anonymously", config.DATASET_SLUG)
        else:
            logger.info("Downloading %s with Kaggle authentication", config.DATASET_SLUG)

        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(args.timeout),
            headers={"User-Agent": config.USER_AGENT},
            auth=auth,
        ) as client:
            if archive_path.exists() and not args.force:
                archive = inspect_archive(archive_path)
                logger.info("Using existing archive: %s", archive_path)
            else:
                archive = download_archive(
                    client,
                    config.DATASET_DOWNLOAD_URL,
                    archive_path,
                )

        extract_csv(archive_path, csv_path, force=args.force)
        csv_info = validate_csv(csv_path)
        manifest_path = write_manifest(
            output_dir=output_dir,
            archive=archive,
            csv_info=csv_info,
        )

        _log_dataset_summary(csv_info)
        logger.info("Manifest: %s", manifest_path)
        return 0
    except FetchError as exc:
        logger.error("%s", exc)
        return 1
    except httpx.HTTPError as exc:
        logger.error("HTTP error while downloading the dataset: %s", exc)
        return 1
    except (OSError, zipfile.BadZipFile) as exc:
        logger.error("Filesystem or archive error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
