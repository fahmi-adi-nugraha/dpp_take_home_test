"""Download and validate the Yelp Review Full dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from yelp_sentiment import config

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when the dataset cannot be downloaded or validated."""


@dataclass(frozen=True)
class FileSpec:
    """Expected metadata for one Hugging Face parquet file."""

    split: str
    remote_path: str
    filename: str
    expected_rows: int
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class ParquetInfo:
    """Validated parquet contents."""

    rows: int
    columns: tuple[str, ...]
    label_counts: dict[int, int]


@dataclass(frozen=True)
class FileInfo:
    """Validated file metadata used to write the manifest."""

    spec: FileSpec
    path: Path
    size_bytes: int
    sha256: str
    parquet: ParquetInfo
    resolved_commit: str | None = None


DEFAULT_FILE_SPECS = (
    FileSpec(
        split="train",
        remote_path="yelp_review_full/train-00000-of-00001.parquet",
        filename="train-00000-of-00001.parquet",
        expected_rows=650_000,
        expected_bytes=299_436_850,
        expected_sha256=("da3ca3dcc52b1e2e44195b53668610d645a59010a985dd45aa12e1109cd44601"),
    ),
    FileSpec(
        split="test",
        remote_path="yelp_review_full/test-00000-of-00001.parquet",
        filename="test-00000-of-00001.parquet",
        expected_rows=50_000,
        expected_bytes=23_515_519,
        expected_sha256=("bf06d5969bff93ecd4a6d4b330643761ce108b42e9b8a894cf82c9df02a08540"),
    ),
)


def build_file_url(spec: FileSpec, revision: str = config.DATASET_REVISION) -> str:
    """Build a direct Hugging Face resolve URL for a parquet file."""
    return (
        f"https://huggingface.co/datasets/{config.DATASET_ID}/resolve/"
        f"{revision}/{spec.remote_path}?download=true"
    )


def resolve_headers() -> dict[str, str]:
    """Return request headers, including optional Hugging Face bearer auth."""
    headers = {"User-Agent": config.USER_AGENT}
    token = (os.getenv("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(config.CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _has_parquet_magic(path: Path) -> bool:
    try:
        if path.stat().st_size < 8:
            return False
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                return False
            handle.seek(-4, os.SEEK_END)
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def _verify_download(size: int, sha256: str, spec: FileSpec) -> None:
    if size != spec.expected_bytes:
        raise FetchError(f"{spec.split} file has {size} bytes, expected {spec.expected_bytes}.")
    if sha256 != spec.expected_sha256:
        raise FetchError(
            f"{spec.split} file SHA-256 is {sha256}, expected "
            f"{spec.expected_sha256}. The dataset may have changed."
        )


def validate_parquet(
    path: Path,
    spec: FileSpec | None = None,
) -> ParquetInfo:
    """Validate parquet magic, schema, label values, and row count."""
    if not _has_parquet_magic(path):
        raise FetchError(f"File is not a valid parquet file: {path}")

    try:
        schema = pl.read_parquet_schema(path)
    except Exception as exc:
        raise FetchError(f"Could not read parquet schema: {path}") from exc

    columns = tuple(schema)
    missing = [name for name in config.EXPECTED_COLUMNS if name not in schema]
    if missing:
        raise FetchError(f"Parquet file {path} is missing required columns: {', '.join(missing)}")

    extra = [name for name in columns if name not in config.EXPECTED_COLUMNS]
    if extra:
        logger.warning("Parquet has unexpected columns: %s", ", ".join(extra))
    elif columns != config.EXPECTED_COLUMNS:
        logger.warning("Parquet columns are in a different order than expected.")

    if schema[config.TARGET_COLUMN] not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        raise FetchError(f"Target column {config.TARGET_COLUMN!r} must be an integer.")
    if schema[config.TEXT_COLUMN] != pl.String:
        raise FetchError(f"Text column {config.TEXT_COLUMN!r} must be a string.")

    try:
        lazy_frame = pl.scan_parquet(path)
        rows = lazy_frame.select(pl.len()).collect().item()
        null_labels = lazy_frame.select(pl.col(config.TARGET_COLUMN).null_count()).collect().item()
        null_texts = lazy_frame.select(pl.col(config.TEXT_COLUMN).null_count()).collect().item()
        label_counts_frame = (
            lazy_frame.group_by(config.TARGET_COLUMN).len().collect().sort(config.TARGET_COLUMN)
        )
    except Exception as exc:
        raise FetchError(f"Could not read parquet contents: {path}") from exc

    if null_labels:
        raise FetchError(f"Target column {config.TARGET_COLUMN!r} contains null values.")
    if null_texts:
        logger.warning("Text column contains %d null values.", null_texts)

    label_counts = {
        int(row[config.TARGET_COLUMN]): int(row["len"])
        for row in label_counts_frame.iter_rows(named=True)
    }
    unexpected_labels = sorted(label for label in label_counts if label not in config.LABEL_MAP)
    if unexpected_labels:
        raise FetchError(f"Unexpected labels: {unexpected_labels}")

    if spec is not None and rows != spec.expected_rows:
        logger.warning(
            "%s has %d rows, expected %d.",
            spec.split,
            rows,
            spec.expected_rows,
        )

    return ParquetInfo(rows=rows, columns=columns, label_counts=label_counts)


def inspect_file(path: Path, spec: FileSpec) -> FileInfo:
    """Validate an existing downloaded file without using the network."""
    if not path.is_file():
        raise FetchError(f"File not found: {path}")

    size, sha256 = _sha256(path)
    _verify_download(size, sha256, spec)
    parquet = validate_parquet(path, spec)
    return FileInfo(
        spec=spec,
        path=path,
        size_bytes=size,
        sha256=sha256,
        parquet=parquet,
    )


def _check_response(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise FetchError(
            "Hugging Face rejected the download request. The dataset is public, "
            "but you can set HF_TOKEN in .env if access is restricted."
        )
    if response.status_code == 429:
        raise FetchError(
            "Hugging Face rate-limited the download. Retry later or set HF_TOKEN "
            "in .env for higher limits."
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"Hugging Face download failed with HTTP {response.status_code}.") from exc


def download_file(
    client: httpx.Client,
    url: str,
    destination: Path,
    spec: FileSpec,
) -> FileInfo:
    """Stream, verify, and atomically save one parquet file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    digest = hashlib.sha256()
    size = 0
    resolved_commit: str | None = None

    try:
        with client.stream("GET", url) as response:
            _check_response(response)
            resolved_commit = response.headers.get("x-repo-commit") or None

            with part_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)

        sha256 = digest.hexdigest()
        _verify_download(size, sha256, spec)
        parquet = validate_parquet(part_path, spec)
        os.replace(part_path, destination)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    return FileInfo(
        spec=spec,
        path=destination,
        size_bytes=size,
        sha256=sha256,
        parquet=parquet,
        resolved_commit=resolved_commit,
    )


def fetch_file(
    client: httpx.Client,
    spec: FileSpec,
    output_dir: Path,
    revision: str,
) -> FileInfo:
    """Download one missing or invalid file."""
    destination = output_dir / spec.filename
    logger.info("Downloading %s split from %s", spec.split, build_file_url(spec, revision))
    info = download_file(client, build_file_url(spec, revision), destination, spec)
    logger.info(
        "Saved %s: %d rows, %d bytes",
        info.path,
        info.parquet.rows,
        info.size_bytes,
    )
    return info


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _is_commit(revision: str) -> bool:
    return len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)


def write_manifest(
    output_dir: Path,
    file_infos: Sequence[FileInfo],
    revision: str,
) -> Path:
    """Write a JSON manifest describing the fetched dataset."""
    manifest_path = output_dir / config.MANIFEST_FILENAME
    part_path = manifest_path.with_name(f"{manifest_path.name}.part")
    resolved_commit = next(
        (info.resolved_commit for info in file_infos if info.resolved_commit),
        revision if _is_commit(revision) else None,
    )
    payload: dict[str, Any] = {
        "dataset": config.DATASET_ID,
        "source_url": config.DATASET_URL,
        "revision": revision,
        "resolved_commit": resolved_commit,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "license": config.DATASET_LICENSE,
        "target_column": config.TARGET_COLUMN,
        "text_column": config.TEXT_COLUMN,
        "label_map": {str(key): value for key, value in sorted(config.LABEL_MAP.items())},
        "total_rows": sum(info.parquet.rows for info in file_infos),
        "files": [
            {
                "split": info.spec.split,
                "path": _project_relative(info.path),
                "url": build_file_url(info.spec, revision),
                "bytes": info.size_bytes,
                "sha256": info.sha256,
                "rows": info.parquet.rows,
                "columns": list(info.parquet.columns),
                "label_counts": {
                    str(key): value for key, value in sorted(info.parquet.label_counts.items())
                },
                "resolved_commit": info.resolved_commit,
            }
            for info in file_infos
        ],
    }

    try:
        part_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(part_path, manifest_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download and validate the Yelp Review Full dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help=f"Directory for parquet files (default: {config.RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when valid local copies exist.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {config.DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--revision",
        default=config.DATASET_REVISION,
        help="Hugging Face dataset revision or commit (default: pinned commit).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _log_summary(file_infos: Sequence[FileInfo]) -> None:
    for info in file_infos:
        logger.info(
            "%s: %s | rows=%d | labels=%s",
            info.spec.split,
            info.path,
            info.parquet.rows,
            info.parquet.label_counts,
        )
    logger.info("Total rows: %d", sum(info.parquet.rows for info in file_infos))


def main(argv: Sequence[str] | None = None) -> int:
    """Download, validate, and manifest the Yelp Review Full dataset."""
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
    file_infos_by_split: dict[str, FileInfo] = {}
    pending_specs: list[FileSpec] = []

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        for spec in DEFAULT_FILE_SPECS:
            path = output_dir / spec.filename
            if path.exists() and not args.force:
                try:
                    file_infos_by_split[spec.split] = inspect_file(path, spec)
                except FetchError as exc:
                    logger.warning(
                        "Existing %s file is invalid (%s); re-downloading.",
                        spec.split,
                        exc,
                    )
                    pending_specs.append(spec)
                else:
                    logger.info("Using existing %s: %s", spec.split, path)
            else:
                pending_specs.append(spec)

        if pending_specs:
            headers = resolve_headers()
            auth_mode = (
                "with Hugging Face authentication" if "Authorization" in headers else "anonymously"
            )
            logger.info("Downloading %s %s", config.DATASET_ID, auth_mode)
            with httpx.Client(
                follow_redirects=True,
                timeout=httpx.Timeout(args.timeout),
                headers=headers,
            ) as client:
                for spec in pending_specs:
                    file_infos_by_split[spec.split] = fetch_file(
                        client,
                        spec,
                        output_dir,
                        args.revision,
                    )

        file_infos = [file_infos_by_split[spec.split] for spec in DEFAULT_FILE_SPECS]
        manifest_path = write_manifest(output_dir, file_infos, args.revision)
        _log_summary(file_infos)
        logger.info("Manifest: %s", manifest_path)
        return 0
    except FetchError as exc:
        logger.error("%s", exc)
        return 1
    except httpx.HTTPError as exc:
        logger.error("HTTP error while downloading the dataset: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Filesystem error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
