"""Tests for the delivery-time data fetcher."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from delivery_time import config, fetch


def write_csv(
    path: Path,
    columns: Sequence[str] = config.EXPECTED_COLUMNS,
    rows: int = 2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(rows):
            writer.writerow([f"value-{index}"] * len(columns))
    return path


def make_zip(
    path: Path,
    csv_path: Path,
    member_name: str = config.EXPECTED_CSV_FILENAME,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(csv_path, arcname=member_name)
    return path


def test_resolve_auth_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    assert fetch.resolve_auth() is None


def test_resolve_auth_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")

    auth = fetch.resolve_auth()

    assert auth is not None
    request = next(auth.auth_flow(httpx.Request("GET", "https://example.com")))
    assert request.headers["authorization"].startswith("Basic ")


def test_resolve_auth_rejects_partial_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with pytest.raises(fetch.FetchError, match="both KAGGLE_USERNAME and KAGGLE_KEY"):
        fetch.resolve_auth()


def test_download_archive_streams_and_hashes(tmp_path: Path) -> None:
    source_csv = write_csv(tmp_path / "source.csv")
    source_archive = make_zip(tmp_path / "source.zip", source_csv)
    body = source_archive.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(config.DATASET_DOWNLOAD_URL)
        return httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=body,
        )

    destination = tmp_path / "out" / config.ARCHIVE_FILENAME
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        info = fetch.download_archive(
            client,
            config.DATASET_DOWNLOAD_URL,
            destination,
        )

    assert info.path == destination
    assert destination.read_bytes() == body
    assert info.size_bytes == len(body)
    assert info.sha256 == hashlib.sha256(body).hexdigest()


def test_download_archive_follows_redirect(tmp_path: Path) -> None:
    source_csv = write_csv(tmp_path / "source.csv")
    source_archive = make_zip(tmp_path / "source.zip", source_csv)
    body = source_archive.read_bytes()
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "www.kaggle.com":
            return httpx.Response(
                302,
                headers={"location": "https://storage.example.com/archive.zip"},
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=body,
        )

    destination = tmp_path / "out" / config.ARCHIVE_FILENAME
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        info = fetch.download_archive(
            client,
            config.DATASET_DOWNLOAD_URL,
            destination,
        )

    assert requested_hosts == ["www.kaggle.com", "storage.example.com"]
    assert info.path.read_bytes() == body


def test_download_archive_rejects_html(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>login</html>",
        )

    destination = tmp_path / "out" / config.ARCHIVE_FILENAME
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as client,
        pytest.raises(fetch.FetchError, match="content type"),
    ):
        fetch.download_archive(
            client,
            config.DATASET_DOWNLOAD_URL,
            destination,
        )

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_download_archive_reports_auth_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    destination = tmp_path / "out" / config.ARCHIVE_FILENAME
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as client,
        pytest.raises(fetch.FetchError, match="authentication"),
    ):
        fetch.download_archive(
            client,
            config.DATASET_DOWNLOAD_URL,
            destination,
        )

    assert not destination.exists()


def test_download_archive_cleans_up_invalid_zip(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=b"not a zip archive",
        )

    destination = tmp_path / "out" / config.ARCHIVE_FILENAME
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as client,
        pytest.raises(fetch.FetchError, match="valid ZIP"),
    ):
        fetch.download_archive(
            client,
            config.DATASET_DOWNLOAD_URL,
            destination,
        )

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_extract_csv(tmp_path: Path) -> None:
    source_csv = write_csv(tmp_path / "source.csv")
    source_archive = make_zip(tmp_path / "source.zip", source_csv)
    destination = tmp_path / "out" / config.EXPECTED_CSV_FILENAME

    result = fetch.extract_csv(source_archive, destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == source_csv.read_text(encoding="utf-8")


def test_extract_csv_prefers_expected_member(tmp_path: Path) -> None:
    expected_csv = write_csv(tmp_path / "expected.csv", rows=1)
    other_csv = write_csv(tmp_path / "other.csv", rows=3)
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(other_csv, arcname="other.csv")
        archive.write(expected_csv, arcname=config.EXPECTED_CSV_FILENAME)

    destination = tmp_path / "out" / config.EXPECTED_CSV_FILENAME
    fetch.extract_csv(archive_path, destination)

    assert destination.read_text(encoding="utf-8") == expected_csv.read_text(encoding="utf-8")


def test_validate_csv_requires_target(tmp_path: Path) -> None:
    columns = tuple(name for name in config.EXPECTED_COLUMNS if name != config.TARGET_COLUMN)
    csv_path = write_csv(tmp_path / "data.csv", columns=columns)

    with pytest.raises(fetch.FetchError, match="Target column"):
        fetch.validate_csv(csv_path)


def test_validate_csv_warns_on_extra_columns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    columns = (*config.EXPECTED_COLUMNS, "extra_column")
    csv_path = write_csv(tmp_path / "data.csv", columns=columns)

    with caplog.at_level(logging.WARNING):
        info = fetch.validate_csv(csv_path)

    assert info.row_count == 2
    assert "unexpected columns: extra_column" in caplog.text


def test_main_reuses_existing_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_csv(tmp_path / config.EXPECTED_CSV_FILENAME)

    def fail_client(**kwargs: object) -> None:
        raise AssertionError("The network client should not be created.")

    monkeypatch.setattr(fetch.httpx, "Client", fail_client)

    assert fetch.main(["--output-dir", str(tmp_path)]) == 0


def test_main_force_re_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = write_csv(tmp_path / "source.csv")
    source_archive = make_zip(tmp_path / "source.zip", source_csv)
    body = source_archive.read_bytes()
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=body,
        )

    real_client = httpx.Client

    def client_factory(**kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch.httpx, "Client", client_factory)

    output_dir = tmp_path / "out"
    assert fetch.main(["--output-dir", str(output_dir), "--force"]) == 0

    assert len(calls) == 1
    assert (output_dir / config.ARCHIVE_FILENAME).is_file()
    assert (output_dir / config.EXPECTED_CSV_FILENAME).is_file()
    manifest_path = output_dir / config.MANIFEST_FILENAME
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["target_column"] == config.TARGET_COLUMN
    assert manifest["csv"]["row_count"] == 2
    assert len(manifest["archive"]["sha256"]) == 64
