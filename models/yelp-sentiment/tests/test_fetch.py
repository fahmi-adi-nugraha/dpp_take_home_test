"""Tests for the Yelp Review Full dataset fetcher."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import httpx
import polars as pl
import pytest
from yelp_sentiment import config, fetch


def write_parquet(
    path: Path,
    *,
    rows: int = 5,
    extra_columns: tuple[str, ...] = (),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "label": [index % 5 for index in range(rows)],
        "text": [f"review {index}" for index in range(rows)],
    }
    for column in extra_columns:
        data[column] = list(range(rows))
    pl.DataFrame(data).write_parquet(path)
    return path


def make_spec(
    path: Path,
    *,
    split: str = "test",
    rows: int = 5,
) -> fetch.FileSpec:
    body = path.read_bytes()
    return fetch.FileSpec(
        split=split,
        remote_path=f"yelp_review_full/{path.name}",
        filename=path.name,
        expected_rows=rows,
        expected_bytes=len(body),
        expected_sha256=hashlib.sha256(body).hexdigest(),
    )


def test_build_file_url_uses_revision_and_remote_path(tmp_path: Path) -> None:
    source = write_parquet(tmp_path / "test-00000-of-00001.parquet")
    spec = make_spec(source)

    url = fetch.build_file_url(spec, "main")

    assert url == (
        f"https://huggingface.co/datasets/{config.DATASET_ID}/resolve/main/"
        f"yelp_review_full/test-00000-of-00001.parquet?download=true"
    )


def test_resolve_headers_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert fetch.resolve_headers() == {"User-Agent": config.USER_AGENT}


def test_resolve_headers_with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    headers = fetch.resolve_headers()

    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["User-Agent"] == config.USER_AGENT


def test_download_file_streams_verifies_and_saves(tmp_path: Path) -> None:
    source = write_parquet(tmp_path / "test-00000-of-00001.parquet")
    spec = make_spec(source)
    body = source.read_bytes()
    url = fetch.build_file_url(spec, "main")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(url)
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
                "x-repo-commit": "fetched-commit",
            },
            content=body,
        )

    destination = tmp_path / "out" / spec.filename
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        info = fetch.download_file(client, url, destination, spec)

    assert info.path == destination
    assert destination.read_bytes() == body
    assert info.size_bytes == len(body)
    assert info.sha256 == hashlib.sha256(body).hexdigest()
    assert info.parquet.rows == 5
    assert info.parquet.label_counts == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert info.resolved_commit == "fetched-commit"
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_download_file_follows_redirect(tmp_path: Path) -> None:
    source = write_parquet(tmp_path / "test-00000-of-00001.parquet")
    spec = make_spec(source)
    body = source.read_bytes()
    url = fetch.build_file_url(spec, "main")
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "huggingface.co":
            return httpx.Response(
                302,
                headers={"location": "https://cdn.example.com/data.parquet"},
            )
        return httpx.Response(200, content=body)

    destination = tmp_path / "out" / spec.filename
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        fetch.download_file(client, url, destination, spec)

    assert requested_hosts == ["huggingface.co", "cdn.example.com"]
    assert destination.read_bytes() == body


def test_download_file_reports_auth_failure(tmp_path: Path) -> None:
    source = write_parquet(tmp_path / "test-00000-of-00001.parquet")
    spec = make_spec(source)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    destination = tmp_path / "out" / spec.filename
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as client,
        pytest.raises(fetch.FetchError, match="HF_TOKEN"),
    ):
        fetch.download_file(
            client,
            fetch.build_file_url(spec, "main"),
            destination,
            spec,
        )

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_download_file_cleans_up_invalid_parquet(tmp_path: Path) -> None:
    body = b"<html>not a parquet file</html>"
    spec = fetch.FileSpec(
        split="test",
        remote_path="yelp_review_full/test.parquet",
        filename="test.parquet",
        expected_rows=0,
        expected_bytes=len(body),
        expected_sha256=hashlib.sha256(body).hexdigest(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    destination = tmp_path / "out" / spec.filename
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as client,
        pytest.raises(fetch.FetchError, match="parquet"),
    ):
        fetch.download_file(
            client,
            fetch.build_file_url(spec, "main"),
            destination,
            spec,
        )

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_validate_parquet_requires_label_and_text(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    pl.DataFrame({"text": ["only text"]}).write_parquet(path)

    with pytest.raises(fetch.FetchError, match="label"):
        fetch.validate_parquet(path)


def test_validate_parquet_rejects_unknown_label(tmp_path: Path) -> None:
    path = tmp_path / "bad-label.parquet"
    pl.DataFrame({"label": [0, 5], "text": ["a", "b"]}).write_parquet(path)

    with pytest.raises(fetch.FetchError, match="Unexpected labels"):
        fetch.validate_parquet(path)


def test_validate_parquet_warns_on_extra_columns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = write_parquet(tmp_path / "extra.parquet", extra_columns=("extra",))
    spec = make_spec(source)

    with caplog.at_level(logging.WARNING):
        info = fetch.validate_parquet(source, spec)

    assert info.rows == 5
    assert info.columns == ("label", "text", "extra")
    assert "unexpected columns: extra" in caplog.text


def test_inspect_file_validates_existing_file(tmp_path: Path) -> None:
    source = write_parquet(tmp_path / "test-00000-of-00001.parquet")
    spec = make_spec(source)

    info = fetch.inspect_file(source, spec)

    assert info.path == source
    assert info.parquet.rows == 5
    assert info.sha256 == spec.expected_sha256


def test_main_reuses_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = write_parquet(tmp_path / "train-00000-of-00001.parquet", rows=10)
    test = write_parquet(tmp_path / "test-00000-of-00001.parquet", rows=5)
    specs = (
        make_spec(train, split="train", rows=10),
        make_spec(test, split="test", rows=5),
    )
    monkeypatch.setattr(fetch, "DEFAULT_FILE_SPECS", specs)

    def fail_client(**kwargs: object) -> None:
        raise AssertionError("The network client should not be created.")

    monkeypatch.setattr(fetch.httpx, "Client", fail_client)

    assert fetch.main(["--output-dir", str(tmp_path)]) == 0

    manifest = json.loads((tmp_path / config.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["dataset"] == config.DATASET_ID
    assert manifest["total_rows"] == 15
    assert {item["split"] for item in manifest["files"]} == {"train", "test"}


def test_main_force_redownloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_source = write_parquet(
        tmp_path / "source-train.parquet",
        rows=10,
    )
    test_source = write_parquet(
        tmp_path / "source-test.parquet",
        rows=5,
    )
    train_spec = make_spec(train_source, split="train", rows=10)
    train_spec = fetch.FileSpec(
        split="train",
        remote_path="yelp_review_full/train-00000-of-00001.parquet",
        filename="train-00000-of-00001.parquet",
        expected_rows=train_spec.expected_rows,
        expected_bytes=train_spec.expected_bytes,
        expected_sha256=train_spec.expected_sha256,
    )
    test_spec = make_spec(test_source, split="test", rows=5)
    test_spec = fetch.FileSpec(
        split="test",
        remote_path="yelp_review_full/test-00000-of-00001.parquet",
        filename="test-00000-of-00001.parquet",
        expected_rows=test_spec.expected_rows,
        expected_bytes=test_spec.expected_bytes,
        expected_sha256=test_spec.expected_sha256,
    )
    monkeypatch.setattr(fetch, "DEFAULT_FILE_SPECS", (train_spec, test_spec))

    bodies = {
        "train-00000-of-00001.parquet": train_source.read_bytes(),
        "test-00000-of-00001.parquet": test_source.read_bytes(),
    }
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        filename = request.url.path.rsplit("/", 1)[-1]
        calls.append(filename)
        return httpx.Response(200, content=bodies[filename])

    real_client = httpx.Client

    def client_factory(**kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch.httpx, "Client", client_factory)

    output_dir = tmp_path / "out"
    assert fetch.main(["--output-dir", str(output_dir), "--force"]) == 0

    assert sorted(calls) == [
        "test-00000-of-00001.parquet",
        "train-00000-of-00001.parquet",
    ]
    assert (output_dir / config.MANIFEST_FILENAME).is_file()


def test_main_rejects_invalid_timeout() -> None:
    with pytest.raises(SystemExit):
        fetch.main(["--timeout", "0"])
