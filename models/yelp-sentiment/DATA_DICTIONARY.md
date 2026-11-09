# Yelp Review Full Dataset — Data Dictionary

This dictionary describes the schema and field meaning for the parquet files
downloaded by `uv run fetch-sentiment-data`. Field descriptions are adapted from
the Hugging Face dataset card.

## Dataset at a glance

| Item | Value |
|---|---|
| Source | <https://huggingface.co/datasets/Yelp/yelp_review_full> |
| Publisher | Yelp |
| Dataset commit | `c1f9ee939b7d05667af864ee1cb066393154bf85` |
| Local files | `data/raw/yelp_review_full/` |
| Fetch command | `uv run fetch-sentiment-data` |
| Manifest | `data/raw/yelp_review_full/fetch-manifest.json` |
| License | Other (Yelp license) |
| Format | Parquet |
| Rows | 700,000 total: 650,000 train + 50,000 test |
| Columns | 2 |
| Target | `label` |
| Text field | `text` |
| Label classes | 5 balanced classes |

The parquet schema is:

```text
label: Int64
text: String
```

## File metadata

| Split | File | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| train | `train-00000-of-00001.parquet` | 650,000 | 299,436,850 | `da3ca3dcc52b1e2e44195b53668610d645a59010a985dd45aa12e1109cd44601` |
| test | `test-00000-of-00001.parquet` | 50,000 | 23,515,519 | `bf06d5969bff93ecd4a6d4b330643761ce108b42e9b8a894cf82c9df02a08540` |

The checksums and row counts are also recorded in `fetch-manifest.json`.

## Field dictionary

| # | Column | Role | Logical type | Description |
|---:|---|---|---|---|
| 1 | `label` | Target | integer | Star-rating class: 0 = 1 star through 4 = 5 stars. |
| 2 | `text` | Feature | string | Review body stored as a raw parquet string. |

## Label mapping

| Code | Meaning | Train rows | Test rows | Total | Share |
|---:|---|---:|---:|---:|---:|
| 0 | 1 star | 130,000 | 10,000 | 140,000 | 20% |
| 1 | 2 star | 130,000 | 10,000 | 140,000 | 20% |
| 2 | 3 stars | 130,000 | 10,000 | 140,000 | 20% |
| 3 | 4 stars | 130,000 | 10,000 | 140,000 | 20% |
| 4 | 5 stars | 130,000 | 10,000 | 140,000 | 20% |
