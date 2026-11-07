# Yelp Review Full Dataset — Data Dictionary

This dictionary describes the actual parquet files downloaded by
`uv run fetch-sentiment-data`. Field descriptions are adapted from the
Hugging Face dataset card; observed values, missing counts, and data-quality
notes come from the downloaded snapshot.

## Dataset at a glance

| Item | Value |
|---|---|
| Source | <https://huggingface.co/datasets/Yelp/yelp_review_full> |
| Publisher | Yelp |
| Dataset commit | `c1f9ee939b7d05667af864ee1cb066393154bf85` |
| Local files | `data/raw/yelp_review_full/` |
| Fetch command | `uv run fetch-sentiment-data` |
| Manifest | `data/raw/yelp_review_full/fetch-manifest.json` |
| Snapshot inspected | 2026-11-07 |
| License | Other (Yelp license) |
| Rows | 700,000 total: 650,000 train + 50,000 test |
| Columns | 2 |
| Target | `label` |
| Text field | `text` |
| Missing cells | 0 |
| Exact duplicate rows | 0 |
| Duplicate review texts | 0 |
| Unique review texts | 700,000 |
| Train/test exact-text overlap | 0 |
| Label classes | 5 |
| Class balance | Exactly 20% per class |
| Total file size | 322,952,369 bytes (307.9 MiB) |

The parquet schema is:

```text
label: Int64
text: String
```

Raw labels are preserved as `0–4`. They map to 1–5 stars:

| Code | Meaning | Train rows | Test rows | Total | Share |
|---:|---|---:|---:|---:|---:|
| 0 | 1 star | 130,000 | 10,000 | 140,000 | 20% |
| 1 | 2 star | 130,000 | 10,000 | 140,000 | 20% |
| 2 | 3 stars | 130,000 | 10,000 | 140,000 | 20% |
| 3 | 4 stars | 130,000 | 10,000 | 140,000 | 20% |
| 4 | 5 stars | 130,000 | 10,000 | 140,000 | 20% |

The classes are perfectly balanced in both splits. Accuracy and macro F1 should
therefore be close, but macro F1 and per-class precision/recall remain useful
for checking whether the model performs evenly across sentiment levels.

## File metadata

| Split | File | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| train | `train-00000-of-00001.parquet` | 650,000 | 299,436,850 | `da3ca3dcc52b1e2e44195b53668610d645a59010a985dd45aa12e1109cd44601` |
| test | `test-00000-of-00001.parquet` | 50,000 | 23,515,519 | `bf06d5969bff93ecd4a6d4b330643761ce108b42e9b8a894cf82c9df02a08540` |

The checksums and row counts are also recorded in `fetch-manifest.json`.

## Field dictionary

| # | Column | Role | Logical type | Description | Missing | Observed values / range |
|---:|---|---|---|---|---|---|
| 1 | `label` | Target | integer | Star-rating class: 0 = 1 star through 4 = 5 stars. | 0 | 0–4; 140,000 rows per class overall. |
| 2 | `text` | Feature | string | Review body. Values contain literal escape sequences such as `\n`, `\"`, and `\u`. | 0 | 1–5,637 characters; median 539 characters; 1–1,052 whitespace tokens. |

There are no null values in either column and no empty or whitespace-only
reviews.

## Text length statistics

Character counts are Unicode code points in the raw parquet text. Word counts
are whitespace-delimited tokens from the regex `\S+`, not BERT tokenizer tokens.
Lengths are computed **before** decoding the literal escape sequences, so they
are slightly longer than the decoded text for many rows.

### Characters

| Statistic | Train | Test | Combined |
|---|---:|---:|---:|
| Rows | 650,000 | 50,000 | 700,000 |
| Nulls | 0 | 0 | 0 |
| Mean | 732.3 | 733.4 | 732.4 |
| Standard deviation | 663.7 | 664.0 | 663.7 |
| Minimum | 1 | 1 | 1 |
| 25th percentile | 287 | 287 | 287 |
| Median | 539 | 540 | 539 |
| 75th percentile | 957 | 957 | 957 |
| 90th percentile | 1,542 | 1,542 | 1,542 |
| 95th percentile | 2,025 | 2,008 | 2,024 |
| 99th percentile | 3,289 | 3,327 | 3,292 |
| Maximum | 5,637 | 5,225 | 5,637 |

### Words

| Statistic | Train | Test | Combined |
|---|---:|---:|---:|
| Mean | 134.1 | 134.3 | 134.1 |
| Standard deviation | 121.4 | 121.2 | 121.4 |
| Minimum | 1 | 1 | 1 |
| 25th percentile | 52 | 53 | 52 |
| Median | 99 | 99 | 99 |
| 75th percentile | 175 | 176 | 175 |
| 90th percentile | 282 | 283 | 282 |
| 95th percentile | 370 | 368 | 370 |
| 99th percentile | 601 | 605 | 601 |
| Maximum | 1,052 | 1,009 | 1,052 |

### Text length by label

Averages are computed on the combined 700,000 rows, before unescaping.

| Label | Mean characters | Median characters | Mean words | Median words |
|---:|---:|---:|---:|---:|
| 0 (1 star) | 815.1 | 582 | 151.4 | 108 |
| 1 (2 star) | 827.6 | 629 | 152.0 | 116 |
| 2 (3 stars) | 759.3 | 583 | 138.7 | 107 |
| 3 (4 stars) | 680.6 | 512 | 123.5 | 94 |
| 4 (5 stars) | 579.4 | 408 | 104.9 | 74 |

Lower-star reviews tend to be longer than higher-star reviews. This is useful
context for truncation and for checking whether the model is using review
length as a shortcut.

## Sample reviews

Excerpts are shortened and their literal escape sequences are decoded for
readability. The underlying `text` field remains raw.

| Label | Excerpt |
|---:|---|
| 0 (1 star) | "Used to go there for tires, brakes, etc. Their prices have gone way up-$400 for 4 mid-level tires for a Toyota. Plus, 1 of the new tires went flat within 3 weeks..." |
| 1 (2 star) | "Wing sauce is like water. Pretty much a lot of butter and some hot sauce. The whole wings are good size and crispy, but for $1 a wing the sauce could be better..." |
| 2 (3 stars) | "Decent range somewhat close to the city. The mats are pretty solid; however, the grass range needs to be tended too. It's like hitting out of US Open type rough..." |
| 3 (4 stars) | "Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years..." |
| 4 (5 stars) | "Top notch doctor in a top notch practice. Can't say I am surprised when I was referred to him by another doctor who I think is wonderful..." |

## Data quality notes

### Missing values, duplicates, and split leakage

- `label` and `text` have no null values.
- There are no exact duplicate rows.
- All 700,000 review texts are unique.
- There is no exact-text overlap between train and test.
- Near-duplicate reviews were not checked. Exact matching is enough to rule out
  direct train/test leakage, but it does not rule out paraphrases or repeated
  templates.

### Label semantics

- `label` is the original five-class star rating, not a binary sentiment label.
- The 3-star class is neutral/ambiguous. If the project moves to binary
  sentiment, decide explicitly whether to:
  - drop 3-star reviews and use 1–2 stars as negative and 4–5 as positive;
  - keep 3 stars as a neutral class;
  - collapse 1–3 stars into negative and 4–5 into positive.
- Do not collapse labels in the raw data. The mapping belongs in preprocessing or
  EDA documentation.

### Literal escape sequences in text

The `text` field is not plain prose in all rows. It contains literal
JSON-style escape sequences:

| Sequence | Rows containing it | Share |
|---|---:|---:|
| Any backslash | 389,072 | 55.58% |
| `\n` | 350,784 | 50.11% |
| `\"` | 125,990 | 18.00% |
| `\u` | 16,473 | 2.35% |
| `\\` | 349 | 0.05% |
| `\r` | 190 | 0.03% |
| `\t` | 3 | 0.0004% |

There are **no actual newline characters** in the parquet text. The raw strings
contain two-character sequences such as backslash + `n` instead.

There are also no non-ASCII characters in the raw strings; non-ASCII content
appears to be represented with `\u` escapes. Text preprocessing should decode
these sequences carefully and validate that decoding does not fail on malformed
rows.

### Long reviews and truncation

- 488 reviews (0.070%) are longer than 5,000 characters.
- 16 reviews (0.002%) have more than 1,000 whitespace tokens.
- The 99th percentile is 3,292 characters and 601 whitespace tokens.
- The maximum is 5,637 characters and 1,052 whitespace tokens.

BERT-family models will require a maximum sequence length and truncation policy.
Do not choose the limit from character counts alone: tokenize a sample after
unescaping, then inspect token-length percentiles. Because lower-star reviews are
longer on average, aggressive truncation may remove more evidence from negative
reviews.

### Very short reviews

- 1,636 reviews (0.234%) are shorter than 20 characters.
- 3,181 reviews (0.454%) have fewer than 5 whitespace tokens.
- The shortest examples are one-character strings such as `A`, `.`, `?`, and
  `!`.

Very short reviews may be ambiguous, noisy, or label-dependent on context not
present in the text. Keep them for the first baseline, then test whether
filtering or separate handling improves validation performance.

### URLs and formatting

- 2,928 reviews (0.418%) contain the literal substring `http`.
- 2,820 reviews (0.403%) contain `www.`.
- 17 reviews (0.002%) contain `&amp;`.
- There are no `<br>` tags and no actual newline characters.

URL handling is a preprocessing decision. A BERT tokenizer can process URLs, but
normalizing or removing them may reduce noise. Test this as a preprocessing
ablation rather than assuming one choice is better.

## Preprocessing implications

- Decode literal escape sequences before tokenization, including `\n`, `\"`,
  `\uXXXX`, `\\`, `\r`, and `\t`. Validate malformed rows.
- Recompute text-length and token-length statistics after unescaping.
- Keep case and punctuation for a BERT-family model unless an ablation shows a
  benefit from normalization. Aggressive lowercasing and punctuation removal can
  remove sentiment cues such as negation and emphasis.
- Measure token lengths with the actual tokenizer before choosing
  `max_length`. Consider head-only truncation versus head + tail truncation.
- Use the supplied train/test split. Carve a class-stratified validation split
  from train for tuning; do not tune on the test set.
- Use accuracy, macro F1, and per-class precision/recall. Class balance makes
  accuracy less misleading than usual, but macro F1 still catches uneven
  performance.
- Decide the sentiment label mapping during EDA, not during fetching. The raw
  data should remain five-class star ratings.
- Check normalized near-duplicates if validation scores look unusually high.
- Treat review length as a potential shortcut. Evaluate whether a length-only
  baseline predicts the label surprisingly well.

## Open questions for EDA

1. Should the task stay five-class or collapse to binary sentiment?
2. If binary, what should happen to the 3-star class?
3. What token length covers 95–99% of reviews after unescaping?
4. Does decoding escape sequences materially change token counts or model input?
5. Do URLs, very short reviews, or near-duplicates affect class-specific errors?
6. Does review length correlate strongly enough with sentiment to act as a
   shortcut?
