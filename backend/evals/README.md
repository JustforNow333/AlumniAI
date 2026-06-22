# Backend Evals

This directory contains a repeatable evaluation harness for the AI Spreadsheet
Analyst backend. It is separate from production app behavior: eval code uploads
a sanitized CSV through the real Flask API, asks natural-language questions, and
scores the returned answer against the gold dataset.

## Run

From `backend/`:

```bash
python -m evals.run_evals
```

On this Windows-backed checkout the existing virtualenv is often invoked as:

```bash
./venv/Scripts/python.exe -m evals.run_evals
```

The runner writes:

- `backend/evals/results/latest.json`
- `backend/evals/results/latest.md`

Useful filters:

```bash
python -m evals.run_evals --case-id tech_001
python -m evals.run_evals --category industry_classification
python -m evals.run_evals --limit 10
python -m evals.run_evals --output backend/evals/results/latest.json
python -m evals.run_evals --markdown backend/evals/results/latest.md
```

## Modes

By default the runner uses `--mode offline`, disables the OpenAI client, and
does not spend API budget:

```bash
python -m evals.run_evals --mode offline
```

Available modes:

- `offline`: deterministic backend evals. AI is disabled for every case.
- `hybrid`: product API evals with per-case AI policy. Deterministic cases keep
  AI disabled; broad product cases may use the live client when configured.
- `classifier-live`: direct classifier evals for the row/industry classifier.
  Cases can require the LLM classifier when deterministic rules are intentionally
  insufficient.
- `smoke-live`: a small mixed live smoke subset.

The old flag still works as an alias for `--mode hybrid`:

```bash
python -m evals.run_evals --use-live-ai
```

## Gold Data And Leakage

The gold fixture is `evals/datasets/synthetic_alumni_500.csv`. It contains
normal app-facing columns plus eval-only labels such as `expected_industry`.

The app must not see those labels. Each eval run creates:

```text
evals/generated/synthetic_alumni_500_app_view.csv
```

That app-facing CSV removes `expected_industry`, every `expected_*` column, and
every `eval_*` column before upload. Scoring uses the original gold dataframe in
memory.

## Cases

Cases live in `evals/cases.jsonl`, one JSON object per line. Common fields:

- `id`: stable case id.
- `category`: group used by `--category`.
- `question`: user-facing natural-language prompt.
- `expected_industry`: gold `expected_industry` label for industry evals.
- `expected_filter`: deterministic filter spec for exact cases.
- `exact_match`: require exact count and returned row names.
- `precision_threshold` / `recall_threshold`: per-case pass thresholds.
- `required_columns` / `forbidden_columns`: displayed table column checks.
- `required_phrases` / `forbidden_phrases`: answer text checks.
- `must_include_names` / `must_exclude_names`: regression targets.
- `modes`: eval modes that should run the case.
- `eval_kind`: `api` for `/api/ask` cases or `direct_classifier` for direct
  classifier cases.
- `execution`: expected execution behavior, including `model_calls`,
  `llm_classifier`, `final_model_synthesis`, and expected `scored_from`.

Add new cases by appending a JSON object to `cases.jsonl`. Prefer computing
expected rows from `expected_industry` or `expected_filter` rather than copying
counts by hand.

## Scoring

The scorer first uses structured API data from `/api/ask`, especially
`operation_results[*].rows`, `columns`, and `metrics`. If those are unavailable,
it falls back to structured answer table blocks, then to conservative text
extraction.

Every case reports trace fields:

- `total_model_calls`
- `used_llm_classifier`
- `llm_classifier_calls`
- `used_final_model_synthesis`
- `answer_source`
- `scored_from`
- `backend_testing_mode`
- `ai_enabled`
- `model_name`

Industry cases report:

- precision = correct returned names / total returned names
- recall = correct returned names / total expected names
- false positives
- false negatives
- hallucinated names

Exact deterministic cases compare the app-reported count and returned rows to
rows computed directly from the gold CSV.

Display-rule cases catch forbidden fields such as `expected_industry`,
`Match Reason`, internal debug fields, and `Notes` when a case forbids them.
They also check required columns such as clean first/last names and LinkedIn URL
for list-style alumni answers.

Direct classifier cases call the classifier layer directly and score the
returned `classification` and `count_as_match` fields. They are useful for
ambiguous employer regressions such as Spotify, Google, FanAmp, Cogni DAO,
Bright Ventures, McKinsey, Goldman Sachs, Capital One, and high-school cases.

## Interpreting Failures

Read `evals/results/latest.md` first. Failed cases include the case id,
category, question, failure reasons, expected count, returned count, displayed
count, precision/recall, hallucinations, false positives, sample false
negatives, and a raw answer excerpt.

Precision failures usually mean the app included people who do not belong in
the requested group. Recall failures usually mean the app missed too many gold
matches. Count failures mean the reported count, displayed count, or returned
rows are inconsistent with each other or with the gold CSV.

Failure categories identify whether a failure came from `row_selection`,
`classification`, `count_mismatch`, `forbidden_columns`, `hallucinated_names`,
`response_parsing`, or `execution_behavior`.

Update thresholds when the case intentionally becomes narrower or broader. Do
not loosen thresholds to hide regressions; add notes to the case when a tradeoff
is intentional.
