# SpamAssassin Detector

SpamAssassin is the rule-based industry-standard baseline. No training
needed — we run it against the real test set (for baseline metrics) and
the synthetic phishing set (for the detection gap).

## Files

- `setup.md` — one-time macOS install & `spamd` startup
- `eval.py` — scores a CSV of emails via `spamc`, writes per-email results
- `gap.py` — reads per-email results, computes baseline and detection gap

## Full run sequence

### 0. Setup (one-time)

Follow `setup.md`:
1. `brew install spamassassin`
2. `sudo sa-update --nogpg`
3. Start `spamd` in a terminal and leave it running

### 1. Score the real test set (~3–5 min on 2,647 emails, 8 workers)

From repo root:

```bash
mkdir -p results/spamassassin

python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_test.csv \
    --split  test \
    --workers 8
```

### 2. Score the synthetic phishing set (~1 min on 457 emails)

```bash
python detection/spamassassin/eval.py \
    --input  data/processed/synthetic_phishing_clean.csv \
    --output results/spamassassin/synthetic.csv \
    --workers 8
```

### 3. Compute baseline metrics and detection gap

```bash
python detection/spamassassin/gap.py \
    --real      results/spamassassin/real_test.csv \
    --synthetic results/spamassassin/synthetic.csv \
    --synthetic-meta data/processed/synthetic_phishing_clean.csv \
    --outdir    results/spamassassin/
```

This prints a summary and writes two tables to `results/spamassassin/`:

- **`baseline.csv`** — one row with accuracy/precision/recall/F1 and
  confusion matrix (TP/FP/TN/FN) on the real test set.
- **`detection_gap.csv`** — long-form table, one row per grouping level.
  The `grouping` column takes values `real_baseline`, `overall`, `method`,
  `sophistication`, or `method_scenario`, and `group_value` specifies
  which group within that level. Columns: `detector, grouping,
  group_value, n, caught, detection_rate, mean_score`.

To get (say) just the per-method breakdown:
```python
import pandas as pd
df = pd.read_csv("results/spamassassin/detection_gap.csv")
by_method = df[df["grouping"] == "method"]
```

This format pivots cleanly for the Phase 4 cross-detector comparison —
concatenate the four detectors' `detection_gap.csv` files and you have
every breakdown in one dataframe.

## Interpreting the output

The headline number is the detection gap: baseline recall on real phishing
minus detection rate on synthetic. If it's positive and large (say > 0.20),
SpamAssassin is materially worse at catching AI-generated phishing than at
catching the real phishing it was designed against — the result the project
hypothesis predicts.

Break it down further by `method` to see whether one generation approach
(zero-shot, few-shot, rephrasing) is especially evasive. Prior work
(Afane et al.) found rephrasing to be the strongest evader.

## Resume after interruption

Both scoring commands are idempotent. If the script crashes or you kill
it partway, just re-run the same command — it skips ids already in the
output CSV. The output file is flushed every 50 emails, so you lose at
most a few dozen emails of work.

## Common issues

**"spamd not reachable"** — you didn't start `spamd`, or it's on a
different port. Check with `lsof -i :783` on macOS.

**Very slow (<2 emails/sec)** — you started `spamd` with
`--max-children=1` or you're running `spamassassin` binary instead of
`spamc`. Re-check setup.

**All scores ~0 or very low** — your rule set is missing. Re-run
`sudo sa-update --nogpg`.

**Many "no_spam_headers_in_output" errors** — spamc may be failing
silently. Try `echo "Subject: test" | spamc` and see if it returns
annotated output.
