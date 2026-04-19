# SpamAssassin Detector

SpamAssassin is the rule-based industry-standard baseline. No training
needed — we score emails with the stock SA rule set and tune the decision
threshold on a labeled training sample (like the ML detectors do). This
gives SA a fair comparison: the default threshold of 5.0 is designed for
bulk spam, not phishing, and significantly under-recalls on this corpus.

## Files

- `setup.md` — one-time Docker setup for running `spamd`
- `eval.py` — scores a CSV of emails through `spamd`, writes per-email scores
- `gap.py` — tunes the threshold on training data, computes baseline and
  detection gap, writes the four standard result files

## Full run sequence

### 0. Setup (one-time)

Follow `setup.md` to start `spamd` in Docker. Leave the container running
for the duration of scoring.

### 1. Score the real training set (~20–30 min on 10,586 emails)

Used only to pick the decision threshold. Scored once; reused across any
analysis.

```bash
mkdir -p results/spamassassin

python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_train.csv \
    --split  train \
    --workers 8
```

### 2. Score the real test set (~3–5 min on 2,647 emails)

```bash
python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_test.csv \
    --split  test \
    --workers 8
```

### 3. Score the synthetic phishing set (~1 min on 457 emails)

```bash
python detection/spamassassin/eval.py \
    --input  data/processed/synthetic_phishing_clean.csv \
    --output results/spamassassin/synthetic.csv \
    --workers 8
```

### 4. Tune threshold and compute detection gap

```bash
python detection/spamassassin/gap.py \
    --train     results/spamassassin/real_train.csv \
    --real      results/spamassassin/real_test.csv \
    --synthetic results/spamassassin/synthetic.csv \
    --synthetic-meta data/processed/synthetic_phishing_clean.csv \
    --outdir    results/spamassassin/
```

This picks the F1-optimal threshold on the training data, re-classifies the
test and synthetic sets at that threshold, and writes the four standard
result files:

- **`real_test.csv`** / **`synthetic.csv`** — per-email results with
  standardized columns `id, label, subject, score, prediction`
  (matching the LR/XGBoost/DistilBERT schema).
- **`baseline.csv`** — one row: `detector, threshold, n, tp, fp, tn, fn,
  accuracy, precision, recall, f1` on the real test set.
- **`detection_gap.csv`** — long-form breakdown. `grouping` column takes
  values `real_baseline`, `overall`, `method`, `sophistication`, or
  `method_scenario`.

## Threshold tuning options

By default, `gap.py` picks the F1-optimal threshold on the training set.
To use a different criterion:

```bash
python detection/spamassassin/gap.py ... --objective precision_98
```

- `f1` (default) — maximize F1 on training data
- `precision_95` — highest recall subject to 95% precision on training data
- `precision_98` — highest recall subject to 98% precision (production-grade
  false-positive tolerance)

You can also skip tuning entirely by passing a fixed threshold:

```bash
python detection/spamassassin/gap.py ... --threshold 5.0
```

This reverts to the SA default (useful as an ablation to show how much the
tuning matters).

## Methodology note for the writeup

SpamAssassin's default threshold of 5.0 is tuned for bulk unsolicited
commercial email (what SA was designed against in the early 2000s).
Phishing scores cluster differently: even emails with clear phishing content
often hit 2–4 points from the rule set because modern phishing patterns
aren't in the stock rule set. Using the default threshold substantially
under-recalls on this task (~50% recall on real phishing).

Tuning on a labeled training sample is how any production SA deployment is
configured, and it's consistent with how the ML detectors in this project
use training data. The tuned threshold typically lands around 0.9–1.0 and
gives F1 in the 0.83–0.86 range on real phishing.

## Resume after interruption

All scoring commands are idempotent. If scoring crashes or you kill it
partway, just re-run the same command — it skips ids already in the output
CSV. The output file is flushed every 50 emails, so you lose at most a few
dozen emails of work.

## Common issues

**"spamd not reachable"** — container isn't running. `docker start spamd`.

**All scores ~0 or very low** — rules may be out of date:
```bash
docker exec spamd sa-update --nogpg
docker restart spamd
```

**Training step is very slow** — 10,586 emails at 3-5 emails/sec is 30–60 min
wall-clock. If that's too long, `--limit 3000` on the training scoring is
enough for a stable threshold estimate (the threshold value converges well
before 10k samples).
