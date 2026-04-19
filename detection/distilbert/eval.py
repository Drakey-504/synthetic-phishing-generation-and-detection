"""
detection/distilbert/eval.py

Evaluate a fine-tuned DistilBERT model on the real test set and the synthetic
phishing set. Writes the four standard result CSVs.

Loads model weights from models/distilbert/<mode>/.
Writes results to results/distilbert/<mode>/.

Usage (Colab or local):
    python detection/distilbert/eval.py --mode baseline
    python detection/distilbert/eval.py --mode augmented
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _shared  # noqa: E402


DETECTOR = "distilbert"
THRESHOLD = 0.5
MAX_LENGTH = 512


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], device: str,
            batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return (proba_phishing, pred_label) for a list of texts."""
    model.eval()
    all_probas: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probas.extend(proba.tolist())
        if i % (batch_size * 20) == 0 and i > 0:
            print(f"  {i}/{len(texts)}", flush=True)
    proba_arr = np.array(all_probas, dtype=float)
    pred_arr = (proba_arr >= THRESHOLD).astype(int)
    return proba_arr, pred_arr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["baseline", "augmented"])
    parser.add_argument("--model-dir", type=Path,
                        default=Path("models/distilbert"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/distilbert"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_path = args.model_dir / args.mode
    if not model_path.exists() or not (model_path / "config.json").exists():
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        print(f"Run: python detection/distilbert/train.py --mode {args.mode}",
              file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)

    out_dir = args.output_dir / args.mode

    # Real test ------------------------------------------------------------
    _, real_test_df = _shared.load_real()
    print(f"\nScoring real test set ({len(real_test_df)} rows)...")
    real_proba, real_pred = predict(
        model, tokenizer,
        real_test_df["text"].fillna("").astype(str).tolist(),
        device, args.batch_size,
    )
    y_real = _shared.encode_labels(real_test_df["label"])
    real_metrics = _shared.classification_metrics(y_real, real_pred)
    print(f"  Accuracy:  {real_metrics['accuracy']:.4f}")
    print(f"  Precision: {real_metrics['precision']:.4f}")
    print(f"  Recall:    {real_metrics['recall']:.4f}")
    print(f"  F1:        {real_metrics['f1']:.4f}")
    print(f"  Confusion: TP={real_metrics['tp']} FP={real_metrics['fp']} "
          f"TN={real_metrics['tn']} FN={real_metrics['fn']}")

    _shared.write_per_email_results(
        real_test_df, real_pred, real_proba,
        out_dir / "real_test.csv",
    )
    _shared.write_baseline(DETECTOR, THRESHOLD, real_metrics,
                           out_dir / "baseline.csv")

    # Synthetic ------------------------------------------------------------
    if args.mode == "baseline":
        synth_df = _shared.load_synthetic()
        print(f"\nScoring synthetic set: full {len(synth_df)} rows (Phase 3).")
    else:
        _, synth_df = _shared.load_synthetic_split()
        print(f"\nScoring synthetic set: {len(synth_df)} held-out rows (Phase 4).")

    synth_proba, synth_pred = predict(
        model, tokenizer,
        synth_df["text"].fillna("").astype(str).tolist(),
        device, args.batch_size,
    )
    synth_caught = int(synth_pred.sum())
    synth_rate = synth_caught / len(synth_pred) if len(synth_pred) else 0.0

    print(f"  Caught:         {synth_caught} / {len(synth_df)}")
    print(f"  Detection rate: {synth_rate:.4f}")
    print(f"  Gap vs real:    {real_metrics['recall'] - synth_rate:+.4f}")

    _shared.write_per_email_results(
        synth_df, synth_pred, synth_proba,
        out_dir / "synthetic.csv",
    )

    real_out = pd.read_csv(out_dir / "real_test.csv")
    synth_out = pd.read_csv(out_dir / "synthetic.csv")
    _shared.write_detection_gap(
        DETECTOR, real_out, synth_out,
        out_dir / "detection_gap.csv",
    )

    print(f"\n✓ Wrote results to {out_dir}/")
    for fname in ["real_test.csv", "synthetic.csv", "baseline.csv", "detection_gap.csv"]:
        print(f"    {out_dir / fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
