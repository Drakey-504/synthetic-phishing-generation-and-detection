"""
eval.py
Evaluate SpamAssassin on a corpus of emails by talking to spamd directly.

This script:
  1. Reads an input CSV (real test set OR synthetic phishing set).
  2. Synthesizes minimal RFC 5322 headers per email so SpamAssassin's
     header-based rules can fire.
  3. Sends each email to spamd over TCP using the SPAMC protocol
     (REPORT command). No local spamc binary required.
  4. Parses the Spam: header and rule breakdown returned by spamd.
  5. Writes a per-email results CSV with score, prediction, and fired rules.

Usage (real test set):
    python detection/spamassassin/eval.py \
        --input  data/processed/emails_clean.csv \
        --output results/spamassassin/real_test.csv \
        --split  test

Usage (synthetic phishing):
    python detection/spamassassin/eval.py \
        --input  data/processed/synthetic_phishing_clean.csv \
        --output results/spamassassin/synthetic.csv

Resume: if --output already exists, previously-scored ids are skipped and
new results are appended. Safe to re-run after an interruption.

Prerequisites: spamd must be reachable at --host:--port (default
127.0.0.1:783). See setup.md for the Docker-based setup.
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import hashlib
import random
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# RFC 5322 header synthesis
# ---------------------------------------------------------------------------
# We synthesize plausible headers so SpamAssassin can exercise its header-based
# rules (From spoofing checks, Date sanity, etc.). Values are drawn from small
# pools seeded deterministically by row id for reproducibility.

FROM_DOMAINS = [
    "service-notifications.com", "account-alerts.net", "secure-mail.org",
    "customer-support.net", "notifications.com", "mail-delivery.net",
    "corporate-services.com", "info-updates.net",
]
FROM_LOCALPARTS = [
    "noreply", "support", "notifications", "service", "info",
    "alerts", "no-reply", "account", "security", "team",
]
TO_ADDRESS = "recipient@example.com"


def _rng(row_id: str, seed: int = 42) -> random.Random:
    """Deterministic per-row RNG (same row_id -> same headers every run)."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def synthesize_email(row_id: str, subject: str, body: str,
                     seed: int = 42) -> bytes:
    """Build a minimal RFC 5322 plain-text message for this row.

    We deliberately do NOT make the headers 'spammy' (no suspicious sender
    patterns, no malformed Date). The goal is to let SpamAssassin score the
    *content*, which is what the detection-gap experiment is measuring.
    """
    rng = _rng(row_id, seed)

    local = rng.choice(FROM_LOCALPARTS)
    domain = rng.choice(FROM_DOMAINS)
    from_addr = f"{local}@{domain}"

    # Date: random point in the last 365 days (SpamAssassin flags future /
    # very old dates).
    days_ago = rng.randint(1, 365)
    date_hdr = email.utils.formatdate(
        timeval=time.time() - days_ago * 86400,
        localtime=False,
        usegmt=True,
    )

    msg_id = f"<{row_id}.{rng.randrange(10**10):010d}@{domain}>"

    # Normalize subject (strip newlines that would break header format).
    subj = (subject or "").replace("\r", " ").replace("\n", " ").strip() or "(no subject)"

    headers = (
        f"From: {from_addr}\r\n"
        f"To: {TO_ADDRESS}\r\n"
        f"Subject: {subj}\r\n"
        f"Date: {date_hdr}\r\n"
        f"Message-ID: {msg_id}\r\n"
        f"MIME-Version: 1.0\r\n"
        f"Content-Type: text/plain; charset=UTF-8\r\n"
        f"Content-Transfer-Encoding: 8bit\r\n"
    )

    # Body: normalize line endings to CRLF (RFC-compliant).
    body_norm = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    return (headers + "\r\n" + body_norm).encode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# spamd protocol client (replaces spamc subprocess)
# ---------------------------------------------------------------------------
# We talk directly to spamd over TCP using the SPAMC protocol. This avoids
# needing spamc installed locally — useful when spamd runs in Docker and the
# host machine has no SpamAssassin Perl distribution.
#
# Protocol reference: https://svn.apache.org/repos/asf/spamassassin/trunk/spamd/PROTOCOL
#
# Request format:
#     REPORT SPAMC/1.5\r\n
#     Content-length: <N>\r\n
#     \r\n
#     <N bytes of RFC 5322 message>
#
# Response format (REPORT):
#     SPAMD/1.5 0 EX_OK\r\n
#     Spam: True ; 7.5 / 5.0\r\n
#     \r\n
#     <report body with fired-rule breakdown>

SPAM_LINE_RE = re.compile(
    rb"^Spam:\s*(True|False)\s*;\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)",
    re.MULTILINE | re.IGNORECASE,
)
# Rule lines in REPORT output look like: " 2.5 RULE_NAME          Description"
# (optional leading whitespace, score, rule name in uppercase/underscores,
# then description).
FIRED_RULE_RE = re.compile(
    rb"^\s*-?\d+\.\d+\s+([A-Z][A-Z0-9_]{2,})\b", re.MULTILINE,
)


@dataclass
class SpamcResult:
    id: str
    sa_score: float | None
    sa_is_spam: bool | None
    fired_rules: str  # comma-separated
    error: str  # empty on success


def _recv_all(sock: socket.socket, max_bytes: int = 10 * 1024 * 1024) -> bytes:
    """Read until the server closes its half of the connection."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError(f"response exceeded {max_bytes} bytes")
    return b"".join(chunks)


def run_spamc(message: bytes, host: str, port: int,
              timeout: float = 30.0) -> SpamcResult:
    """Score a message by speaking the SPAMC protocol to spamd directly.

    One TCP connection per message. spamd handles concurrent connections
    natively (prefork workers), so the caller's thread pool parallelizes
    scoring without any special handling here.
    """
    # Build the REPORT request.
    request = (
        b"REPORT SPAMC/1.5\r\n"
        b"Content-length: " + str(len(message)).encode("ascii") + b"\r\n"
        b"\r\n"
        + message
    )

    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request)
            # Half-close so spamd knows the request is complete and starts
            # streaming the response.
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            response = _recv_all(sock)
    except (socket.timeout, TimeoutError):
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="", error="spamd_timeout")
    except OSError as e:
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="", error=f"spamd_conn:{e.__class__.__name__}:{e}")

    if not response:
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="", error="spamd_empty_response")

    # Status line check: "SPAMD/<ver> <code> <msg>"
    first_line = response.split(b"\n", 1)[0]
    if not first_line.startswith(b"SPAMD/"):
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="",
                           error=f"spamd_bad_status:{first_line[:80]!r}")
    parts = first_line.split()
    if len(parts) >= 2 and parts[1] != b"0":
        # Non-zero response code -> protocol error.
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="",
                           error=f"spamd_rc:{first_line.decode('latin1', 'replace')[:120]}")

    # Parse the Spam: header.
    m = SPAM_LINE_RE.search(response)
    if not m:
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="", error="spamd_no_spam_header")
    is_spam = m.group(1).lower() == b"true"
    try:
        score = float(m.group(2))
    except ValueError:
        return SpamcResult(id="", sa_score=None, sa_is_spam=None,
                           fired_rules="", error="spamd_bad_score")

    # Body starts after the first blank line; that's where REPORT writes the
    # rule breakdown.
    body_start = response.find(b"\r\n\r\n")
    if body_start == -1:
        body_start = response.find(b"\n\n")
        body = response[body_start + 2:] if body_start != -1 else b""
    else:
        body = response[body_start + 4:]

    rules = FIRED_RULE_RE.findall(body)
    # De-duplicate while preserving order.
    seen: set[bytes] = set()
    ordered: list[bytes] = []
    for r in rules:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    fired = ",".join(r.decode("ascii", errors="replace") for r in ordered)

    return SpamcResult(id="", sa_score=score, sa_is_spam=is_spam,
                       fired_rules=fired, error="")


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------
def spamd_is_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def already_scored_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["id"])
        return set(existing["id"].astype(str))
    except Exception:
        # Corrupt / partial file; caller will notice at append time.
        return set()


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "id", "label", "subject",
    "sa_score", "sa_is_spam", "sa_prediction",
    "fired_rules", "error",
]
SA_THRESHOLD = 5.0  # SpamAssassin default; anything >= this is classified spam.


def iter_rows(df: pd.DataFrame, skip_ids: set[str]) -> Iterator[dict]:
    for _, row in df.iterrows():
        rid = str(row["id"])
        if rid in skip_ids:
            continue
        yield {
            "id": rid,
            "label": row.get("label", ""),
            "subject": row.get("subject", ""),
            "text": row.get("text", "") or "",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Input CSV (must have id, subject, text, label).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV for per-email results.")
    parser.add_argument("--split", default=None,
                        help="If set, filter input rows where split==<value>.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=783)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N rows (for quick tests).")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-email spamc timeout (seconds).")
    args = parser.parse_args()

    # Sanity checks --------------------------------------------------------
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    if not spamd_is_reachable(args.host, args.port):
        print(
            f"ERROR: spamd not reachable at {args.host}:{args.port}.\n"
            f"If using Docker:\n"
            f"  docker ps                   # check container is running\n"
            f"  docker logs spamd           # check for startup errors\n"
            f"  docker start spamd          # if stopped\n"
            f"Or start a fresh container:\n"
            f"  docker run -d --name spamd -p 783:783 instantlinux/spamassassin",
            file=sys.stderr,
        )
        return 3

    # Load input -----------------------------------------------------------
    df = pd.read_csv(args.input)
    required = {"id", "subject", "text"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: input missing columns: {missing}", file=sys.stderr)
        return 2

    if args.split is not None:
        if "split" not in df.columns:
            print("ERROR: --split given but no 'split' column in input.", file=sys.stderr)
            return 2
        df = df[df["split"] == args.split].reset_index(drop=True)

    if "label" not in df.columns:
        df["label"] = ""
    df["subject"] = df["subject"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    # Resume ---------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    skip = already_scored_ids(args.output)
    todo = df[~df["id"].astype(str).isin(skip)].reset_index(drop=True)

    total = len(df)
    already = len(df) - len(todo)
    print(f"Input rows:        {total}")
    print(f"Already scored:    {already}")
    print(f"To process:        {len(todo)}")
    print(f"spamd:             {args.host}:{args.port}  |  workers={args.workers}")
    print(f"Threshold:         {SA_THRESHOLD}  |  timeout={args.timeout}s/email")
    if len(todo) == 0:
        print("Nothing to do.")
        return 0

    # Open output in append mode (write header only if file is new/empty) --
    write_header = (not args.output.exists()) or args.output.stat().st_size == 0
    f_out = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        f_out.flush()

    # Worker function ------------------------------------------------------
    def score_one(row: dict) -> SpamcResult:
        msg = synthesize_email(row["id"], row["subject"], row["text"],
                               seed=args.seed)
        res = run_spamc(msg, args.host, args.port, timeout=args.timeout)
        res.id = row["id"]
        return res

    # Run ------------------------------------------------------------------
    t0 = time.time()
    done = 0
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(score_one, row): row
                for row in iter_rows(todo, skip_ids=set())
            }
            for fut in as_completed(futures):
                row = futures[fut]
                res = fut.result()
                prediction = ""
                if res.sa_score is not None:
                    prediction = "phishing" if res.sa_score >= SA_THRESHOLD else "legitimate"
                if res.error:
                    errors += 1

                writer.writerow({
                    "id": res.id,
                    "label": row["label"],
                    "subject": row["subject"],
                    "sa_score": res.sa_score if res.sa_score is not None else "",
                    "sa_is_spam": "" if res.sa_is_spam is None else int(res.sa_is_spam),
                    "sa_prediction": prediction,
                    "fired_rules": res.fired_rules,
                    "error": res.error,
                })
                done += 1
                if done % 50 == 0 or done == len(todo):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    eta = (len(todo) - done) / rate if rate else 0
                    print(
                        f"  {done}/{len(todo)}  "
                        f"({rate:.1f}/s, ETA {eta/60:.1f} min, "
                        f"errors: {errors})",
                        flush=True,
                    )
                    f_out.flush()
    finally:
        f_out.close()

    print(f"\nDone. Wrote {args.output}")
    print(f"Errors: {errors} / {done}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
