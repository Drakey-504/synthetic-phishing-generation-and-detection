"""
Cleanup script for synthetic_phishing.csv

Fixes three issues found during QA:
  1. Trailing meta-commentary in rephrasing outputs (e.g., "Note: The rewritten email...")
  2. Unfilled template placeholders (e.g., [Recipient], [Your Name], [amount])
  3. Exact-duplicate rows in few_shot password_reset scenario

Usage:
    python cleanup_synthetic_phishing.py \
        --input  /mnt/user-data/uploads/synthetic_phishing.csv \
        --output /mnt/user-data/outputs/synthetic_phishing_clean.csv
"""

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Meta-note stripping
# ---------------------------------------------------------------------------
# Matches a trailing block that starts with "Note:" / "(Note:" / "Disclaimer:"
# preceded by a blank line, and extends to the end of the string.
# We are deliberately strict about the blank-line prefix so we do NOT strip
# legitimate in-character notes that appear mid-email (e.g. "Note: This file
# is available for 3 days." in zs_bulk_document_share_8).
META_NOTE_RE = re.compile(
    r"""
    \n\s*\n                      # blank line before the note block
    \(?                          # optional opening paren
    (?:Note|Disclaimer|Important)# keyword
    \s*:                         # colon
    .*\Z                         # ...to end of string
    """,
    re.DOTALL | re.VERBOSE | re.IGNORECASE,
)

# Known meta-tells. If the note block contains ANY of these, strip it.
# If it doesn't match any, the note is probably in-character and we keep it.
META_TELLS = re.compile(
    r"(?i)("
    r"rewritten|rephrased|revised|"
    r"i(?:'ve| have) (?:rewritten|rephrased|removed|kept|replaced|made)|"
    r"maintains? the (?:core|same|original)|"
    r"(?:realistic|fake)[- ]?(?:looking)? (?:url|link|value|email|fake)|"
    r"original(?:'s)? (?:tone|url|email|intent)|"
    r"call to action|"
    r"deceptive request|"
    r"all caps|excessive urgency|grammatical errors"
    r")"
)


def strip_meta_note(text: str) -> tuple[str, bool]:
    """Return (cleaned_text, was_stripped)."""
    m = META_NOTE_RE.search(text)
    if not m:
        return text, False
    note_block = m.group(0)
    if not META_TELLS.search(note_block):
        # Looks like an in-character note, leave it alone.
        return text, False
    return text[: m.start()].rstrip() + "\n", True


# ---------------------------------------------------------------------------
# 2. Placeholder filling — category-based pools
# ---------------------------------------------------------------------------
# Each placeholder keyword is mapped to a CATEGORY. At fill-time we pick a
# value at random from that category's POOL, using a per-row seed derived
# from the row ID so results are deterministic & reproducible.
#
# Pools are sized so that across the 76 rows needing fixes, no single value
# should appear more than ~5–8 times. That's well below what TF-IDF would
# pick up as a discriminative feature.

POOLS = {
    "recipient_name": [
        "Valued Customer", "Dear Customer", "Account Holder", "Dear User",
        "Valued Member", "Dear Subscriber", "Dear Client", "Dear Member",
        "Valued Client", "Dear Account Holder",
    ],
    "recipient_email": [
        "user@example.com", "customer@mail.com", "account.holder@webmail.net",
        "member@inbox.com", "subscriber@email.net", "client@mailservice.com",
        "contact@webmail.org", "user@mailbox.net", "account@emailservice.com",
        "holder@mail.net",
    ],
    "amount": [
        "$1,250.00", "$2,450.00", "$875.50", "$3,200.00", "$1,899.99",
        "$4,150.00", "$725.00", "$2,975.00", "$1,540.00", "$3,680.00",
        "$950.00", "$2,100.00",
    ],
    "phone": [
        "+1 (555) 018-4472", "+1 (888) 237-6091", "+1 (800) 412-8853",
        "+1 (877) 604-2319", "+1 (866) 945-7728", "+1 (844) 351-8806",
        "+1 (855) 729-4460", "+1 (833) 188-5517",
    ],
    "url": [
        "https://secure-portal-verify.com/login",
        "https://account-services-online.com/verify",
        "https://myaccount-portal.net/access",
        "https://secure-login-center.com/auth",
        "https://customer-verification.net/confirm",
        "https://account-center-online.com/login",
        "https://portal-access-verify.com/session",
        "https://secure-customer-portal.net/signin",
    ],
    "company": [
        "Global Services Inc.", "Premier Solutions Ltd.",
        "Atlas Enterprises LLC", "Summit Group International",
        "Meridian Holdings Inc.", "Pinnacle Consulting Group",
        "Northbridge Partners", "Vanguard Services Co.",
    ],
    "date": [
        "March 22, 2025", "April 8, 2025", "May 15, 2025", "June 3, 2025",
        "July 19, 2025", "August 27, 2025", "September 11, 2025",
        "October 5, 2025", "February 14, 2025", "November 2, 2025",
    ],
    "contact_email": [
        "support@company-services.com", "help@account-services.net",
        "contact@customer-support.com", "info@services-online.net",
        "admin@support-center.com", "assistance@clientservices.net",
    ],
}

# Map normalized placeholder key -> pool name.
CATEGORY_MAP = {
    # Recipient names
    "recipient": "recipient_name",
    "recipient's name": "recipient_name",
    "recipients name": "recipient_name",
    "recipient name": "recipient_name",
    "name": "recipient_name",
    "your name": "recipient_name",

    # Recipient emails
    "recipient's email": "recipient_email",
    "recipients email": "recipient_email",
    "recipient email": "recipient_email",
    "email": "recipient_email",
    "email address": "recipient_email",

    # Money
    "amount": "amount",
    "insert amount": "amount",

    # Contact info
    "insert contact information": "contact_email",
    "contact information": "contact_email",
    "phone": "phone",
    "insert phone": "phone",

    # Company / link
    "company": "company",
    "your company": "company",
    "link": "url",
    "insert link": "url",
    "url": "url",
    "insert url": "url",

    # Dates
    "date": "date",
    "insert date": "date",
}

# Keyword-based fallback (for placeholders NOT in CATEGORY_MAP).
# Order matters — more specific patterns first.
FALLBACK_RULES = [
    (re.compile(r"(?i)email"),       "recipient_email"),
    (re.compile(r"(?i)phone|tel|contact\s*number"), "phone"),
    (re.compile(r"(?i)url|link"),    "url"),
    (re.compile(r"(?i)amount|\$"),   "amount"),
    (re.compile(r"(?i)date"),        "date"),
    (re.compile(r"(?i)name|recipient|partner|investor|director|owner|contact\s*person"),
     "recipient_name"),
    (re.compile(r"(?i)company|org|firm"), "company"),
]

# Second-pass handling for brackets that aren't explicit template slots:
#   [John Smith]   -> John Smith          (strip brackets, keep proper name)
#   [Mr. Johnson]  -> Mr. Johnson
#   [www.foo.com]  -> www.foo.com         (strip brackets on URLs)
#   [11]           -> 11                  (strip brackets on bare numbers)
#   [briefly summarize any notable changes] -> (dropped entirely — prompt-leak)
#   [not applicable] -> (replaced with fallback value or dropped)
PROPER_NAME_RE = re.compile(
    r"^(?:mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$|"
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$",
    re.IGNORECASE,
)
URL_LIKE_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
PURE_NUMBER_RE = re.compile(r"^\d+$")
INSTRUCTION_LIKE_RE = re.compile(
    r"\b(briefly|summarize|describe|list|not\s+applicable)\b", re.IGNORECASE
)

# Find a bracketed token that is NOT a markdown link [text](url).
# We use a negative lookahead for the opening paren.
PLACEHOLDER_RE = re.compile(r"\[([^\]\n]{1,60})\](?!\()")


def _make_rng(row_id: str, seed: int) -> random.Random:
    """Deterministic per-row RNG so the same row always yields the same fillers."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    # First 8 bytes -> a stable integer seed for random.Random.
    return random.Random(int.from_bytes(digest[:8], "big"))


def fill_placeholders(
    text: str, row_id: str, seed: int = 42
) -> tuple[str, int]:
    """Replace [placeholders] with plausible values drawn from per-category pools.

    Sampling is deterministic per (row_id, seed) so outputs are reproducible.
    Within a single row, independent draws are taken for each placeholder
    so repeated placeholders of the same category don't collapse to one value.
    """
    filled = 0
    rng = _make_rng(row_id, seed)

    def pick(category: str) -> str:
        return rng.choice(POOLS[category])

    def _replace(match: re.Match) -> str:
        nonlocal filled
        inner = match.group(1).strip()
        key = inner.lower()

        # --- Pass 1: exact category-map hit -----------------------------
        if key in CATEGORY_MAP:
            filled += 1
            return pick(CATEGORY_MAP[key])

        # Try "insert X" -> "X"
        if key.startswith("insert "):
            sub = key[len("insert ") :]
            if sub in CATEGORY_MAP:
                filled += 1
                return pick(CATEGORY_MAP[sub])

        # --- Pass 2: keyword fallback -----------------------------------
        for pattern, category in FALLBACK_RULES:
            if pattern.search(inner):
                filled += 1
                return pick(category)

        # --- Pass 3: structural rules -----------------------------------
        # Proper name like "[John Smith]" or "[Mr. Johnson]" -> strip brackets.
        if PROPER_NAME_RE.match(inner):
            filled += 1
            return inner

        # URL-looking content -> strip brackets.
        if URL_LIKE_RE.match(inner):
            filled += 1
            return inner

        # Bare number -> strip brackets.
        if PURE_NUMBER_RE.match(inner):
            filled += 1
            return inner

        # Prompt-leak instruction like "[briefly summarize any notable changes]"
        # -> drop the whole bracketed span.
        if INSTRUCTION_LIKE_RE.search(inner):
            filled += 1
            return ""

        # Unknown — leave untouched for audit.
        return match.group(0)

    new_text = PLACEHOLDER_RE.sub(_replace, text)

    # Clean up any double-spaces / stranded punctuation left by Pass-3 drops.
    new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    new_text = re.sub(r" +([,.!?])", r"\1", new_text)

    # Fix doubled greetings created when the source email had "Dear [Recipient]"
    # and we filled the placeholder with another greeting word. Collapses
    # "Dear Dear Customer" -> "Dear Customer", "Hi Hello User" -> "Hello User",
    # "Dear Valued Customer" is fine ("Valued" isn't a greeting) and is left alone.
    GREETING_WORDS = r"(?:Dear|Hi|Hello|Greetings|Hey)"
    new_text = re.sub(
        rf"\b{GREETING_WORDS}\s+({GREETING_WORDS})\b",
        r"\1",
        new_text,
    )

    return new_text, filled


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame, seed: int = 42, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    stats = {
        "input_rows": len(df),
        "meta_notes_stripped": 0,
        "placeholders_filled": 0,
        "rows_with_placeholders_fixed": 0,
        "unresolved_placeholder_rows": 0,
        "duplicate_rows_dropped": 0,
        "output_rows": 0,
    }

    # --- 1. Strip trailing meta-notes -------------------------------------
    stripped_flags = []
    new_texts = []
    for txt in df["text"]:
        new_txt, was_stripped = strip_meta_note(txt)
        new_texts.append(new_txt)
        stripped_flags.append(was_stripped)
    df = df.copy()
    df["text"] = new_texts
    stats["meta_notes_stripped"] = sum(stripped_flags)

    # --- 2. Fill placeholders (deterministic per row_id) ------------------
    fill_counts = []
    new_texts = []
    for row_id, txt in zip(df["id"], df["text"]):
        new_txt, count = fill_placeholders(txt, row_id=row_id, seed=seed)
        new_texts.append(new_txt)
        fill_counts.append(count)
    df["text"] = new_texts
    stats["placeholders_filled"] = sum(fill_counts)
    stats["rows_with_placeholders_fixed"] = sum(1 for c in fill_counts if c > 0)

    # Audit: how many rows still have unresolved [...] tokens?
    still_has = df["text"].apply(lambda t: bool(PLACEHOLDER_RE.search(t))).sum()
    stats["unresolved_placeholder_rows"] = int(still_has)

    # --- 3. Drop exact-duplicate text rows --------------------------------
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    stats["duplicate_rows_dropped"] = before - len(df)

    stats["output_rows"] = len(df)
    return df, stats


def print_report(stats: dict) -> None:
    print("=" * 60)
    print("CLEANUP REPORT")
    print("=" * 60)
    print(f"Input rows:                        {stats['input_rows']}")
    print(f"Meta-notes stripped:               {stats['meta_notes_stripped']}")
    print(f"Placeholders filled:               {stats['placeholders_filled']}")
    print(f"Rows with placeholders fixed:      {stats['rows_with_placeholders_fixed']}")
    print(f"Rows still containing [brackets]:  {stats['unresolved_placeholder_rows']}")
    print(f"Duplicate rows dropped:            {stats['duplicate_rows_dropped']}")
    print(f"Output rows:                       {stats['output_rows']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for placeholder filler sampling (default: 42).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    clean_df, stats = clean(df, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)

    print_report(stats)
    print(f"\nWrote cleaned CSV -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
