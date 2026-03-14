"""
generate_phishing.py
Generates synthetic phishing emails using Llama 3 via Ollama.

Three generation methods:
    1. Zero-shot: prompt describes scenario, no examples
    2. Few-shot: real phishing examples provided before generation instruction
    3. Rephrasing: rewrites real phishing emails to sound more professional

Usage:
    python scripts/generate_phishing.py

Output:
    data/processed/synthetic_phishing.csv
"""

import json
import csv
import time
import random
import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────────

MODEL = "llama3"
PROMPTS_DIR = Path("generation/prompts")
CLEAN_DATA = Path("data/processed/emails_clean.csv")
OUTPUT = Path("data/processed/synthetic_phishing.csv")

# How many emails to generate per prompt template
EMAILS_PER_ZERO_SHOT = 10
EMAILS_PER_FEW_SHOT = 10
MAX_REPHRASE = 200  # number of real phishing emails to rephrase

SEED = 42
random.seed(SEED)


# ── Helpers ─────────────────────────────────────────────────────────────────

def call_llm(system_prompt, user_prompt, temperature=0.8):
    """Call Ollama and return the response text."""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": temperature}
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def parse_email_response(response_text):
    """
    Parse the LLM response to extract Subject and Body.
    Handles various formats the LLM might output.
    """
    if not response_text:
        return None, None

    subject = ""
    body = ""

    lines = response_text.strip().split("\n")

    # Find subject line
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            # Everything after the subject line (skipping blank lines) is the body
            body_lines = []
            found_body_start = False
            for remaining in lines[i + 1:]:
                if remaining.strip().lower().startswith("body:"):
                    body_lines.append(remaining.split(":", 1)[1].strip())
                    found_body_start = True
                elif found_body_start or remaining.strip():
                    found_body_start = True
                    body_lines.append(remaining)
            body = "\n".join(body_lines).strip()
            break

    # Fallback: if no Subject: found, use first line as subject, rest as body
    if not subject and lines:
        subject = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

    return subject, body


# ── Method 1: Zero-Shot Generation ──────────────────────────────────────────

def generate_zero_shot():
    """Generate phishing emails using zero-shot prompts."""
    print("\n" + "=" * 60)
    print("METHOD 1: ZERO-SHOT GENERATION")
    print("=" * 60)

    with open(PROMPTS_DIR / "zero_shot.json") as f:
        prompts = json.load(f)

    records = []

    for category in ["zero_shot_bulk", "zero_shot_spear"]:
        templates = prompts[category]
        print(f"\n  Category: {category} ({len(templates)} templates × {EMAILS_PER_ZERO_SHOT} each)")

        for template in templates:
            print(f"    Generating: {template['id']}...")

            for j in range(EMAILS_PER_ZERO_SHOT):
                # Vary temperature slightly for diversity
                temp = random.uniform(0.7, 0.95)
                response = call_llm(template["system"], template["user"], temperature=temp)

                if response:
                    subject, body = parse_email_response(response)
                    if body and len(body) > 20:
                        records.append({
                            "id": f"{template['id']}_{j}",
                            "method": "zero_shot",
                            "sophistication": template["sophistication"],
                            "scenario": template["scenario"],
                            "prompt_id": template["id"],
                            "model": MODEL,
                            "temperature": round(temp, 2),
                            "subject": subject,
                            "text": body,
                            "label": "phishing"
                        })

                # Small delay to avoid overwhelming Ollama
                time.sleep(0.5)

    print(f"\n  Zero-shot total: {len(records)} emails generated")
    return records


# ── Method 2: Few-Shot Generation ───────────────────────────────────────────

def generate_few_shot():
    """Generate phishing emails using few-shot prompts with real examples."""
    print("\n" + "=" * 60)
    print("METHOD 2: FEW-SHOT GENERATION")
    print("=" * 60)

    with open(PROMPTS_DIR / "few_shot.json") as f:
        prompts = json.load(f)

    # Build the examples block
    examples_text = ""
    for i, ex in enumerate(prompts["examples"], 1):
        examples_text += f"\nExample {i}:\nSubject: {ex['subject']}\nBody: {ex['body']}\n"

    system_prompt = prompts["system"]
    records = []

    for category in ["few_shot_bulk", "few_shot_spear"]:
        templates = prompts[category]
        print(f"\n  Category: {category} ({len(templates)} templates × {EMAILS_PER_FEW_SHOT} each)")

        for template in templates:
            print(f"    Generating: {template['id']}...")

            # Combine examples with the generation instruction
            user_prompt = examples_text + "\n" + template["user"]

            for j in range(EMAILS_PER_FEW_SHOT):
                temp = random.uniform(0.7, 0.95)
                response = call_llm(system_prompt, user_prompt, temperature=temp)

                if response:
                    subject, body = parse_email_response(response)
                    if body and len(body) > 20:
                        records.append({
                            "id": f"{template['id']}_{j}",
                            "method": "few_shot",
                            "sophistication": template["sophistication"],
                            "scenario": template["scenario"],
                            "prompt_id": template["id"],
                            "model": MODEL,
                            "temperature": round(temp, 2),
                            "subject": subject,
                            "text": body,
                            "label": "phishing"
                        })

                time.sleep(0.5)

    print(f"\n  Few-shot total: {len(records)} emails generated")
    return records


# ── Method 3: LLM Rephrasing ───────────────────────────────────────────────

def generate_rephrased():
    """Rephrase real phishing emails to sound more professional."""
    print("\n" + "=" * 60)
    print("METHOD 3: LLM REPHRASING")
    print("=" * 60)

    with open(PROMPTS_DIR / "rephrasing.json") as f:
        config = json.load(f)

    system_prompt = config["system"]

    # Load real phishing emails from the training set
    df = pd.read_csv(CLEAN_DATA)
    phishing_train = df[(df["label"] == "phishing") & (df["split"] == "train")]

    # Sample emails to rephrase — mix of both sources
    nazario = phishing_train[phishing_train["source"] == "nazario"]
    nigerian = phishing_train[phishing_train["source"] == "nigerian"]

    # Take more from Nigerian (longer, more varied) and some from Nazario
    n_nazario = min(80, len(nazario))
    n_nigerian = min(MAX_REPHRASE - n_nazario, len(nigerian))

    sample = pd.concat([
        nazario.sample(n=n_nazario, random_state=SEED),
        nigerian.sample(n=n_nigerian, random_state=SEED)
    ])

    print(f"  Rephrasing {len(sample)} real phishing emails ({n_nazario} Nazario + {n_nigerian} Nigerian)")

    records = []

    for idx, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="  Rephrasing")):
        original_subject = row["subject"] if pd.notna(row["subject"]) else "(no subject)"
        original_body = row["text"][:3000]  # Cap to avoid token limits

        user_prompt = f"Original email:\nSubject: {original_subject}\nBody: {original_body}\n\nRewritten email:"

        response = call_llm(system_prompt, user_prompt, temperature=0.7)

        if response:
            subject, body = parse_email_response(response)
            if body and len(body) > 20:
                records.append({
                    "id": f"rephrase_{row['source']}_{idx}",
                    "method": "rephrasing",
                    "sophistication": "rephrased",
                    "scenario": "rephrased_real",
                    "prompt_id": f"rephrase_{row['id']}",
                    "model": MODEL,
                    "temperature": 0.7,
                    "subject": subject,
                    "text": body,
                    "label": "phishing"
                })

        time.sleep(0.3)

    print(f"\n  Rephrasing total: {len(records)} emails generated")
    return records


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SYNTHETIC PHISHING EMAIL GENERATION")
    print(f"Model: {MODEL}")
    print("=" * 60)

    all_records = []

    # Method 1: Zero-shot
    all_records.extend(generate_zero_shot())

    # Method 2: Few-shot
    all_records.extend(generate_few_shot())

    # Method 3: Rephrasing
    all_records.extend(generate_rephrased())

    # Save to CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "method", "sophistication", "scenario", "prompt_id",
                  "model", "temperature", "subject", "text", "label"]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # Summary
    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total synthetic emails: {len(all_records)}")

    method_counts = {}
    for r in all_records:
        key = f"{r['method']} ({r['sophistication']})"
        method_counts[key] = method_counts.get(key, 0) + 1

    print(f"\nBy method × sophistication:")
    for key, count in sorted(method_counts.items()):
        print(f"  {key}: {count}")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
