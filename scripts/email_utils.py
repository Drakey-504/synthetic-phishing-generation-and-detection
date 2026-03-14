"""
email_utils.py
Shared utility functions for parsing email datasets.
"""

import email
import email.header
import html2text


# html2text converter setup
H2T = html2text.HTML2Text()
H2T.ignore_links = False
H2T.ignore_images = True
H2T.body_width = 0  # don't wrap lines


def extract_body(msg):
    """
    Extract plain text body from an email message object.
    Handles plain text, HTML, and multipart messages.
    """
    if msg.is_multipart():
        text_plain = None
        text_html = None

        for part in msg.walk():
            content_type = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            if content_type == "text/plain" and text_plain is None:
                text_plain = decode_payload(part)
            elif content_type == "text/html" and text_html is None:
                text_html = decode_payload(part)

        if text_plain:
            return clean_text(text_plain)
        elif text_html:
            return clean_text(strip_html(text_html))
        else:
            return ""
    else:
        content_type = msg.get_content_type()
        payload = decode_payload(msg)

        if not payload:
            return ""

        if content_type == "text/html":
            return clean_text(strip_html(payload))
        else:
            return clean_text(payload)


def decode_payload(part):
    """Decode an email part's payload to a string."""
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")
    except Exception:
        return ""


def strip_html(html_content):
    """Convert HTML to plain text."""
    try:
        return H2T.handle(html_content)
    except Exception:
        return html_content


def clean_text(text):
    """Normalize whitespace and strip edges."""
    if not text:
        return ""
    lines = text.strip().split("\n")
    cleaned_lines = [line.strip() for line in lines]
    text = "\n".join(cleaned_lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def extract_subject(msg):
    """Extract and decode the Subject header."""
    subject = msg.get("Subject", "")
    if subject:
        decoded_parts = email.header.decode_header(subject)
        parts = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                charset = charset or "utf-8"
                try:
                    parts.append(part.decode(charset, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    parts.append(part.decode("utf-8", errors="replace"))
            else:
                parts.append(part)
        subject = " ".join(parts)
    return subject.strip()


def parse_email_file(filepath):
    """Parse a single email file (Enron / SpamAssassin format)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            msg = email.message_from_file(f)
        subject = extract_subject(msg)
        body = extract_body(msg)
        return subject, body
    except Exception as e:
        print(f"  Warning: failed to parse {filepath}: {e}")
        return "", ""
