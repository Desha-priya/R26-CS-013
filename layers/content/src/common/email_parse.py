"""
src/common/email_parse.py
Shared, dependency-free email parsing for every NeuraShield email intake
(IMAP watcher, Gmail API watcher, .eml drop-folder, manual .eml upload).

WHY THIS EXISTS
    The original IMAP watcher only read the text/plain part of an email.
    A large share of real phishing is HTML-only (no plain-text part), so
    those messages were skipped as "empty body" and never scored. This
    module adds an HTML -> visible-text fallback. It is only used when no
    usable text/plain part exists, so emails that already worked produce
    exactly the same text as before.

SAFETY
    - Pure stdlib (email + html.parser). Nothing is executed or fetched.
    - <script>, <style>, <head> content is dropped; hidden-text tricks are
      not interpreted (we only want what a human would plausibly read).
    - Output is length-capped so a hostile 50 MB email cannot exhaust memory
      in the tokenizer.
"""

import re
from email.header import decode_header, make_header
from html.parser import HTMLParser

MAX_BODY_CHARS = 20000          # model only reads 256 tokens; this is a safety cap
_SKIP_TAGS = {"script", "style", "head", "title", "noscript"}
_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4",
               "h5", "h6", "table", "blockquote", "hr"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Visible text of an HTML string. Never raises."""
    if not html:
        return ""
    try:
        p = _TextExtractor()
        p.feed(html[:MAX_BODY_CHARS * 5])
        p.close()
        text = "".join(p.parts)
    except Exception:
        # malformed HTML: crude tag strip is better than nothing
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:MAX_BODY_CHARS]


def decode_header_value(value) -> str:
    """RFC 2047 header -> str. Never raises."""
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value))))
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""


def _part_text(part) -> str:
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        return payload.decode(part.get_content_charset() or "utf-8",
                              errors="ignore")
    except Exception:
        return ""


def html_fallback_text(msg) -> str:
    """Visible text from the HTML part(s) of a message ('' if none)."""
    chunks = []
    try:
        parts = msg.walk() if msg.is_multipart() else [msg]
        for part in parts:
            if part.get_content_type() != "text/html":
                continue
            if "attachment" in str(part.get("Content-Disposition") or ""):
                continue
            t = html_to_text(_part_text(part))
            if t:
                chunks.append(t)
    except Exception:
        pass
    return "\n\n".join(chunks)[:MAX_BODY_CHARS]


def plain_text(msg) -> str:
    """First non-attachment text/plain part ('' if none)."""
    try:
        parts = msg.walk() if msg.is_multipart() else [msg]
        for part in parts:
            if part.get_content_type() != "text/plain":
                continue
            if "attachment" in str(part.get("Content-Disposition") or ""):
                continue
            t = _part_text(part)
            if t.strip():
                return t[:MAX_BODY_CHARS]
    except Exception:
        pass
    return ""


def extract_email(msg):
    """
    (subject, sender, body) for an email.message.Message.
    Body = text/plain if present, otherwise visible text of the HTML part.
    """
    subject = decode_header_value(msg.get("Subject"))
    sender = decode_header_value(msg.get("From"))
    body = plain_text(msg) or html_fallback_text(msg)
    return subject, sender, body


def build_scan_text(subject: str, body: str) -> str:
    """The exact string the text model scores (matches the IMAP watcher)."""
    return f"{subject}\n\n{body}".strip()
