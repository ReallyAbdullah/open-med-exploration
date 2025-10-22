import re

SAFE_HARBOR_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    re.compile(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?){2,}\d{2,4}\b"),
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    re.compile(r"https?://\S+"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),
]


def safe_harbor_redact(text: str) -> str:
    red = text
    for pat in SAFE_HARBOR_PATTERNS:
        red = pat.sub("[REDACTED]", red)
    return red
