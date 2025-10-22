from src.orchestrator.redaction import safe_harbor_redact


def test_redaction_removes_identifiers():
    sample = "John Doe, 03/04/2023, 555-123-4567, john@x.com"
    redacted = safe_harbor_redact(sample)
    assert "John Doe" not in redacted
    assert "03/04/2023" not in redacted
    assert "555-123-4567" not in redacted
    assert "john@x.com" not in redacted
