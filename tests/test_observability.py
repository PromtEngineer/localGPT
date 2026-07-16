from backend.agent_runtime.observability import redact


def test_observability_redacts_secrets_recursively() -> None:
    value = {
        "authorization": "Bearer secret",
        "nested": {"api_key": "abc", "safe": "visible"},
    }

    assert redact(value) == {
        "authorization": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]", "safe": "visible"},
    }


def test_observability_redacts_tokens_inside_error_text() -> None:
    assert redact("upstream rejected Bearer secret.token-value") == (
        "upstream rejected Bearer [REDACTED]"
    )
