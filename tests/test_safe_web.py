import pytest

from backend.agent_runtime.builtin_tools import UnsafeURL, validate_public_url


def test_web_fetch_rejects_loopback_and_private_addresses() -> None:
    for url in (
        "http://127.0.0.1/admin",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ):
        with pytest.raises(UnsafeURL):
            validate_public_url(url)
