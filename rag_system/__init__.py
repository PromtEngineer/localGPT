import logging
import os

from rag_system.utils.logging_utils import configure_logging, system_logger

# ---------------------------------------------------------
# Global logging setup for the entire `rag_system` package.
# ---------------------------------------------------------
# You can control verbosity with an env variable, e.g.:
#   export RAG_LOG_LEVEL=DEBUG  (or INFO / WARNING / ERROR)
# If not set, we default to INFO to avoid excessive noise.
# ---------------------------------------------------------
_level_str = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
configure_logging(_level_str)

system_logger.debug("initialized_rag_system_logging", log_level=_level_str)

# ---------------------------------------------------------
# Authenticate to Hugging Face Hub if a token is provided
# ---------------------------------------------------------
from typing import Optional


def _hf_auto_login() -> None:
    """Attempt to authenticate with Hugging Face Hub using an env token.

    We support both the new canonical env var name (HF_TOKEN) and the two
    historical variants to avoid breaking user setups. The login call is
    idempotent: if a cached token already exists, the hub library will simply
    reuse it, so it is safe to run on every import.
    """

    import os

    token: Optional[str] = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    if not token:
        logging.getLogger(__name__).debug("No Hugging Face token found in env; proceeding anonymously.")
        return

    try:
        from huggingface_hub import login as hf_login

        hf_login(token=token, add_to_git_credential=False)  # type: ignore
        logging.getLogger(__name__).info("Authenticated to Hugging Face Hub via env token.")
    except Exception as exc:  # pragma: no cover – best-effort login
        logging.getLogger(__name__).warning(
            "Failed to login to Hugging Face Hub automatically: %s", exc
        )


# Run on module import
_hf_auto_login() 