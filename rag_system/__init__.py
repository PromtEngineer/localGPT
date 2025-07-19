import logging
import os

# ---------------------------------------------------------
# Global logging setup for the entire `rag_system` package.
# ---------------------------------------------------------
# You can control verbosity with an env variable, e.g.:
#   export RAG_LOG_LEVEL=DEBUG  (or INFO / WARNING / ERROR)
# If not set, we default to INFO to avoid excessive noise.
# ---------------------------------------------------------
_level_str = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_str, logging.INFO)

# Only configure root logger if it hasn't been configured yet
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
else:
    logging.getLogger().setLevel(_level)

logging.getLogger(__name__).debug(
    "Initialized rag_system logging (level=%s)", _level_str
)

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