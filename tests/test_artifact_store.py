import hashlib
import tempfile
from pathlib import Path

from backend.agent_runtime.artifacts import ArtifactStore


def test_artifact_content_is_addressed_and_tracks_provenance() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        store = ArtifactStore(root / "artifacts.db", root / "objects")

        artifact = store.put_bytes(
            b"grounded facts",
            filename="facts.txt",
            mime_type="text/plain",
            session_id="session-1",
            provenance={"source": "upload"},
        )

        assert artifact.sha256 == hashlib.sha256(b"grounded facts").hexdigest()
        assert store.read_bytes(artifact.id) == b"grounded facts"
        assert artifact.provenance == {"source": "upload"}
