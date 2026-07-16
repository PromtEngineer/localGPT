import tempfile
from pathlib import Path

from backend.agent_runtime.skills import SkillStore


def test_skill_versions_are_immutable_and_declare_allowed_tools() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SkillStore(Path(temp_dir) / "skills.db")
        first = store.create(
            """---
name: analyst
description: Analyze grounded data
allowed_tools: [search_knowledge, tabular_analysis]
---
Always cite the supplied evidence.
"""
        )
        second = store.create_version(first.skill_id, first.content + "\nBe concise.")

        assert first.version != second.version
        assert store.get_version(first.skill_id, first.version).content == first.content
        assert second.allowed_tools == ["search_knowledge", "tabular_analysis"]
