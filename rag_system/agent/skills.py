"""Trusted, prompt-only skill packs.

A "skill" is a committed Markdown file that injects a reusable *instruction*
(report writing, contract comparison, incident analysis, ...) into the synthesis
prompt. Skills are prompts, never code — there is nothing to execute.

Security model (deliberately strict):
- **Allowlisted source only.** Skills load exclusively from the trusted skills
  directory (repo `skills/`, overridable via `LOCALGPT_SKILLS_DIR`). Arbitrary
  user-uploaded Markdown is NEVER loaded as a skill.
- **Name, not path.** A request selects a skill by its id (the validated file
  stem). No path, glob, or content ever crosses the request boundary, so there
  is no traversal surface.
- **Bounded.** Per-file size cap; only `*.md` files directly inside the dir
  (no recursion, no symlink escape — each path is containment-checked).
- **Subordinate to grounding.** The injected text guides tone/structure; the
  caller keeps the citation/grounding rules above it (see chat synthesis).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

_MAX_SKILL_BYTES = 16_384
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,48}$")


def skills_dir() -> Path:
    """The single allowlisted skills directory."""
    override = os.getenv("LOCALGPT_SKILLS_DIR")
    if override:
        return Path(override)
    # repo-root/skills (this file is rag_system/agent/skills.py)
    return Path(__file__).resolve().parents[2] / "skills"


@dataclass(frozen=True)
class SkillPack:
    id: str
    name: str
    description: str
    version: str
    body: str


def parse_skill(text: str, skill_id: str) -> Optional[SkillPack]:
    """Parse a skill file's frontmatter + body. Returns None (skip) on anything
    malformed — a bad skill must never break loading of the others."""
    if not _ID_RE.match(skill_id):
        return None
    if not text.startswith("---"):
        return None  # frontmatter is required
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    meta: Dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("\"'")
    body = parts[2].strip()
    name = meta.get("name", "").strip() or skill_id
    if not body:
        return None
    return SkillPack(
        id=skill_id,
        name=name,
        description=meta.get("description", "").strip(),
        version=meta.get("version", "").strip(),
        body=body,
    )


def load_skills(directory: Optional[Path] = None) -> Dict[str, SkillPack]:
    """Load every valid skill from the allowlisted directory, keyed by id.

    Missing dir -> empty registry. Files are taken only from the top level of
    the dir; each candidate is containment-checked so a symlink can't point
    outside, and oversize files are skipped."""
    directory = (directory or skills_dir()).resolve()
    if not directory.is_dir():
        return {}
    out: Dict[str, SkillPack] = {}
    for entry in sorted(directory.glob("*.md")):
        try:
            resolved = entry.resolve()
            if resolved.parent != directory:  # symlink escape
                continue
            if not resolved.is_file() or resolved.stat().st_size > _MAX_SKILL_BYTES:
                continue
            skill = parse_skill(resolved.read_text(encoding="utf-8"), entry.stem)
        except OSError:
            continue
        if skill is not None:
            out[skill.id] = skill
    return out


def list_skills(directory: Optional[Path] = None) -> list:
    """Selectable skills as plain dicts for the API/UI (no body)."""
    return [
        {"id": s.id, "name": s.name, "description": s.description, "version": s.version}
        for s in load_skills(directory).values()
    ]


def get_skill_instruction(
    skill_id: Optional[str], directory: Optional[Path] = None
) -> Optional[str]:
    """Resolve a selected skill id to its instruction block, or None if the id
    is empty/unknown. Safe: only ever returns text from an allowlisted file."""
    if not skill_id or not isinstance(skill_id, str):
        return None
    skill = load_skills(directory).get(skill_id)
    return skill.body if skill is not None else None
