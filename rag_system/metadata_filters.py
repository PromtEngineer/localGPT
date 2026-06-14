"""Typed per-index metadata schemas and validated query filters.

Pattern adapted from the NVIDIA RAG Blueprint custom-metadata design:
- each index may define a typed schema (string/integer/float/boolean fields)
- document metadata is validated strictly at upload (unknown fields and
  type mismatches are rejected, required fields enforced)
- query-time filters are validated against the schema and compiled to a
  LanceDB SQL ``where`` clause over dedicated ``meta_<field>`` columns —
  filter input is NEVER interpolated into SQL without validation/escaping

This module is dependency-free so the backend, the RAG server, and the
indexing worker can all import it cheaply.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ALLOWED_TYPES = {"string", "integer", "float", "boolean"}
# Existing top-level LanceDB columns; schema fields must not collide
RESERVED_NAMES = {
    "vector",
    "text",
    "chunk_id",
    "document_id",
    "chunk_index",
    "metadata",
}
COLUMN_PREFIX = "meta_"

_OPS_BY_TYPE = {
    "string": {"==", "!=", "in"},
    "integer": {"==", "!=", ">", ">=", "<", "<=", "in"},
    "float": {"==", "!=", ">", ">=", "<", "<="},
    "boolean": {"==", "!="},
}


class FilterError(ValueError):
    """Raised when a filter or metadata value does not fit the schema."""


def validate_schema(schema: Any) -> List[str]:
    """Return a list of problems; empty list means the schema is valid.

    Expected shape: [{"name": "project", "type": "string", "required": bool?,
    "description": str?}, ...]
    """
    errors: List[str] = []
    if not isinstance(schema, list) or not schema:
        return ["schema must be a non-empty list of field definitions"]
    seen = set()
    for i, field in enumerate(schema):
        if not isinstance(field, dict):
            errors.append(f"field #{i}: must be an object")
            continue
        name = field.get("name")
        ftype = field.get("type")
        if not isinstance(name, str) or not FIELD_NAME_RE.match(name):
            errors.append(f"field #{i}: name must match {FIELD_NAME_RE.pattern}")
            continue
        if name in RESERVED_NAMES:
            errors.append(f"field '{name}': reserved name")
        if name in seen:
            errors.append(f"field '{name}': duplicate")
        seen.add(name)
        if ftype not in ALLOWED_TYPES:
            errors.append(
                f"field '{name}': type must be one of {sorted(ALLOWED_TYPES)}"
            )
        if "required" in field and not isinstance(field["required"], bool):
            errors.append(f"field '{name}': required must be true or false")
        if "description" in field and not isinstance(field["description"], str):
            errors.append(f"field '{name}': description must be a string")
    return errors


def coerce_value(ftype: str, value: Any, field: str) -> Any:
    """Coerce a raw value (possibly a string from a form/UI) to the schema type."""
    try:
        if ftype == "string":
            if not isinstance(value, str):
                raise FilterError(f"'{field}' expects a string")
            return value
        if ftype == "integer":
            if isinstance(value, bool):
                raise FilterError(f"'{field}' expects an integer")
            return int(value)
        if ftype == "float":
            if isinstance(value, bool):
                raise FilterError(f"'{field}' expects a number")
            return float(value)
        if ftype == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise FilterError(f"'{field}' expects true/false")
    except (TypeError, ValueError):
        raise FilterError(f"'{field}': cannot interpret {value!r} as {ftype}")
    raise FilterError(f"'{field}': unknown type {ftype}")


def validate_document_metadata(
    schema: List[Dict[str, Any]], metadata: Any
) -> Dict[str, Any]:
    """Validate one document's metadata against the schema.

    Strict, NVIDIA-style: unknown fields are rejected (catches typos that
    would otherwise silently make documents unfilterable), required fields
    must be present, values are type-coerced. Returns the cleaned dict with
    EVERY schema field present (None when not provided).
    """
    metadata = metadata or {}
    if not isinstance(metadata, dict):
        raise FilterError("document metadata must be an object")
    by_name = {f["name"]: f for f in schema}
    unknown = set(metadata) - set(by_name)
    if unknown:
        raise FilterError(
            f"unknown metadata field(s): {sorted(unknown)} — schema defines {sorted(by_name)}"
        )
    cleaned: Dict[str, Any] = {}
    for name, field in by_name.items():
        if name in metadata and metadata[name] is not None:
            cleaned[name] = coerce_value(field["type"], metadata[name], name)
        elif field.get("required"):
            raise FilterError(f"required metadata field missing: '{name}'")
        else:
            cleaned[name] = None
    return cleaned


def flatten_columns(
    schema: Optional[List[Dict[str, Any]]], metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Map cleaned metadata to LanceDB column values (every field present,
    None when untagged — Arrow tables need a stable per-row column set)."""
    if not schema:
        return {}
    metadata = metadata or {}
    return {f"{COLUMN_PREFIX}{f['name']}": metadata.get(f["name"]) for f in schema}


def _sql_literal(ftype: str, value: Any) -> str:
    if ftype == "string":
        return "'" + str(value).replace("'", "''") + "'"
    if ftype == "boolean":
        return "true" if value else "false"
    return str(value)


def compile_filters(
    schema: Optional[List[Dict[str, Any]]], filters: Any
) -> Optional[str]:
    """Compile a filters object to a SQL where clause, or None for no filter.

    Accepted per-field forms:
      {"project": "Antapaccay"}                  equality
      {"year": {">=": 2020, "<": 2024}}          typed operators
      {"project": ["Antapaccay", "Lumwana"]}     IN list

    Raises FilterError on unknown fields, bad operators, or type mismatches —
    callers must surface the error, never fall back to executing raw input.
    """
    if not filters:
        return None
    if not isinstance(filters, dict):
        raise FilterError("filters must be an object of field → condition")
    if not schema:
        raise FilterError(
            "this index has no metadata schema — define one before filtering"
        )

    by_name = {f["name"]: f for f in schema}
    clauses: List[str] = []
    for name, condition in filters.items():
        field = by_name.get(name)
        if not field:
            raise FilterError(
                f"unknown filter field '{name}' — schema defines {sorted(by_name)}"
            )
        ftype = field["type"]
        col = f"{COLUMN_PREFIX}{name}"
        allowed_ops = _OPS_BY_TYPE[ftype]

        if isinstance(condition, list):
            if "in" not in allowed_ops:
                raise FilterError(f"'{name}' ({ftype}) does not support list filters")
            values = [coerce_value(ftype, v, name) for v in condition]
            if not values:
                raise FilterError(f"'{name}': empty list filter")
            clauses.append(
                f"{col} IN ({', '.join(_sql_literal(ftype, v) for v in values)})"
            )
        elif isinstance(condition, dict):
            if not condition:
                raise FilterError(f"'{name}': empty condition")
            for op, raw in condition.items():
                if op not in allowed_ops or op == "in":
                    raise FilterError(
                        f"'{name}' ({ftype}) does not support operator '{op}'"
                    )
                value = coerce_value(ftype, raw, name)
                sql_op = "=" if op == "==" else op
                clauses.append(f"{col} {sql_op} {_sql_literal(ftype, value)}")
        else:
            value = coerce_value(ftype, condition, name)
            clauses.append(f"{col} = {_sql_literal(ftype, value)}")

    return " AND ".join(clauses) if clauses else None
