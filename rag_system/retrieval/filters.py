"""A deterministic metadata filter DSL that compiles to LanceDB SQL (roadmap 4.4).

The surface the roadmap asks for is ``field=value`` / ``field in (a, b)``. Here it
is JSON rather than a filter *string*, for one reason: a string has to be parsed,
and a parser is the thing an injection attack aims at. A JSON object is already
parsed by the time it reaches us, so the only work left is **validation**, and
validation is the whole security story of this module.

    {"document_id": "07_nda.pdf"}
    {"document_id": {"in": ["07_nda.pdf", "03_ip_certification.pdf"]}}
    {"document_name": {"contains": "nda"}, "chunk_index": {"gte": 0, "lte": 4}}

Top-level keys are ANDed. There is no OR, no NOT and no nesting — not because
they are hard, but because nothing has asked for them and every operator added
here is another string that ends up inside a SQL predicate.

No LLM is involved. The same filter object always compiles to the same
where-clause, byte for byte (fields are emitted in a fixed canonical order, not
in dict order), so a filtered retrieval is as reproducible as an unfiltered one.

Rules this module holds to
--------------------------
* **Refuse, don't escape.** Following ``rag_system/retrieval/document_fetch.py``:
  a value carrying a quote, a backslash or a control character is rejected with
  a ``FilterError``, never repaired. The one exception is the ``LIKE`` wildcards
  ``%`` and ``_``, which are escaped (with an explicit ``ESCAPE '\\'`` clause) so
  that ``contains`` means *substring*, literally, for filenames like
  ``01_acquisition_agreement.pdf``.
* **Fail loud.** Unknown field, unknown operator, wrong value type, empty filter
  — all raise ``FilterError``. The API turns that into a 400. A filter that
  cannot be honoured must never degrade into an unfiltered search: that would
  hand the caller results they explicitly excluded.
* **Only real columns.** The LanceDB text table has exactly six columns
  (``vector``, ``text``, ``chunk_id``, ``document_id``, ``chunk_index``,
  ``metadata``) and ``metadata`` is a JSON *string*. So page numbers, dates and
  every other per-chunk field are **not** filterable here; see the decision file
  for the schema change that would fix it. Filtering on a JSON substring would
  look like it worked and quietly be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# A LanceDB int32 column; a value outside this cannot match anything and is far
# more likely to be a client bug than an intention.
_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)

# Long enough for any real document id (uuid + filename), short enough that a
# filter cannot be used to push a megabyte of text into a query plan.
_MAX_STRING_LENGTH = 256

# An IN-list this long is a client that means "no filter".
_MAX_IN_ITEMS = 256

# Characters that could terminate or extend the SQL literal we are building.
# Rejected, not escaped — see the module docstring.
_FORBIDDEN_CHARS = ("'", '"', "\\", ";", "`", "\n", "\r", "\t", "\x00")


class FilterError(ValueError):
    """A filter that cannot be compiled. Callers must surface this, not swallow it."""


@dataclass(frozen=True)
class CompiledFilter:
    """A validated filter plus the SQL it compiles to.

    ``where`` is the only thing that reaches LanceDB. ``spec`` is the caller's
    original object, kept for logging, SSE payloads and the semantic-cache key —
    two queries with different filters are different queries.
    """

    where: str
    spec: Dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.where

    @property
    def signature(self) -> str:
        """A stable identity for this filter, for cache keys."""
        return self.where


# --------------------------------------------------------------------------
# field table
# --------------------------------------------------------------------------
#
# (column, kind, allowed operators). Order matters: it is the canonical emission
# order, so the compiled where-clause does not depend on JSON key order.

_STRING_OPS = ("eq", "in", "contains")
_INT_OPS = ("eq", "in", "gt", "gte", "lt", "lte")

_FIELDS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    # name             column          kind      operators
    ("document_id",    "document_id",  "string", _STRING_OPS),
    # `document_name` is not a column. Document ids are the file's basename for
    # anything indexed by the CLI and "<uuid>_<basename>" for anything uploaded
    # through the UI, so matching a *name* means substring-matching the id.
    # Only `contains` is offered, because `eq` would silently miss every
    # UI-uploaded document and that is a trap, not a feature.
    ("document_name",  "document_id",  "string", ("contains",)),
    ("chunk_id",       "chunk_id",     "string", ("eq", "in")),
    ("chunk_index",    "chunk_index",  "int",    _INT_OPS),
)

_FIELD_BY_NAME = {name: (column, kind, ops) for name, column, kind, ops in _FIELDS}

_COMPARISON_SQL = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def _describe_support() -> str:
    return "; ".join(f"{name} ({', '.join(ops)})" for name, _c, _k, ops in _FIELDS)


# --------------------------------------------------------------------------
# value validation
# --------------------------------------------------------------------------

def _check_string(field: str, operator: str, value: Any) -> str:
    if not isinstance(value, str):
        raise FilterError(
            f"filters.{field}.{operator} must be a string, got {type(value).__name__}."
        )
    if not value.strip():
        raise FilterError(f"filters.{field}.{operator} must not be empty.")
    if len(value) > _MAX_STRING_LENGTH:
        raise FilterError(
            f"filters.{field}.{operator} is longer than {_MAX_STRING_LENGTH} characters."
        )
    for char in _FORBIDDEN_CHARS:
        if char in value:
            raise FilterError(
                f"filters.{field}.{operator} contains a forbidden character "
                f"({char!r}); quoting characters are refused, not escaped."
            )
    # Anything else non-printable would be invisible in a log line.
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        raise FilterError(
            f"filters.{field}.{operator} contains a control character."
        )
    return value


def _check_int(field: str, operator: str, value: Any) -> int:
    # bool is an int subclass in Python; True as a chunk_index is a client bug.
    if isinstance(value, bool) or not isinstance(value, int):
        raise FilterError(
            f"filters.{field}.{operator} must be an integer, got {type(value).__name__}."
        )
    if not (_INT32_MIN <= value <= _INT32_MAX):
        raise FilterError(f"filters.{field}.{operator} is out of the int32 range.")
    return value


def _quote(value: str) -> str:
    """Wrap an already-validated string in SQL single quotes.

    Safe only because ``_check_string`` has refused every quoting character; this
    function deliberately does no escaping of its own, so that the validation
    above stays the single place where trust is granted.
    """
    return "'" + value + "'"


def _like_pattern(value: str) -> str:
    """``%value%`` with the LIKE wildcards inside *value* escaped."""
    escaped = value.replace("%", "\\%").replace("_", "\\_")
    return "'%" + escaped + "%'"


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------

def _compile_field(field: str, column: str, kind: str,
                   allowed: Tuple[str, ...], constraint: Any) -> List[str]:
    """The SQL predicates for one field. ANDed together by the caller."""
    # A bare scalar is shorthand for equality — the roadmap's `field=value`.
    if not isinstance(constraint, dict):
        constraint = {"eq": constraint}
    if not constraint:
        raise FilterError(f"filters.{field} has no operators.")

    predicates: List[str] = []
    for operator in sorted(constraint):  # canonical order, not dict order
        if operator not in allowed:
            raise FilterError(
                f"filters.{field} does not support the '{operator}' operator. "
                f"Supported: {', '.join(allowed)}."
            )
        value = constraint[operator]

        if operator == "in":
            if not isinstance(value, list):
                raise FilterError(f"filters.{field}.in must be a list.")
            if not value:
                raise FilterError(
                    f"filters.{field}.in is an empty list; an IN-list that "
                    "matches nothing is refused rather than silently dropped."
                )
            if len(value) > _MAX_IN_ITEMS:
                raise FilterError(
                    f"filters.{field}.in has {len(value)} items (max {_MAX_IN_ITEMS})."
                )
            if kind == "string":
                items = [_quote(_check_string(field, "in", v)) for v in value]
            else:
                items = [str(_check_int(field, "in", v)) for v in value]
            predicates.append(f"{column} IN ({', '.join(items)})")

        elif operator == "eq":
            if kind == "string":
                predicates.append(f"{column} = {_quote(_check_string(field, 'eq', value))}")
            else:
                predicates.append(f"{column} = {_check_int(field, 'eq', value)}")

        elif operator == "contains":
            pattern = _like_pattern(_check_string(field, "contains", value))
            # The ESCAPE clause is what makes '_' in "01_acquisition.pdf" a
            # literal underscore instead of "any single character".
            predicates.append(f"{column} LIKE {pattern} ESCAPE '\\'")

        else:  # gt / gte / lt / lte
            number = _check_int(field, operator, value)
            predicates.append(f"{column} {_COMPARISON_SQL[operator]} {number}")

    return predicates


def compile_filters(filters: Any) -> Optional[CompiledFilter]:
    """Validate and compile a filter object into a LanceDB where-clause.

    Returns ``None`` for ``None`` (the no-filter path, which must stay
    byte-identical to having no filter support at all) and a ``CompiledFilter``
    otherwise. Raises ``FilterError`` on anything it cannot compile — including
    an empty object, which is far more likely to be a bug in the caller than a
    request for "no filtering".

    An already-compiled filter passes straight through, so plumbing can call
    this at every layer without recompiling.
    """
    if filters is None:
        return None
    if isinstance(filters, CompiledFilter):
        return filters
    if not isinstance(filters, dict):
        raise FilterError(
            f"filters must be a JSON object, got {type(filters).__name__}. "
            f"Supported fields: {_describe_support()}."
        )
    if not filters:
        raise FilterError(
            "filters is empty. Omit the field entirely to search without a filter; "
            "an empty filter object is refused so that a client bug cannot look "
            "like an unfiltered search."
        )

    unknown = [k for k in filters if k not in _FIELD_BY_NAME]
    if unknown:
        raise FilterError(
            f"Unsupported filter field(s): {', '.join(map(str, sorted(map(str, unknown))))}. "
            f"Supported: {_describe_support()}."
        )

    predicates: List[str] = []
    for name, column, kind, allowed in _FIELDS:  # canonical field order
        if name not in filters:
            continue
        predicates.extend(_compile_field(name, column, kind, allowed, filters[name]))

    if not predicates:
        raise FilterError("filters produced no predicates.")

    return CompiledFilter(where=" AND ".join(predicates), spec=dict(filters))


def combine(where: Optional[str], compiled: Optional[CompiledFilter]) -> Optional[str]:
    """AND an internally-generated where-clause with the caller's filter.

    Used where the pipeline already restricts a search on its own (the
    cross-reference hop, the overview prefilter's restrict mode): the user's
    filter must still apply, and it must apply as a *narrowing*, never as a
    replacement.
    """
    user = compiled.where if compiled is not None else None
    parts = [p for p in (where, user) if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({p})" for p in parts)
