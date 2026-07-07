"""
vektor.filtering.parser
------------------------
Translates user-facing filter dicts into parameterised SQL WHERE clauses.

Output is always a tuple: (where_clause_string, bound_parameters_list).
The where_clause_string uses only ? placeholders — no user values embedded.

Supported operators:
    Equality:          {"field": "value"} or {"field": {"eq": "value"}}
    Greater than:      {"field": {"gt": value}}
    Greater or equal:  {"field": {"gte": value}}
    Less than:         {"field": {"lt": value}}
    Less or equal:     {"field": {"lte": value}}
    Set membership:    {"field": {"in": [v1, v2, v3]}}
    Range:             {"field": {"gte": 2020, "lte": 2024}}

All metadata fields are stored as JSON in the vectors.metadata column.
SQLite's json_extract() is used for all field access.
[Certain] Direct column references won't work — metadata is a JSON string.
"""

from __future__ import annotations

from typing import Any


class FilterParseError(Exception):
    """Raised when a filter dict cannot be translated to valid SQL."""


OPERATOR_MAP = {
    "eq":  "=",
    "gt":  ">",
    "gte": ">=",
    "lt":  "<",
    "lte": "<=",
}


def _json_field(field_name: str) -> str:
    """
    Build a json_extract expression for a metadata field.

    SQLite: json_extract(metadata, '$.field_name')
    Field names are not user-injectable here because they appear inside
    a string literal that json_extract interprets, not as SQL syntax.
    However, we still validate field names to prevent path traversal.
    """
    if not field_name.replace("_", "").replace("-", "").isalnum():
        raise FilterParseError(
            f"Filter field name '{field_name}' contains invalid characters. "
            f"Only alphanumeric, underscore, and hyphen are allowed."
        )
    return f"json_extract(metadata, '$.{field_name}')"


def parse_filter(filter_dict: dict) -> tuple[str, list]:
    """
    Translate a filter dict into a SQL WHERE clause and parameter list.

    Args:
        filter_dict: User-supplied filter specification.

    Returns:
        (where_clause, params) where where_clause uses ? placeholders
        and params is the ordered list of values to bind.

    Raises:
        FilterParseError: Invalid filter structure or unsupported operator.

    Examples:
        parse_filter({"year": 2024})
        → ("json_extract(metadata, '$.year') = ?", [2024])

        parse_filter({"year": {"gte": 2020, "lte": 2024}})
        → ("json_extract(metadata, '$.year') >= ? AND ... <= ?", [2020, 2024])

        parse_filter({"tag": {"in": ["ml", "rag"]}})
        → ("json_extract(metadata, '$.tag') IN (?,?)", ["ml", "rag"])
    """
    if not filter_dict:
        return "", []

    clauses = []
    params = []

    for field, condition in filter_dict.items():
        json_field = _json_field(field)

        if not isinstance(condition, dict):
            # Shorthand equality: {"year": 2024}
            clauses.append(f"{json_field} = ?")
            params.append(condition)
            continue

        # Operator dict
        for op, value in condition.items():
            if op == "in":
                if not isinstance(value, (list, tuple)) or len(value) == 0:
                    raise FilterParseError(
                        f"Filter operator 'in' requires a non-empty list. "
                        f"Got: {value!r}"
                    )
                placeholders = ",".join("?" * len(value))
                clauses.append(f"{json_field} IN ({placeholders})")
                params.extend(value)

            elif op in OPERATOR_MAP:
                sql_op = OPERATOR_MAP[op]
                clauses.append(f"{json_field} {sql_op} ?")
                params.append(value)

            else:
                raise FilterParseError(
                    f"Unsupported filter operator '{op}'. "
                    f"Supported: {sorted(list(OPERATOR_MAP.keys()) + ['in'])}"
                )

    where_clause = " AND ".join(clauses)
    return where_clause, params


def get_eligible_slot_ids(
    conn,
    collection_name: str,
    filter_dict: dict,
) -> frozenset[int]:
    """
    Query SQLite for live slot IDs satisfying a filter.

    Combines the filter WHERE clause with `deleted = 0` automatically.

    Args:
        conn:            Open SQLite connection.
        collection_name: Collection to query.
        filter_dict:     User filter specification.

    Returns:
        frozenset of integer slot IDs that are live and match the filter.
    """
    where_clause, params = parse_filter(filter_dict)

    if where_clause:
        sql = (
            f"SELECT slot_id FROM vectors "
            f"WHERE collection = ? AND deleted = 0 AND {where_clause}"
        )
        rows = conn.execute(sql, [collection_name] + params).fetchall()
    else:
        sql = "SELECT slot_id FROM vectors WHERE collection = ? AND deleted = 0"
        rows = conn.execute(sql, [collection_name]).fetchall()

    return frozenset(row[0] for row in rows)