"""Table schemas, column kinds and conversion of literals to canonical comparable values.

Canonical values: ``int`` columns use Python ints, ``date`` columns use days since the Unix epoch
(also ints), ``str`` columns use strings and ``float`` columns use floats. Integer-like kinds are
discrete, which lets interval sets merge adjacent values and turn ``> 5`` into ``>= 6``.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta

import pyarrow as pa

FACT = "sales"
FACT_KEY = "sale_id"
EPOCH = date(1970, 1, 1)


@dataclass(frozen=True)
class ColumnSpec:
    kind: str  # "int" | "date" | "str" | "float" | "timestamp"
    zonemap: bool  # whether a min/max zone map is kept and predicates on it can prune


FACT_COLUMNS: dict[str, ColumnSpec] = {
    "sale_id": ColumnSpec("int", True),
    "sale_date": ColumnSpec("date", True),
    "sale_month": ColumnSpec("str", True),
    "sale_ts": ColumnSpec("timestamp", False),
    "store_id": ColumnSpec("int", True),
    "store_region": ColumnSpec("str", True),
    "product_id": ColumnSpec("int", True),
    "quantity": ColumnSpec("int", True),
    "amount_cents": ColumnSpec("int", True),
    "promo_code": ColumnSpec("str", False),
}

# dimension table -> (key column, columns)
DIMENSIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "products": (
        "product_id",
        ("product_id", "sku", "product_name", "category", "unit_price_cents"),
    ),
    "stores": ("store_id", ("store_id", "store_name", "region", "opened_date")),
}

FACT_ARROW_SCHEMA = pa.schema(
    [
        ("sale_id", pa.int64()),
        ("sale_date", pa.date32()),
        ("sale_month", pa.string()),
        ("sale_ts", pa.timestamp("us")),
        ("store_id", pa.int32()),
        ("store_region", pa.string()),
        ("product_id", pa.int32()),
        ("quantity", pa.int32()),
        ("amount_cents", pa.int64()),
        ("promo_code", pa.string()),
    ]
)


def date_to_days(d: date) -> int:
    return (d - EPOCH).days


def days_to_date(days: int) -> date:
    return EPOCH + timedelta(days=days)


def month_bounds(month: str) -> tuple[int, int]:
    """Inclusive (first, last) day of a ``YYYY-MM`` month as epoch days."""
    year, mon = int(month[:4]), int(month[5:7])
    last = calendar.monthrange(year, mon)[1]
    return date_to_days(date(year, mon, 1)), date_to_days(date(year, mon, last))


# Partition columns that are a monotone transform of another column (hidden partitioning):
# the partition column name -> (source column, function giving the source's inclusive bounds).
DERIVED_PARTITIONS = {"sale_month": ("sale_date", month_bounds)}


class LiteralMismatch(ValueError):
    """A literal cannot be compared with the column's type."""


def canonical(kind: str, raw: tuple[str, object]):
    """Convert a parsed literal ``(tag, value)`` to the canonical value for a column kind."""
    tag, value = raw
    if kind == "int":
        if tag == "num" and float(value) == int(value):  # type: ignore[arg-type]
            return int(value)  # type: ignore[call-overload]
    elif kind == "float":
        if tag == "num":
            return float(value)  # type: ignore[arg-type]
    elif kind == "str":
        if tag == "str":
            return str(value)
    elif kind == "date":
        if tag == "date":
            return date_to_days(value)  # type: ignore[arg-type]
        if tag == "str":
            try:
                return date_to_days(date.fromisoformat(str(value)))
            except ValueError:
                pass
    raise LiteralMismatch(f"literal {value!r} does not fit a {kind} column")


def to_json_value(kind: str, value):
    if value is None:
        return None
    return days_to_date(value).isoformat() if kind == "date" else value


def from_json_value(kind: str, value):
    if value is None:
        return None
    return date_to_days(date.fromisoformat(value)) if kind == "date" else value


def kind_of_arrow(t: pa.DataType) -> str:
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_date(t):
        return "date"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "str"
    if pa.types.is_floating(t):
        return "float"
    return "timestamp"
