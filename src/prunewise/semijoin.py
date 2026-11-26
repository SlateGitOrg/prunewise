"""Resolve a filter on a small dimension table into the set of matching keys."""

from __future__ import annotations

import duckdb
import pyarrow as pa

from .schema import DIMENSIONS


class DimensionResolver:
    """Evaluates ``WHERE <condition>`` on a dimension table in DuckDB and caches the key list."""

    def __init__(self, products: pa.Table, stores: pa.Table) -> None:
        self._con = duckdb.connect()
        self._con.register("products", products)
        self._con.register("stores", stores)
        self._cache: dict[tuple[str, str], list[int]] = {}

    def keys(self, dimension: str, condition_sql: str) -> list[int]:
        cache_key = (dimension, condition_sql)
        if cache_key not in self._cache:
            key_col = DIMENSIONS[dimension][0]
            rows = self._con.execute(
                f"SELECT {key_col} FROM {dimension} WHERE {condition_sql} ORDER BY {key_col}"
            ).fetchall()
            self._cache[cache_key] = [int(r[0]) for r in rows]
        return self._cache[cache_key]

    def close(self) -> None:
        self._con.close()
