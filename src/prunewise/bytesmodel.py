"""Compressed bytes per row for each column, measured from one reference write."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"


def measure_bytes_per_row(table: pa.Table, row_group_size: int) -> dict[str, float]:
    """Write the whole table once (in memory) and divide each column's compressed size by rows."""
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=row_group_size, compression=COMPRESSION)
    meta = pq.ParquetFile(pa.BufferReader(sink.getvalue())).metadata
    totals: dict[str, int] = {}
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            totals[col.path_in_schema] = (
                totals.get(col.path_in_schema, 0) + col.total_compressed_size
            )
    return {name: size / meta.num_rows for name, size in totals.items()}
