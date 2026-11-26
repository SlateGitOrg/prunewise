"""Read zone-map statistics from the footers of materialised Parquet files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from .layouts import Layout
from .materialize import FILE_NAME
from .schema import FACT_COLUMNS, date_to_days
from .zonemap import FileStats, RowGroupStats, ZoneMapIndex


def _canonical_stat(kind: str, value):
    if value is None:
        return None
    if kind == "date":
        return date_to_days(value) if isinstance(value, date) else int(value)
    return value


def _parse_partition_value(kind: str, text: str):
    if kind == "int":
        return int(text)
    if kind == "date":
        return date_to_days(date.fromisoformat(text))
    return text


def read_footer_index(
    layout_dir: Path, layout: Layout, zone_cols: list[str]
) -> tuple[ZoneMapIndex, list[Path]]:
    """Zone-map index from real footers, plus the file path of each indexed file."""
    entries = []
    for path in layout_dir.rglob(FILE_NAME):
        rel = path.relative_to(layout_dir).parts[:-1]
        names = [p.split("=", 1)[0] for p in rel]
        if tuple(names) != layout.partition:
            raise ValueError(
                f"{path}: directories {names} do not match partition {layout.partition}"
            )
        values = tuple(
            _parse_partition_value(FACT_COLUMNS[c].kind, p.split("=", 1)[1])
            for c, p in zip(layout.partition, rel, strict=True)
        )
        entries.append((values, path))
    entries.sort(key=lambda e: e[0])
    files, paths = [], []
    for values, path in entries:
        meta = pq.ParquetFile(path).metadata
        col_pos = {meta.schema.column(i).path: i for i in range(meta.num_columns)}
        groups = []
        for g in range(meta.num_row_groups):
            rg = meta.row_group(g)
            minmax = {}
            for col in zone_cols:
                stats = rg.column(col_pos[col]).statistics
                if stats is not None and stats.has_min_max:
                    kind = FACT_COLUMNS[col].kind
                    minmax[col] = (
                        _canonical_stat(kind, stats.min),
                        _canonical_stat(kind, stats.max),
                    )
                else:
                    minmax[col] = (None, None)
            groups.append(RowGroupStats(rg.num_rows, minmax))
        files.append(FileStats(dict(zip(layout.partition, values, strict=True)), groups))
        paths.append(path)
    return ZoneMapIndex(layout.partition, files), paths
