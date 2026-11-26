"""Zone-map pruning: which files a query opens and which row groups it must read.

The same function runs over statistics computed from a layout plan (the simulator) and statistics
read from real Parquet footers (the validator), so the two can be compared exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .intervals import IntervalSet
from .schema import DERIVED_PARTITIONS


@dataclass
class RowGroupStats:
    rows: int
    minmax: dict[str, tuple]  # column -> (min, max) in canonical values; (None, None) if unknown


@dataclass
class FileStats:
    partition: dict  # partition column -> canonical value
    row_groups: list[RowGroupStats]


@dataclass
class ZoneMapIndex:
    partition_columns: tuple[str, ...]
    files: list[FileStats]

    @property
    def n_row_groups(self) -> int:
        return sum(len(f.row_groups) for f in self.files)


@dataclass
class QueryCost:
    query_id: str
    frequency: float
    files_opened: int
    row_groups_read: int
    row_groups_total: int
    bytes_read: float
    read: list[tuple[int, int]]  # (file index, row-group index), not serialised

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "frequency": self.frequency,
            "files_opened": self.files_opened,
            "row_groups_read": self.row_groups_read,
            "row_groups_total": self.row_groups_total,
            "bytes_read": self.bytes_read,
        }


def partition_matches(partition: dict, restrictions: dict[str, IntervalSet]) -> bool:
    """True if a file with these partition values may hold a matching row.

    A month partition is a hidden transform of the date column, so a date restriction prunes it
    through the month's first and last day.
    """
    for col, value in partition.items():
        rs = restrictions.get(col)
        if rs is not None and not rs.contains(value):
            return False
        derived = DERIVED_PARTITIONS.get(col)
        if derived is not None:
            source, bounds = derived
            src = restrictions.get(source)
            if src is not None and not src.intersects(*bounds(value)):
                return False
    return True


def row_group_matches(stats: RowGroupStats, restrictions: dict[str, IntervalSet]) -> bool:
    for col, rs in restrictions.items():
        mn, mx = stats.minmax.get(col, (None, None))
        if not rs.intersects(mn, mx):
            return False
    return True


def simulate_query(
    index: ZoneMapIndex,
    restrictions: dict[str, IntervalSet],
    referenced: list[str],
    bytes_per_row: dict[str, float],
    query_id: str = "",
    frequency: float = 0.0,
) -> QueryCost:
    row_bytes = sum(bytes_per_row[c] for c in referenced)
    opened = 0
    read: list[tuple[int, int]] = []
    nbytes = 0.0
    for fi, f in enumerate(index.files):
        if not partition_matches(f.partition, restrictions):
            continue
        opened += 1
        for ri, rg in enumerate(f.row_groups):
            if row_group_matches(rg, restrictions):
                read.append((fi, ri))
                nbytes += rg.rows * row_bytes
    return QueryCost(query_id, frequency, opened, len(read), index.n_row_groups, nbytes, read)


def weighted_cost(costs: list[QueryCost], cfg: Config, n_files: int, small_files: bool) -> float:
    """Sum over queries of frequency x (bytes read + file_open_cost x files opened)."""
    total = sum(
        c.frequency * (c.bytes_read + cfg.file_open_cost_bytes * c.files_opened) for c in costs
    )
    if small_files:
        total += sum(c.frequency for c in costs) * cfg.small_file_list_cost_bytes * n_files
    return total
