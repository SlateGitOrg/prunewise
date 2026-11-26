"""The simulator on a hand-built file of 8 row groups with known min/max."""

import pytest

from prunewise.config import PRESETS
from prunewise.predicates import extract_query
from prunewise.schema import date_to_days
from prunewise.zonemap import (
    FileStats,
    QueryCost,
    RowGroupStats,
    ZoneMapIndex,
    partition_matches,
    simulate_query,
    weighted_cost,
)

BASE = date_to_days(__import__("datetime").date(2026, 1, 1))

# (date offset range, store range) per row group, 100 rows each
GROUPS = [
    ((0, 9), (1, 20)),
    ((10, 19), (1, 20)),
    ((20, 29), (21, 40)),
    ((30, 39), (21, 40)),
    ((40, 49), (41, 60)),
    ((50, 59), (41, 60)),
    ((60, 69), (1, 60)),
    ((70, 79), (1, 60)),
]
BPR = {"sale_date": 4.0, "store_id": 2.0, "amount_cents": 8.0, "sale_id": 8.0}


@pytest.fixture
def index():
    rgs = [
        RowGroupStats(100, {"sale_date": (BASE + d[0], BASE + d[1]), "store_id": s})
        for d, s in GROUPS
    ]
    return ZoneMapIndex((), [FileStats({}, rgs)])


def run(env, index, where):
    m = extract_query(f"SELECT sum(amount_cents) FROM sales WHERE {where}", env.resolver)
    cost = simulate_query(index, m.restrictions, m.referenced_columns, BPR, "q", 1.0)
    return [ri for _, ri in cost.read], cost


def test_date_range_selects_the_three_overlapping_row_groups(env, index):
    read, cost = run(env, index, "sale_date BETWEEN '2026-01-26' AND '2026-02-14'")
    assert read == [2, 3, 4]
    assert cost.row_groups_total == 8
    # 3 row groups x 100 rows x (sale_date 4 + amount_cents 8) bytes
    assert cost.bytes_read == 3 * 100 * 12


def test_point_predicate_selects_every_group_whose_range_contains_it(env, index):
    assert run(env, index, "store_id = 30")[0] == [2, 3, 6, 7]


def test_in_set_and_range_combine(env, index):
    read, _ = run(env, index, "store_id IN (5, 55) AND sale_date >= '2026-02-20'")
    assert read == [5, 6, 7]


def test_two_columns_intersect_and_unrestricted_reads_everything(env, index):
    assert run(env, index, "sale_date BETWEEN '2026-01-16' AND '2026-01-23' AND store_id >= 21")[
        0
    ] == [2]
    assert run(env, index, "store_id = 99")[0] == []
    assert run(env, index, "store_id = 3 OR sale_date > '2026-01-01'")[0] == list(range(8))


def test_boundary_values_are_inclusive_for_closed_and_exclusive_for_open_bounds(env, index):
    assert run(env, index, "sale_date = '2026-01-10'")[0] == [0]  # offset 9 ends group 0
    assert run(env, index, "sale_date = '2026-01-11'")[0] == [1]
    assert run(env, index, "sale_date > '2026-01-10'")[0] == list(range(1, 8))
    assert run(env, index, "sale_date < '2026-01-01'")[0] == []


def test_partition_pruning_uses_values_and_month_transform(env):
    f = {"store_region": "NORTH", "sale_month": "2026-02"}
    pm = lambda sql: partition_matches(f, extract_query(sql, env.resolver).restrictions)  # noqa: E731
    base = "SELECT 1 FROM sales WHERE "
    assert pm(base + "store_region = 'NORTH'")
    assert not pm(base + "store_region = 'SOUTH'")
    assert pm(base + "sale_date BETWEEN '2026-02-28' AND '2026-03-02'")
    assert not pm(base + "sale_date >= '2026-03-01'")
    assert not pm(base + "sale_month = '2026-01'")


def test_cost_is_bytes_plus_open_cost_and_small_file_listing():
    cfg = PRESETS["tests"]
    costs = [
        QueryCost("a", 10, 2, 3, 8, 500.0, []),
        QueryCost("b", 5, 1, 1, 8, 100.0, []),
        QueryCost("c", 0, 7, 7, 8, 9999.0, []),
    ]
    base = 10 * (500 + cfg.file_open_cost_bytes * 2) + 5 * (100 + cfg.file_open_cost_bytes)
    assert weighted_cost(costs, cfg, 40, False) == base
    assert weighted_cost(costs, cfg, 40, True) == base + 15 * cfg.small_file_list_cost_bytes * 40
