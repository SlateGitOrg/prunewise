from dataclasses import replace

import pytest

from prunewise.layouts import baseline_layout, enumerate_candidates
from prunewise.ranking import rank_layouts
from prunewise.workload import Workload, WorkloadQuery


def test_forty_candidates_are_enumerated_and_the_baseline_is_daily():
    cfg_candidates = enumerate_candidates(
        __import__("prunewise.config", fromlist=["x"]).PRESETS["tests"]
    )
    assert len(cfg_candidates) == 40
    assert len({c.layout_id for c in cfg_candidates}) == 40
    base = baseline_layout(__import__("prunewise.config", fromlist=["x"]).PRESETS["tests"])
    assert base.partition == ("sale_date",) and base.sort is None


def test_top_ranked_layout_partitions_on_region_and_beats_the_baseline(env):
    ranking = env.ranking
    top = ranking.ranked[0]
    assert "store_region" in top.layout.partition
    assert top.cost < ranking.baseline.cost
    assert top.ratio_to_baseline == pytest.approx(top.cost / ranking.baseline.cost)
    costs = [c.cost for c in ranking.ranked]
    assert costs == sorted(costs)


def test_store_by_day_partition_is_flagged_by_the_small_file_rule(env):
    store_day = [
        c for c in env.ranking.candidates if c.layout.partition == ("store_id", "sale_date")
    ]
    assert len(store_day) == 8
    for c in store_day:
        assert c.excluded and c.files == 5400
        assert "small-file rule" in c.exclusion_reason and "5400" in c.exclusion_reason
    assert all(c.excluded for c in env.ranking.candidates[-8:])  # excluded layouts rank last
    assert not any(c.excluded for c in env.ranking.ranked)


def test_median_file_below_minimum_gets_a_listing_penalty(env):
    strict = replace(env.cfg, min_file_rows=100_000)
    ranking = rank_layouts(env.sales, env.workload, strict, env.ranking.bytes_per_row)
    assert all(any("median file" in f for f in c.flags) for c in ranking.ranked)
    plain = {c.layout.layout_id: c for c in env.ranking.ranked}
    for c in ranking.ranked:
        assert c.cost > plain[c.layout.layout_id].cost
        assert c.bytes_per_day == plain[c.layout.layout_id].bytes_per_day


def test_doubling_every_frequency_doubles_every_cost_and_keeps_the_ranking(env):
    doubled = rank_layouts(env.sales, env.workload.scaled(2), env.cfg, env.ranking.bytes_per_row)
    assert [c.layout.layout_id for c in doubled.candidates] == [
        c.layout.layout_id for c in env.ranking.candidates
    ]
    for a, b in zip(env.ranking.candidates, doubled.candidates, strict=True):
        assert b.cost == 2 * a.cost
        assert b.bytes_per_day == 2 * a.bytes_per_day
    assert doubled.baseline.cost == 2 * env.ranking.baseline.cost


def test_a_zero_frequency_query_changes_nothing(env):
    base_q = env.workload.queries[0]
    from prunewise.predicates import extract_query

    sql = "SELECT sum(amount_cents) FROM sales WHERE product_id < 50 AND store_id = 5"
    extra = WorkloadQuery("zero", sql, 0.0, extract_query(sql, env.resolver))
    assert extra.restrictions and base_q.executions_per_day > 0
    ranking = rank_layouts(
        env.sales, env.workload.with_query(extra), env.cfg, env.ranking.bytes_per_row
    )
    assert [(c.layout.layout_id, c.cost) for c in ranking.candidates] == [
        (c.layout.layout_id, c.cost) for c in env.ranking.candidates
    ]


def test_per_query_costs_are_consistent_with_layout_totals(env):
    top = env.ranking.ranked[0]
    assert len(top.costs) == 25
    assert sum(c.frequency * c.bytes_read for c in top.costs) == top.bytes_per_day
    assert all(c.row_groups_read <= c.row_groups_total == top.row_groups for c in top.costs)


def test_workload_and_ranking_survive_a_json_round_trip(env, tmp_path):
    env.workload.save(tmp_path / "w.json")
    loaded = Workload.load(tmp_path / "w.json")
    for a, b in zip(env.workload.queries, loaded.queries, strict=True):
        assert a.restrictions == b.restrictions and a.unprunable == b.unprunable
        assert a.referenced_columns == b.referenced_columns
    from prunewise.ranking import Ranking

    again = Ranking.from_dict(env.ranking.to_dict())
    assert again.ranked[0].layout == env.ranking.ranked[0].layout
    assert again.baseline.cost == env.ranking.baseline.cost


def test_unprunable_templates_are_reported_with_reasons(env):
    found = {(u["query_id"], u["reason"]) for u in env.workload.unprunable_list()}
    assert found == {
        ("q24", "leading wildcard LIKE"),
        ("q25", "function on column"),
    }
    assert env.truth["unprunable_queries"] == {
        "q24": "leading wildcard LIKE",
        "q25": "function on column",
    }
