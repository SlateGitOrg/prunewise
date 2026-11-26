import copy

import pyarrow.parquet as pq

from prunewise.footers import read_footer_index
from prunewise.layouts import baseline_layout, enumerate_candidates
from prunewise.materialize import FILE_NAME
from prunewise.ranking import cardinalities, evaluate_layout, simulate_layout
from prunewise.soundness import check_plan, skipped_with_match
from prunewise.validate import compare_costs, compare_indexes, duckdb_results


def all_layouts(env):
    return [baseline_layout(env.cfg), *enumerate_candidates(env.cfg)]


def test_no_skipped_row_group_holds_a_matching_row_for_any_candidate_and_query(env):
    card = cardinalities(env.columns, ["sale_date", "sale_month", "store_region", "store_id"])
    layouts = all_layouts(env)
    assert len(layouts) == 41
    skipped_total = 0
    for layout in layouts:
        ev, plan, index = evaluate_layout(
            layout,
            env.columns,
            env.sales.num_rows,
            card,
            env.workload,
            env.ranking.bytes_per_row,
            env.cfg,
            check_exclusion=False,
        )
        assert check_plan(plan, ev.costs, env.oracles) == 0, layout.layout_id
        skipped_total += sum(c.row_groups_total - c.row_groups_read for c in ev.costs)
    assert skipped_total > 10_000  # the check is not vacuous


def test_soundness_check_catches_a_row_group_wrongly_skipped(env):
    card = cardinalities(env.columns, ["sale_date", "sale_month", "store_region", "store_id"])
    layout = env.ranking.ranked[0].layout
    ev, plan, _ = evaluate_layout(
        layout,
        env.columns,
        env.sales.num_rows,
        card,
        env.workload,
        env.ranking.bytes_per_row,
        env.cfg,
    )
    qi = next(i for i, o in enumerate(env.oracles) if o.truth_mask.sum() > 0 and ev.costs[i].read)
    forged = copy.copy(ev.costs[qi])
    forged.read = []  # pretend every row group was skipped
    assert skipped_with_match(plan, forged, env.oracles[qi].any_mask) > 0


def test_extractor_never_excludes_a_row_the_real_query_matches(env):
    assert sum(o.extractor_violations for o in env.oracles) == 0
    for o in env.oracles:
        assert o.truth_mask.sum() <= o.model_mask.sum()
    assert sum(o.truth_mask.sum() > 0 for o in env.oracles) >= 20


def test_materialised_files_follow_the_plan(env):
    for layout in env.layouts:
        root = env.lake / layout.layout_id
        files = sorted(root.rglob(FILE_NAME))
        for path in files:
            groups = [
                pq.ParquetFile(path).metadata.row_group(i).num_rows
                for i in range(pq.ParquetFile(path).metadata.num_row_groups)
            ]
            assert all(g == layout.row_group_size for g in groups[:-1])
            assert groups[-1] <= layout.row_group_size
        assert sum(pq.ParquetFile(p).metadata.num_rows for p in files) == 40_000
    top = env.ranking.ranked[0].layout
    assert any(
        p.parent.name.startswith("store_region=")
        for p in (env.lake / top.layout_id).rglob(FILE_NAME)
    )


def test_simulated_row_groups_equal_the_real_footers_exactly(env):
    zone_cols = env.workload.zone_columns
    for layout in env.layouts:
        ev, _, index = evaluate_layout(
            layout,
            env.columns,
            env.sales.num_rows,
            env.card,
            env.workload,
            env.ranking.bytes_per_row,
            env.cfg,
            check_exclusion=False,
        )
        real, _ = read_footer_index(env.lake / layout.layout_id, layout, zone_cols)
        assert compare_indexes(index, real) == []
        real_costs = simulate_layout(real, env.workload, env.ranking.bytes_per_row)
        assert compare_costs(ev.costs, real_costs) == []
        for sim, foot in zip(ev.costs, real_costs, strict=True):
            assert sorted(sim.read) == sorted(foot.read)
            assert sim.row_groups_read == foot.row_groups_read


def test_a_tampered_footer_index_is_reported(env):
    layout = env.layouts[1]
    zone_cols = env.workload.zone_columns
    ev, _, index = evaluate_layout(
        layout,
        env.columns,
        env.sales.num_rows,
        env.card,
        env.workload,
        env.ranking.bytes_per_row,
        env.cfg,
        check_exclusion=False,
    )
    real, _ = read_footer_index(env.lake / layout.layout_id, layout, zone_cols)
    real.files[0].row_groups[0].minmax["sale_date"] = (0, 1)
    assert compare_indexes(index, real)
    real_costs = simulate_layout(real, env.workload, env.ranking.bytes_per_row)
    assert compare_costs(ev.costs, real_costs)


def test_duckdb_results_are_identical_across_baseline_and_three_layouts(env):
    assert len(env.layouts) == 4
    results = []
    for layout in env.layouts:
        paths = sorted((env.lake / layout.layout_id).rglob(FILE_NAME))
        results.append(duckdb_results(paths, env.products, env.stores, env.workload))
    assert all(len(r) == 25 for r in results)
    for r in results[1:]:
        assert r == results[0]
    assert sum(len(rows) for rows in results[0].values()) > 100  # results are not trivially empty


def test_validation_report_is_clean(env):
    s = env.validation["summary"]
    assert s["soundness_violations"] == 0
    assert s["simulator_footer_mismatches"] == 0
    assert s["duckdb_result_mismatches"] == 0
    assert s["extractor_violations"] == 0
    assert s["row_groups_rescanned"] > 1_000
    assert len(env.validation["layouts"]) == 4
