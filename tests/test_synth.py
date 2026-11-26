from collections import Counter

from prunewise.synth import generate


def test_generator_is_deterministic():
    a, b = generate(2_000, seed=3), generate(2_000, seed=3)
    assert a.sales.equals(b.sales) and a.products.equals(b.products)
    assert generate(2_000, seed=4).sales.equals(a.sales) is False


def test_dimensions_and_category_blocks(env):
    assert env.stores.num_rows == 60 and env.products.num_rows == 2000
    regions = Counter(env.stores["region"].to_pylist())
    assert set(regions.values()) == {12} and len(regions) == 5
    cats = env.products["category"].to_pylist()
    assert len(set(cats)) == 12
    for name, (lo, hi) in env.truth["category_blocks"].items():
        assert set(cats[lo - 1 : hi]) == {name}  # one contiguous block per category


def test_fact_table_properties(env):
    sales = env.sales
    assert sales.num_rows == 40_000
    dates = sales["sale_date"].to_pylist()
    assert min(dates).isoformat() == "2026-01-01" and max(dates).isoformat() == "2026-03-31"
    assert len(set(dates)) == 90
    region_of = dict(
        zip(env.stores["store_id"].to_pylist(), env.stores["region"].to_pylist(), strict=True)
    )
    assert all(
        region_of[s] == r
        for s, r in zip(
            sales["store_id"].to_pylist()[:2000],
            sales["store_region"].to_pylist()[:2000],
            strict=True,
        )
    )


def test_query_log_has_25_templates_with_the_documented_weights(env):
    queries = env.workload.queries
    assert len(queries) == 25
    total = sum(q.executions_per_day for q in queries)
    by_prefix = {
        "region": sum(q.executions_per_day for q in queries if "store_region =" in q.sql),
        "category": sum(q.executions_per_day for q in queries if "p.category" in q.sql),
        "store": sum(
            q.executions_per_day for q in queries if "store_id = " in q.sql or "store_name" in q.sql
        ),
        "unprunable": sum(q.executions_per_day for q in queries if q.unprunable),
    }
    assert total == 1000
    assert by_prefix == {"region": 550, "category": 200, "store": 100, "unprunable": 50}
