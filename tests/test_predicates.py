from datetime import date

import pytest

from prunewise.predicates import extract_query
from prunewise.schema import date_to_days, days_to_date


def q(env, where, join=""):
    return extract_query(f"SELECT count(*) FROM sales f {join} WHERE {where}", env.resolver)


def test_range_and_in_set_are_intersected_per_column(env):
    m = q(env, "sale_date BETWEEN '2026-03-01' AND '2026-03-07' AND store_id IN (3, 5)")
    ((lo, lo_inc, hi, hi_inc),) = m.restrictions["sale_date"].intervals
    assert (days_to_date(lo), days_to_date(hi), lo_inc, hi_inc) == (
        date(2026, 3, 1),
        date(2026, 3, 7),
        True,
        True,
    )
    assert m.restrictions["store_id"].values() == [3, 5]
    assert m.unprunable == {}


def test_conjunction_on_one_column_intersects(env):
    m = q(env, "store_id >= 10 AND store_id < 20 AND store_id <> 15 OR store_id = 1")
    # the OR makes the whole predicate a union: [10,19] minus nothing used, plus {1}
    assert m.restrictions["store_id"].values() == [1, *range(10, 20)]
    m = q(env, "store_id > 10 AND store_id <= 12")
    assert m.restrictions["store_id"].values() == [11, 12]


def test_disjunction_on_one_column_is_a_union(env):
    m = q(env, "store_id = 3 OR store_id = 9")
    assert m.restrictions["store_id"].values() == [3, 9]
    assert m.unprunable == {}


def test_cross_column_or_is_unprunable(env):
    m = q(env, "store_id = 3 OR sale_date > '2026-03-01'")
    assert m.restrictions == {}
    assert m.unprunable == {
        "store_id": "disjunction across columns",
        "sale_date": "disjunction across columns",
    }


def test_or_keeps_a_column_every_branch_restricts(env):
    m = q(env, "(store_id = 3 AND sale_date >= '2026-03-01') OR (store_id = 9 AND product_id = 4)")
    assert m.restrictions["store_id"].values() == [3, 9]
    assert "sale_date" not in m.restrictions and "product_id" not in m.restrictions
    assert set(m.unprunable) == {"sale_date", "product_id"}


def test_not_function_and_not_equals_are_unprunable_with_reasons(env):
    assert q(env, "NOT store_id = 3").unprunable == {"store_id": "negation (NOT)"}
    assert q(env, "upper(store_region) = 'NORTH'").unprunable == {
        "store_region": "function on column"
    }
    assert q(env, "store_id != 3").unprunable == {"store_id": "not-equals (!=)"}
    assert q(env, "NOT (store_id = 3 AND sale_date > '2026-03-01')").unprunable.keys() == {
        "store_id",
        "sale_date",
    }
    assert q(env, "store_id = 3 AND NOT store_region = 'EAST'").restrictions[
        "store_id"
    ].values() == [3]


def test_like_wildcards(env):
    assert q(env, "promo_code LIKE '%BOGO%'").unprunable == {"promo_code": "leading wildcard LIKE"}
    assert q(env, "store_region LIKE 'NOR%'").restrictions["store_region"].contains("NORTH")
    assert not q(env, "store_region LIKE 'NOR%'").restrictions["store_region"].contains("SOUTH")
    assert q(env, "store_region LIKE 'N_RTH'").unprunable == {"store_region": "pattern LIKE"}
    assert q(env, "store_region = 'EAST'").unprunable == {}


def test_extract_on_timestamp_is_a_function_on_column(env):
    m = q(env, "extract(hour FROM f.sale_ts) = 18")
    assert m.unprunable == {"sale_ts": "function on column"}


def test_literal_on_the_left_is_flipped_and_dates_parse(env):
    m = q(env, "DATE '2026-03-25' <= f.sale_date AND 50 > f.store_id")
    assert m.restrictions["sale_date"].intervals == (
        (date_to_days(date(2026, 3, 25)), True, None, True),
    )
    assert m.restrictions["store_id"].intervals == ((None, True, 49, True),)


def test_type_mismatch_and_non_zonemapped_columns_are_unprunable(env):
    assert q(env, "store_id = 'abc'").unprunable == {"store_id": "literal type mismatch"}
    assert q(env, "sale_ts >= '2026-03-01'").unprunable == {"sale_ts": "column not zone-mapped"}
    assert q(env, "store_id = store_region").unprunable.keys() == {"store_id", "store_region"}


def test_contradictory_predicate_is_the_empty_set(env):
    assert q(env, "store_id = 3 AND store_id = 9").restrictions["store_id"].is_empty


def test_dimension_filter_resolves_to_the_generators_dairy_block(env):
    m = q(
        env,
        "p.category = 'Dairy'",
        join="JOIN products p USING (product_id)",
    )
    lo, hi = env.truth["category_blocks"]["Dairy"]
    assert m.restrictions["product_id"].intervals == ((lo, True, hi, True),)
    assert m.restrictions["product_id"].values() == list(range(lo, hi + 1))
    dairy = [
        r
        for r, c in zip(
            env.products["product_id"].to_pylist(),
            env.products["category"].to_pylist(),
            strict=True,
        )
        if c == "Dairy"
    ]
    assert m.restrictions["product_id"].values() == dairy
    assert m.semi_joins[0].dimension == "products" and m.semi_joins[0].n_keys == hi - lo + 1


def test_dimension_filter_through_on_join_and_non_key_attribute(env):
    m = q(env, "s.region = 'NORTH'", join="JOIN stores s ON f.store_id = s.store_id")
    north = sorted(int(k) for k, v in env.truth["store_regions"].items() if v == "NORTH")
    assert m.restrictions["store_id"].values() == north
    assert len(north) == 12


def test_dimension_filter_combines_with_fact_predicates(env):
    m = q(
        env,
        "p.category IN ('Dairy', 'Meat') AND f.product_id < 400",
        join="JOIN products p USING (product_id)",
    )
    values = m.restrictions["product_id"].values()
    assert (
        values is not None
        and max(values) < 400
        and min(values) >= env.truth["category_blocks"]["Dairy"][0]
    )


def test_outer_join_blocks_pushdown(env):
    m = q(env, "p.category = 'Dairy'", join="LEFT JOIN products p USING (product_id)")
    assert "product_id" not in m.restrictions
    assert m.unprunable == {"product_id": "outer join"}


def test_referenced_columns_and_errors(env):
    m = extract_query(
        "SELECT p.category, sum(f.amount_cents) FROM sales f JOIN products p USING (product_id) "
        "WHERE p.category = 'Dairy' GROUP BY p.category",
        env.resolver,
    )
    assert m.referenced_columns == ["product_id", "amount_cents"]
    assert extract_query("SELECT * FROM sales", env.resolver).referenced_columns[0] == "sale_id"
    assert len(extract_query("SELECT * FROM sales", env.resolver).referenced_columns) == 10
    with pytest.raises(ValueError, match="fact table"):
        extract_query("SELECT 1 FROM other", env.resolver)
    with pytest.raises(ValueError, match="unknown column"):
        extract_query("SELECT 1 FROM sales WHERE nope = 1", env.resolver)


def test_bare_column_against_a_non_literal_is_not_called_a_function(env):
    reason = {"store_id": "non-literal comparison value"}
    assert q(env, "store_id > abs(-4)").unprunable == reason
    assert q(env, "store_id BETWEEN 1 AND abs(-4)").unprunable == reason
    assert q(env, "store_id IN (1, abs(-4))").unprunable == reason
    assert q(env, "abs(store_id) > 4").unprunable == {"store_id": "function on column"}
