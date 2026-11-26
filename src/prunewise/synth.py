"""Seeded generator for a grocery point-of-sale lake and its query log.

Documented properties (all data is synthetic):
  * 60 stores in 5 regions, 12 per region, with region assigned independently of store id.
  * 2,000 products in 12 categories; product ids are contiguous blocks per category.
  * 90 days (2026-01-01 to 2026-03-31) of line items with a weekend lift and store-size skew.
  * Rows are in ingestion order: each store's feed lands 0-3 days late, so the order is roughly but
    not exactly by date, as in a lake that is loaded by store feeds.
  * The query log has 25 templates whose weights are 55% region equality plus a recent-days range,
    20% category through the products join, 10% one store plus the last 7 days, 10% full-period
    aggregates and 5% unprunable on the fact table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import FACT_ARROW_SCHEMA

REGIONS = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]
CATEGORIES = [
    "Produce",
    "Dairy",
    "Bakery",
    "Meat",
    "Seafood",
    "Frozen",
    "Pantry",
    "Beverages",
    "Snacks",
    "Household",
    "Personal Care",
    "Deli",
]
N_STORES = 60
N_PRODUCTS = 2_000
START = date(2026, 1, 1)
N_DAYS = 90
LAST_DAY = START + timedelta(days=N_DAYS - 1)
PROMOS = [("BOGO10", 50), ("BOGO25", 50), ("SAVE5", 5), ("LOYAL2", 2), ("CLEARANCE", 30)]


@dataclass
class SynthData:
    sales: pa.Table
    products: pa.Table
    stores: pa.Table
    queries: list[dict]
    truth: dict


def _stores(rng: np.random.Generator) -> tuple[pa.Table, np.ndarray]:
    regions = rng.permutation(np.repeat(REGIONS, N_STORES // len(REGIONS)))
    ids = np.arange(1, N_STORES + 1)
    opened = [
        (START - timedelta(days=int(d))).isoformat() for d in rng.integers(100, 4000, N_STORES)
    ]
    table = pa.table(
        {
            "store_id": pa.array(ids, pa.int32()),
            "store_name": [f"Store {i:03d}" for i in ids],
            "region": regions.tolist(),
            "opened_date": pa.array([date.fromisoformat(o) for o in opened], pa.date32()),
        }
    )
    return table, regions


def _products(rng: np.random.Generator) -> tuple[pa.Table, dict[str, list[int]], np.ndarray]:
    base = N_PRODUCTS // len(CATEGORIES)
    sizes = [base + 1 if i < N_PRODUCTS - base * len(CATEGORIES) else base for i in range(12)]
    cats: list[str] = []
    blocks: dict[str, list[int]] = {}
    start = 1
    for name, size in zip(CATEGORIES, sizes, strict=True):
        cats.extend([name] * size)
        blocks[name] = [start, start + size - 1]
        start += size
    ids = np.arange(1, N_PRODUCTS + 1)
    cat_price = rng.integers(150, 1200, len(CATEGORIES))
    price = np.array(
        [int(cat_price[CATEGORIES.index(c)] * rng.lognormal(0, 0.35)) for c in cats]
    ).clip(49, None)
    table = pa.table(
        {
            "product_id": pa.array(ids, pa.int32()),
            "sku": [f"SKU{i:05d}" for i in ids],
            "product_name": [f"{c} item {i}" for c, i in zip(cats, ids, strict=True)],
            "category": cats,
            "unit_price_cents": pa.array(price, pa.int32()),
        }
    )
    return table, blocks, price


def _sales(rng, n, store_regions, price) -> pa.Table:
    days = np.arange(N_DAYS)
    weekday = np.array([(START + timedelta(days=int(d))).weekday() for d in days])
    day_w = (1 + 0.25 * (weekday >= 5)) * (1 + 0.15 * days / N_DAYS)
    day_idx = rng.choice(N_DAYS, n, p=day_w / day_w.sum())
    store_w = rng.lognormal(0, 0.5, N_STORES)
    store_idx = rng.choice(N_STORES, n, p=store_w / store_w.sum())
    prod_w = rng.permutation(1.0 / np.arange(1, N_PRODUCTS + 1) ** 0.6)
    prod_idx = rng.choice(N_PRODUCTS, n, p=prod_w / prod_w.sum())
    hour_w = np.array(
        [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 8, 8, 7, 6, 4, 2, 1, 1, 1, 1, 1], float
    )
    hours = rng.choice(24, n, p=hour_w / hour_w.sum())
    seconds = hours * 3600 + rng.integers(0, 3600, n)
    quantity = np.minimum(1 + rng.poisson(0.8, n), 6)
    promo_pick = rng.choice(len(PROMOS) + 1, n, p=[0.9] + [0.02] * len(PROMOS))
    # promo_pick 0 means no promo (90% of rows); 1..5 pick one of PROMOS (2% each)
    gross = quantity * price[prod_idx]
    pct = np.array([0] + [p[1] for p in PROMOS])[promo_pick]
    amount = gross * (100 - pct) // 100
    delay = rng.integers(0, 3 * 86400, N_STORES)
    ingest = day_idx * 86400 + seconds + delay[store_idx] + rng.integers(0, 6 * 3600, n)
    order = np.argsort(ingest, kind="stable")
    day_idx, store_idx, prod_idx = day_idx[order], store_idx[order], prod_idx[order]
    seconds, quantity, amount, promo_pick = (
        seconds[order],
        quantity[order],
        amount[order],
        promo_pick[order],
    )
    months = np.array([(START + timedelta(days=int(d))).strftime("%Y-%m") for d in days])
    epoch_days = (START - date(1970, 1, 1)).days + day_idx
    ts_us = (epoch_days.astype(np.int64) * 86400 + seconds) * 1_000_000
    promo = [None if k == 0 else PROMOS[k - 1][0] for k in promo_pick]
    return pa.table(
        {
            "sale_id": pa.array(np.arange(1, n + 1), pa.int64()),
            "sale_date": pa.array(epoch_days.astype(np.int32), pa.date32()),
            "sale_month": pa.array(months[day_idx].tolist(), pa.string()),
            "sale_ts": pa.array(ts_us, pa.timestamp("us")),
            "store_id": pa.array(store_idx + 1, pa.int32()),
            "store_region": pa.array(store_regions[store_idx].tolist(), pa.string()),
            "product_id": pa.array(prod_idx + 1, pa.int32()),
            "quantity": pa.array(quantity, pa.int32()),
            "amount_cents": pa.array(amount, pa.int64()),
            "promo_code": pa.array(promo, pa.string()),
        },
        schema=FACT_ARROW_SCHEMA,
    )


def _d(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def query_log() -> list[dict]:
    """The 25 query templates with executions per day (weights sum to 1,000)."""
    last7 = LAST_DAY - timedelta(days=6)
    last28 = LAST_DAY - timedelta(days=27)
    rows: list[tuple[str, int]] = []
    for region in REGIONS:
        rows.append(
            (
                "SELECT f.store_id, sum(f.amount_cents) AS revenue FROM sales f "
                f"WHERE f.store_region = '{region}' AND f.sale_date >= {_d(last7)} "
                "GROUP BY f.store_id",
                70,
            )
        )
        rows.append(
            (
                "SELECT f.sale_date, count(*) AS lines, sum(f.quantity) AS units FROM sales f "
                f"WHERE f.store_region = '{region}' "
                f"AND f.sale_date BETWEEN '{last28}' AND '{LAST_DAY}' GROUP BY f.sale_date",
                40,
            )
        )
    cat = (
        "SELECT p.category, sum(f.amount_cents) AS revenue FROM sales f "
        "JOIN products p USING (product_id) WHERE {where} GROUP BY p.category"
    )
    rows += [
        (cat.format(where="p.category = 'Dairy'"), 40),
        (cat.format(where="p.category = 'Produce'"), 35),
        (cat.format(where=f"p.category = 'Dairy' AND f.sale_date >= {_d(last28)}"), 25),
        (cat.format(where="p.category = 'Bakery'"), 25),
        (cat.format(where="p.category IN ('Meat', 'Seafood')"), 20),
        (cat.format(where="p.category = 'Beverages'"), 30),
        (cat.format(where=f"p.category = 'Snacks' AND f.sale_date >= {_d(last7)}"), 25),
    ]
    store = "SELECT f.product_id, sum(f.quantity) AS units FROM sales f {join} WHERE {where} "
    rows += [
        (
            store.format(join="", where=f"f.store_id = 7 AND f.sale_date >= {_d(last7)}")
            + "GROUP BY f.product_id",
            40,
        ),
        (
            store.format(
                join="JOIN stores s ON f.store_id = s.store_id",
                where=f"s.store_name = 'Store 023' AND f.sale_date >= {_d(last7)}",
            )
            + "GROUP BY f.product_id",
            30,
        ),
        (
            store.format(join="", where=f"f.store_id = 41 AND f.sale_date >= {_d(last7)}")
            + "GROUP BY f.product_id",
            30,
        ),
    ]
    rows += [
        (
            "SELECT sum(f.amount_cents) AS revenue FROM sales f "
            f"WHERE f.sale_date BETWEEN '{START}' AND '{LAST_DAY}'",
            40,
        ),
        (
            "SELECT f.sale_month, sum(f.amount_cents) AS revenue "
            "FROM sales f GROUP BY f.sale_month",
            30,
        ),
        (
            "SELECT f.store_region, count(*) AS lines, sum(f.quantity) AS units FROM sales f "
            "GROUP BY f.store_region",
            30,
        ),
    ]
    rows += [
        (
            "SELECT f.promo_code, count(*) AS lines FROM sales f "
            "WHERE f.promo_code LIKE '%BOGO%' GROUP BY f.promo_code",
            30,
        ),
        (
            "SELECT f.store_id, count(*) AS lines FROM sales f "
            "WHERE extract(hour FROM f.sale_ts) = 18 GROUP BY f.store_id",
            20,
        ),
    ]
    return [
        {"query_id": f"q{i:02d}", "sql": sql, "executions_per_day": freq}
        for i, (sql, freq) in enumerate(rows, start=1)
    ]


def generate(n_rows: int, seed: int = 7) -> SynthData:
    rng = np.random.default_rng(seed)
    stores, regions = _stores(rng)
    products, blocks, price = _products(rng)
    sales = _sales(rng, n_rows, regions, price)
    queries = query_log()
    truth = {
        "seed": seed,
        "n_rows": n_rows,
        "first_day": START.isoformat(),
        "last_day": LAST_DAY.isoformat(),
        "category_blocks": blocks,
        "store_regions": {str(i + 1): str(r) for i, r in enumerate(regions)},
        "template_weights": {
            "region_recent_days": 0.55,
            "category_join": 0.20,
            "single_store_7d": 0.10,
            "full_period": 0.10,
            "unprunable": 0.05,
        },
        "unprunable_queries": {
            queries[-2]["query_id"]: "leading wildcard LIKE",
            queries[-1]["query_id"]: "function on column",
        },
    }
    return SynthData(sales, products, stores, queries, truth)


def write_dataset(data: SynthData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(data.sales, out / "sales.parquet", compression="zstd")
    pq.write_table(data.products, out / "products.parquet", compression="zstd")
    pq.write_table(data.stores, out / "stores.parquet", compression="zstd")
    with (out / "queries.jsonl").open("w") as fh:
        for q in data.queries:
            fh.write(json.dumps(q) + "\n")
    (out / "truth.json").write_text(json.dumps(data.truth, indent=2) + "\n")


def read_tables(data_dir: Path) -> tuple[pa.Table, pa.Table, pa.Table]:
    return (
        pq.read_table(data_dir / "sales.parquet"),
        pq.read_table(data_dir / "products.parquet"),
        pq.read_table(data_dir / "stores.parquet"),
    )
