"""Test configuration: keep scratch files inside the repository's git-ignored .tmp/."""

from __future__ import annotations

from pathlib import Path


def pytest_configure(config) -> None:
    if config.option.basetemp is None:
        base = Path(config.rootpath) / ".tmp"
        base.mkdir(exist_ok=True)
        config.option.basetemp = base / "pytest"


import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from prunewise.config import PRESETS  # noqa: E402
from prunewise.layouts import baseline_layout  # noqa: E402
from prunewise.materialize import materialize_layout, write_manifest  # noqa: E402
from prunewise.plan import load_columns  # noqa: E402
from prunewise.ranking import cardinalities, rank_layouts  # noqa: E402
from prunewise.semijoin import DimensionResolver  # noqa: E402
from prunewise.soundness import build_oracles  # noqa: E402
from prunewise.synth import generate, read_tables, write_dataset  # noqa: E402
from prunewise.validate import validate_lake  # noqa: E402
from prunewise.workload import analyze_log  # noqa: E402


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    """Tests-config dataset, workload, ranking, three materialised layouts and validation."""
    root = tmp_path_factory.mktemp("env")
    cfg = PRESETS["tests"]
    data = generate(cfg.n_rows, seed=7)
    write_dataset(data, root / "data")
    sales, products, stores = read_tables(root / "data")
    resolver = DimensionResolver(products, stores)
    workload = analyze_log(root / "data" / "queries.jsonl", resolver)
    ranking = rank_layouts(sales, workload, cfg)
    base = baseline_layout(cfg)
    layouts = [base] + [ev.layout for ev in ranking.ranked[:3]]
    for layout in layouts:
        materialize_layout(sales, layout, root / "lake")
    write_manifest(root / "lake", layouts, base)
    validation = validate_lake(root / "data", root / "lake", workload, ranking)
    return SimpleNamespace(
        root=root,
        cfg=cfg,
        truth=json.loads((root / "data" / "truth.json").read_text()),
        sales=sales,
        products=products,
        stores=stores,
        resolver=resolver,
        columns=load_columns(sales),
        card=cardinalities(
            load_columns(sales), ["sale_date", "sale_month", "store_region", "store_id"]
        ),
        workload=workload,
        ranking=ranking,
        layouts=layouts,
        lake=root / "lake",
        validation=validation,
        oracles=build_oracles(sales, products, stores, workload),
    )
