"""Write a layout to disk exactly as the plan describes it."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .bytesmodel import COMPRESSION
from .layouts import Layout
from .plan import LayoutPlan, build_plan, load_columns
from .schema import FACT_COLUMNS, days_to_date

FILE_NAME = "part-0.parquet"


def partition_dir(layout: Layout, values: tuple) -> str:
    """Hive-style directory for one file's partition values (empty for an unpartitioned layout)."""
    parts = []
    for col, v in zip(layout.partition, values, strict=True):
        text = days_to_date(v).isoformat() if FACT_COLUMNS[col].kind == "date" else str(v)
        parts.append(f"{col}={text}")
    return "/".join(parts)


def file_path(root: Path, layout: Layout, values: tuple) -> Path:
    sub = partition_dir(layout, values)
    return root / layout.layout_id / sub / FILE_NAME if sub else root / layout.layout_id / FILE_NAME


def materialize_layout(table: pa.Table, layout: Layout, root: Path) -> LayoutPlan:
    columns = load_columns(
        table, list({*layout.partition, *([layout.sort] if layout.sort else [])})
    )
    plan = build_plan(columns, table.num_rows, layout)
    for f in plan.files:
        path = file_path(root, layout, f.partition)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table.take(f.rows),
            path,
            row_group_size=layout.row_group_size,
            compression=COMPRESSION,
        )
    return plan


def write_manifest(root: Path, layouts: list[Layout], baseline: Layout) -> None:
    doc = {"baseline": baseline.layout_id, "layouts": [lay.to_dict() for lay in layouts]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(doc, indent=1) + "\n")


def read_manifest(root: Path) -> tuple[Layout, list[Layout]]:
    doc = json.loads((root / "manifest.json").read_text())
    layouts = [Layout.from_dict(d) for d in doc["layouts"]]
    base = next(lay for lay in layouts if lay.layout_id == doc["baseline"])
    return base, layouts
