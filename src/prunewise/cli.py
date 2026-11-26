"""Command line interface: generate, analyze-workload, rank, materialize, validate, report."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .layouts import baseline_layout
from .materialize import materialize_layout, write_manifest
from .ranking import Ranking, rank_layouts
from .report import render_markdown
from .semijoin import DimensionResolver
from .synth import generate as synth_generate
from .synth import read_tables, write_dataset
from .validate import is_clean, validate_lake
from .workload import Workload, analyze_log

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


@app.command()
def generate(
    config: str = typer.Option("demo", help="Preset (tests, demo) or a TOML file."),
    seed: int = typer.Option(7, help="Random seed."),
    out: Path = typer.Option(Path("data"), help="Output directory."),
) -> None:
    """Write the synthetic sales, products and stores tables and the query log."""
    cfg = load_config(config)
    write_dataset(synth_generate(cfg.n_rows, seed), out)
    console.print(f"wrote {cfg.n_rows:,} line items and queries.jsonl to {out}/")


@app.command("analyze-workload")
def analyze_workload(
    data: Path = typer.Option(Path("data"), help="Directory from `generate`."),
    queries: Path | None = typer.Option(None, help="JSONL log (default: <data>/queries.jsonl)."),
    out: Path = typer.Option(Path("out/workload.json"), help="Workload model JSON."),
) -> None:
    """Parse the query log into per-query predicates, resolving dimension joins."""
    _, products, stores = read_tables(data)
    resolver = DimensionResolver(products, stores)
    workload = analyze_log(queries or data / "queries.jsonl", resolver)
    resolver.close()
    workload.save(out)
    unprunable = len(workload.unprunable_list())
    console.print(f"{len(workload.queries)} queries, {unprunable} unprunable predicates -> {out}")


@app.command()
def rank(
    data: Path = typer.Option(Path("data")),
    workload: Path = typer.Option(Path("out/workload.json")),
    config: str = typer.Option("demo", help="Preset (tests, demo) or a TOML file."),
    out: Path = typer.Option(Path("out/ranking.json")),
) -> None:
    """Simulate every candidate layout on the workload and rank by weighted cost."""
    sales, _, _ = read_tables(data)
    ranking = rank_layouts(sales, Workload.load(workload), load_config(config))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ranking.to_dict(), indent=1) + "\n")
    best = ranking.ranked[0]
    console.print(
        f"{len(ranking.ranked)} layouts ranked, {len(ranking.candidates) - len(ranking.ranked)} "
        f"excluded; best {best.layout.layout_id} at {best.ratio_to_baseline:.3f} of baseline cost"
    )


@app.command()
def materialize(
    data: Path = typer.Option(Path("data")),
    ranking: Path = typer.Option(Path("out/ranking.json")),
    out: Path = typer.Option(Path("lake")),
    top: int = typer.Option(
        3, help="Number of top-ranked layouts to write (baseline is always written)."
    ),
) -> None:
    """Write the baseline and the top-ranked layouts as hive-partitioned Parquet."""
    rk = Ranking.from_dict(json.loads(ranking.read_text()))
    sales, _, _ = read_tables(data)
    base = baseline_layout(rk.config)
    layouts = [base] + [ev.layout for ev in rk.ranked[:top]]
    for layout in layouts:
        plan = materialize_layout(sales, layout, out)
        console.print(
            f"wrote {layout.layout_id}: {len(plan.files)} files, {plan.n_row_groups} row groups"
        )
    write_manifest(out, layouts, base)


@app.command()
def validate(
    data: Path = typer.Option(Path("data")),
    lake: Path = typer.Option(Path("lake")),
    workload: Path = typer.Option(Path("out/workload.json")),
    ranking: Path = typer.Option(Path("out/ranking.json")),
    out: Path = typer.Option(Path("out/validation.json")),
) -> None:
    """Check materialised layouts against footers, a row-level soundness scan and DuckDB."""
    rk = Ranking.from_dict(json.loads(ranking.read_text()))
    report = validate_lake(data, lake, Workload.load(workload), rk)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    s = report["summary"]
    console.print(
        f"soundness violations {s['soundness_violations']}, simulator-vs-footer mismatches "
        f"{s['simulator_footer_mismatches']}, DuckDB result mismatches "
        f"{s['duckdb_result_mismatches']}"
    )
    if not is_clean(report):
        raise typer.Exit(1)


@app.command()
def report(
    ranking: Path = typer.Option(Path("out/ranking.json")),
    workload: Path = typer.Option(Path("out/workload.json")),
    validation: Path | None = typer.Option(None, help="validation.json to append, if any."),
    top: int = typer.Option(8, help="Rows in the ranked table."),
    breakdown: int = typer.Option(3, help="Top layouts that get a per-query breakdown."),
    out: Path | None = typer.Option(None, help="Write Markdown here instead of stdout."),
) -> None:
    """Print the ranked layouts, per-query breakdowns, unprunable predicates and validation."""
    rk = Ranking.from_dict(json.loads(ranking.read_text()))
    val = json.loads(validation.read_text()) if validation else None
    text = render_markdown(rk, Workload.load(workload), val, top, breakdown)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        typer.echo(text)


if __name__ == "__main__":
    app()
