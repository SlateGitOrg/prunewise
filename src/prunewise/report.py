"""Markdown report: ranked layouts, per-query breakdowns, unprunable predicates, validation."""

from __future__ import annotations

from .ranking import LayoutEvaluation, Ranking
from .workload import Workload


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def ranked_table(ranking: Ranking, top: int) -> str:
    rows = []
    entries: list[tuple[str, LayoutEvaluation]] = [("baseline", ranking.baseline)]
    entries += [(str(i), ev) for i, ev in enumerate(ranking.ranked[:top], start=1)]
    for rank, ev in entries:
        ratio = f"{ev.ratio_to_baseline:.3f}" if ev.ratio_to_baseline is not None else "n/a"
        rows.append(
            [
                rank,
                f"`{ev.layout.layout_id}`",
                str(ev.files),
                str(ev.row_groups),
                human_bytes(ev.bytes_per_day),
                ratio,
            ]
        )
    return _table(
        ["rank", "layout", "files", "row groups", "predicted bytes/day", "cost ratio to baseline"],
        rows,
    )


def breakdown_table(ev: LayoutEvaluation) -> str:
    rows = [
        [
            c.query_id,
            f"{c.frequency:g}",
            str(c.files_opened),
            f"{c.row_groups_read} of {c.row_groups_total}",
            human_bytes(c.bytes_read),
        ]
        for c in ev.costs
    ]
    return _table(
        ["query", "executions/day", "files opened", "row groups read", "bytes read"], rows
    )


def unprunable_table(workload: Workload) -> str:
    rows = [[u["query_id"], f"`{u['column']}`", u["reason"]] for u in workload.unprunable_list()]
    return _table(["query", "column", "reason"], rows) if rows else "None."


def validation_summary(validation: dict) -> str:
    s = validation["summary"]
    rows = [
        ["layouts validated", str(len(validation["layouts"]))],
        ["queries per layout", str(validation["queries"])],
        ["row groups re-scanned with Polars", str(s["row_groups_rescanned"])],
        ["soundness violations", str(s["soundness_violations"])],
        ["simulator-vs-footer mismatches", str(s["simulator_footer_mismatches"])],
        ["DuckDB result mismatches", str(s["duckdb_result_mismatches"])],
        ["extractor violations (real match outside model)", str(s["extractor_violations"])],
    ]
    return _table(["check", "result"], rows)


def render_markdown(
    ranking: Ranking,
    workload: Workload,
    validation: dict | None = None,
    top: int = 8,
    breakdown: int = 3,
) -> str:
    out = ["## Ranked layouts", "", ranked_table(ranking, top), ""]
    excluded = [c for c in ranking.candidates if c.excluded]
    if excluded:
        out += ["## Excluded layouts", ""]
        seen: set[str] = set()
        for c in excluded:
            key = f"{'-'.join(c.layout.partition)}: {c.exclusion_reason}"
            if key not in seen:
                seen.add(key)
                out.append(f"- partition `{'+'.join(c.layout.partition)}`: {c.exclusion_reason}")
        out.append("")
    for i, ev in enumerate(ranking.ranked[:breakdown], start=1):
        out += [
            f"## Per-query breakdown, rank {i}: `{ev.layout.layout_id}`",
            "",
            breakdown_table(ev),
            "",
        ]
    out += ["## Unprunable predicates", "", unprunable_table(workload), ""]
    if validation is not None:
        out += ["## Validation", "", validation_summary(validation), ""]
    return "\n".join(out)
