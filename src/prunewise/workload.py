"""The workload model: every logged query as predicates on the fact table, with a frequency."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .intervals import IntervalSet
from .predicates import QueryModel, SemiJoin, extract_query
from .schema import FACT_COLUMNS, from_json_value, to_json_value
from .semijoin import DimensionResolver


@dataclass
class WorkloadQuery:
    query_id: str
    sql: str
    executions_per_day: float
    model: QueryModel

    @property
    def restrictions(self) -> dict[str, IntervalSet]:
        return self.model.restrictions

    @property
    def unprunable(self) -> dict[str, str]:
        return self.model.unprunable

    @property
    def referenced_columns(self) -> list[str]:
        return self.model.referenced_columns


@dataclass
class Workload:
    queries: list[WorkloadQuery]

    @property
    def zone_columns(self) -> list[str]:
        """Columns that any query constrains: the ones a simulator must keep min/max for."""
        cols = {c for q in self.queries for c in q.restrictions}
        return [c for c in FACT_COLUMNS if c in cols]

    def scaled(self, factor: float) -> Workload:
        return Workload(
            [replace(q, executions_per_day=q.executions_per_day * factor) for q in self.queries]
        )

    def with_query(self, query: WorkloadQuery) -> Workload:
        return Workload([*self.queries, query])

    def unprunable_list(self) -> list[dict]:
        return [
            {"query_id": q.query_id, "column": col, "reason": reason, "sql": q.sql}
            for q in self.queries
            for col, reason in q.unprunable.items()
        ]

    # ---- JSON ---------------------------------------------------------
    def to_json(self) -> dict:
        out = []
        for q in self.queries:
            restrictions = {}
            for col, rs in q.restrictions.items():
                kind = FACT_COLUMNS[col].kind
                restrictions[col] = [
                    [to_json_value(kind, iv[0]), iv[1], to_json_value(kind, iv[2]), iv[3]]
                    for iv in rs.intervals
                ]
            out.append(
                {
                    "query_id": q.query_id,
                    "sql": q.sql,
                    "executions_per_day": q.executions_per_day,
                    "restrictions": restrictions,
                    "unprunable": q.unprunable,
                    "referenced_columns": q.referenced_columns,
                    "semi_joins": [vars(s) for s in q.model.semi_joins],
                }
            )
        return {"queries": out}

    @classmethod
    def from_json(cls, doc: dict) -> Workload:
        queries = []
        for item in doc["queries"]:
            restrictions = {}
            for col, ivs in item["restrictions"].items():
                kind = FACT_COLUMNS[col].kind
                items = [
                    [from_json_value(kind, i[0]), i[1], from_json_value(kind, i[2]), i[3]]
                    for i in ivs
                ]
                restrictions[col] = IntervalSet.from_list(kind in ("int", "date"), items)
            model = QueryModel(
                sql=item["sql"],
                restrictions=restrictions,
                unprunable=item["unprunable"],
                referenced_columns=item["referenced_columns"],
                semi_joins=[SemiJoin(**s) for s in item["semi_joins"]],
            )
            queries.append(
                WorkloadQuery(item["query_id"], item["sql"], item["executions_per_day"], model)
            )
        return cls(queries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> Workload:
        return cls.from_json(json.loads(path.read_text()))


def analyze_log(log_path: Path, resolver: DimensionResolver) -> Workload:
    queries = []
    for lineno, line in enumerate(log_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            model = extract_query(rec["sql"], resolver)
            queries.append(
                WorkloadQuery(
                    str(rec.get("query_id", f"q{lineno:02d}")),
                    rec["sql"],
                    float(rec["executions_per_day"]),
                    model,
                )
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{log_path}:{lineno}: {exc}") from exc
    return Workload(queries)
