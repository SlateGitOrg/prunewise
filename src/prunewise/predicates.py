"""Turn a SQL query on the fact table into a per-column predicate model.

Each column that a query constrains gets an :class:`IntervalSet` that is a *necessary* condition
for a row to match: a row group whose min/max cannot intersect it cannot hold a matching row.
Anything that cannot be turned into such a condition is recorded as unprunable for that column,
with a reason, and simply does not restrict it (the safe direction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import sqlglot
from sqlglot import exp

from .intervals import IntervalSet
from .schema import DIMENSIONS, FACT, FACT_COLUMNS, FACT_KEY, LiteralMismatch, canonical
from .semijoin import DimensionResolver

R_NOT = "negation (NOT)"
R_NEQ = "not-equals (!=)"
R_FUNC = "function on column"
R_NONLIT = "non-literal comparison value"
R_LIKE_LEAD = "leading wildcard LIKE"
R_LIKE_OTHER = "pattern LIKE"
R_ILIKE = "case-insensitive LIKE"
R_CROSS_OR = "disjunction across columns"
R_NULL = "null test"
R_COLCOL = "column-to-column comparison"
R_TYPE = "literal type mismatch"
R_NOZONE = "column not zone-mapped"
R_SPANS = "predicate spans tables"
R_OUTER = "outer join"
R_SUBQ = "subquery"
R_OTHER = "unsupported expression"


@dataclass
class Cond:
    restrictions: dict[str, IntervalSet] = field(default_factory=dict)
    unprunable: dict[str, list[str]] = field(default_factory=dict)

    def note(self, col: str, reason: str) -> None:
        reasons = self.unprunable.setdefault(col, [])
        if reason not in reasons:
            reasons.append(reason)


@dataclass
class SemiJoin:
    dimension: str
    fk: str
    condition: str
    n_keys: int


@dataclass
class QueryModel:
    sql: str
    restrictions: dict[str, IntervalSet]
    unprunable: dict[str, str]
    referenced_columns: list[str]
    semi_joins: list[SemiJoin]


class _NotLiteral(Exception):
    pass


def _unparen(node):
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _literal(node) -> tuple[str, object]:
    node = _unparen(node)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return "str", node.this
        text = str(node.this)
        return "num", (float(text) if any(c in text for c in ".eE") else int(text))
    if isinstance(node, exp.Neg):
        tag, val = _literal(node.this)
        if tag == "num":
            return tag, -val  # type: ignore[operator]
    if isinstance(node, exp.Cast) and node.to.this == exp.DataType.Type.DATE:
        inner = _unparen(node.this)
        if isinstance(inner, exp.Literal) and inner.is_string:
            try:
                return "date", date.fromisoformat(inner.this)
            except ValueError:
                pass
    raise _NotLiteral


def _is_literal(node) -> bool:
    try:
        _literal(node)
    except _NotLiteral:
        return False
    return True


class _Ctx:
    """Alias and column resolution for one query."""

    def __init__(self, select: exp.Select) -> None:
        from_ = select.args.get("from_") or select.args.get("from")
        if from_ is None or not isinstance(from_.this, exp.Table) or from_.this.name != FACT:
            raise ValueError(f"query must select from the fact table {FACT!r}")
        self.tables: dict[str, str] = {FACT: FACT}
        self.fact_alias = from_.this.alias or FACT
        self.tables[self.fact_alias] = FACT
        self.joins: list[dict] = []
        for join in select.args.get("joins") or []:
            table = join.this
            if not isinstance(table, exp.Table) or table.name not in DIMENSIONS:
                raise ValueError("only joins to the products and stores dimensions are supported")
            dim = table.name
            alias = table.alias or dim
            self.tables[alias] = dim
            self.tables[dim] = dim
            self.joins.append(self._join_info(join, dim, alias))

    def _join_info(self, join: exp.Join, dim: str, alias: str) -> dict:
        outer = bool(join.side) and join.side.upper() != "INNER"
        using, on = join.args.get("using"), join.args.get("on")
        fk = dim_key = None
        if using and len(using) == 1:
            fk = dim_key = using[0].name
        elif isinstance(on, exp.EQ) and isinstance(on.this, exp.Column):
            if isinstance(on.expression, exp.Column):
                for fact_side, dim_side in ((on.this, on.expression), (on.expression, on.this)):
                    if self.table_of(fact_side) == FACT and self.table_of(dim_side) == dim:
                        fk, dim_key = fact_side.name, dim_side.name
        pushable = fk is not None and not outer and dim_key == DIMENSIONS[dim][0]
        return {"dim": dim, "alias": alias, "fk": fk, "pushable": pushable, "outer": outer}

    def table_of(self, col: exp.Column) -> str:
        if col.table:
            if col.table not in self.tables:
                raise ValueError(f"unknown table or alias {col.table!r}")
            return self.tables[col.table]
        if col.name in FACT_COLUMNS:
            return FACT
        for join in self.joins:
            if col.name in DIMENSIONS[join["dim"]][1]:
                return join["dim"]
        raise ValueError(f"unknown column {col.name!r}")

    def tables_in(self, node) -> set[str]:
        return {self.table_of(c) for c in node.find_all(exp.Column)}

    def fact_cols_in(self, node) -> list[str]:
        out: list[str] = []
        for c in node.find_all(exp.Column):
            if self.table_of(c) == FACT and c.name not in out:
                out.append(c.name)
        return out

    def bare_fact_column(self, node) -> str | None:
        node = _unparen(node)
        if isinstance(node, exp.Column) and self.table_of(node) == FACT:
            return node.name
        return None


def _restrict(col: str, build) -> Cond:
    spec = FACT_COLUMNS[col]
    cond = Cond()
    if not spec.zonemap:
        cond.note(col, R_NOZONE)
        return cond
    try:
        cond.restrictions[col] = build(spec.kind, spec.kind in ("int", "date"))
    except LiteralMismatch:
        cond.note(col, R_TYPE)
    return cond


def _all_unprunable(ctx: _Ctx, node, reason: str) -> Cond:
    cond = Cond()
    for col in ctx.fact_cols_in(node):
        cond.note(col, reason)
    return cond


def _compare(node, ctx: _Ctx) -> Cond:
    left, right = _unparen(node.this), _unparen(node.expression)
    lcol, rcol = ctx.bare_fact_column(left), ctx.bare_fact_column(right)
    if lcol and rcol:
        return _all_unprunable(ctx, node, R_COLCOL)
    if lcol and _is_literal(right):
        col, lit, op = lcol, _literal(right), type(node)
    elif rcol and _is_literal(left):
        flip = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}
        col, lit, op = rcol, _literal(left), flip.get(type(node), type(node))
    elif ctx.fact_cols_in(node):
        return _all_unprunable(ctx, node, R_NONLIT if lcol or rcol else R_FUNC)
    else:
        return Cond()
    if op is exp.NEQ:
        return _all_unprunable(ctx, node, R_NEQ)

    def build(kind: str, discrete: bool) -> IntervalSet:
        v = canonical(kind, lit)
        if op is exp.EQ:
            return IntervalSet.points(discrete, [v])
        if op is exp.GT:
            return IntervalSet.range(discrete, lo=v, lo_inc=False)
        if op is exp.GTE:
            return IntervalSet.range(discrete, lo=v)
        if op is exp.LT:
            return IntervalSet.range(discrete, hi=v, hi_inc=False)
        return IntervalSet.range(discrete, hi=v)

    return _restrict(col, build)


def _between(node, ctx: _Ctx) -> Cond:
    col = ctx.bare_fact_column(node.this)
    low, high = node.args["low"], node.args["high"]
    if col and _is_literal(low) and _is_literal(high):
        lo_lit, hi_lit = _literal(low), _literal(high)
        return _restrict(
            col,
            lambda k, d: IntervalSet.range(
                d, canonical(k, lo_lit), True, canonical(k, hi_lit), True
            ),
        )
    return _all_unprunable(ctx, node, R_NONLIT if col else R_FUNC)


def _in(node, ctx: _Ctx) -> Cond:
    col = ctx.bare_fact_column(node.this)
    if node.args.get("query") is not None or node.args.get("unnest") is not None:
        return _all_unprunable(ctx, node, R_SUBQ)
    items = node.expressions
    if col and items and all(_is_literal(i) for i in items):
        lits = [_literal(i) for i in items]
        return _restrict(col, lambda k, d: IntervalSet.points(d, [canonical(k, v) for v in lits]))
    return _all_unprunable(ctx, node, R_FUNC if not col else R_NONLIT)


def _like(node, ctx: _Ctx) -> Cond:
    if isinstance(node, exp.ILike):
        return _all_unprunable(ctx, node, R_ILIKE)
    col = ctx.bare_fact_column(node.this)
    pattern = _unparen(node.expression)
    if not col:
        return _all_unprunable(ctx, node, R_FUNC)
    if not (isinstance(pattern, exp.Literal) and pattern.is_string) or node.args.get("escape"):
        return _all_unprunable(ctx, node, R_LIKE_OTHER)
    text = pattern.this
    if text.startswith(("%", "_")):
        return _all_unprunable(ctx, node, R_LIKE_LEAD)
    body = text[:-1] if text.endswith("%") else text
    if "%" in body or "_" in body:
        return _all_unprunable(ctx, node, R_LIKE_OTHER)
    if body == text:
        return _restrict(col, lambda k, d: IntervalSet.points(d, [canonical(k, ("str", text))]))
    upper = body[:-1] + chr(ord(body[-1]) + 1)
    return _restrict(
        col,
        lambda k, d: IntervalSet.range(
            d, canonical(k, ("str", body)), True, canonical(k, ("str", upper)), False
        ),
    )


def _and(parts: list[Cond]) -> Cond:
    out = Cond()
    for part in parts:
        for col, rs in part.restrictions.items():
            out.restrictions[col] = (
                out.restrictions[col].intersect(rs) if col in out.restrictions else rs
            )
        for col, reasons in part.unprunable.items():
            for reason in reasons:
                out.note(col, reason)
    return out


def _or(branches: list[Cond]) -> Cond:
    """A column stays restricted only if every branch restricts it; otherwise it is unprunable."""
    out = Cond()
    cols: list[str] = []
    for b in branches:
        for col in [*b.restrictions, *b.unprunable]:
            if col not in cols:
                cols.append(col)
    for col in cols:
        if all(col in b.restrictions for b in branches):
            merged = branches[0].restrictions[col]
            for b in branches[1:]:
                merged = merged.union(b.restrictions[col])
            out.restrictions[col] = merged
            continue
        reasons = [r for b in branches for r in b.unprunable.get(col, [])]
        for reason in reasons or [R_CROSS_OR]:
            out.note(col, reason)
    return out


def _analyze(node, ctx: _Ctx) -> Cond:
    node = _unparen(node)
    if isinstance(node, exp.And):
        return _and([_analyze(node.this, ctx), _analyze(node.expression, ctx)])
    if isinstance(node, exp.Or):
        return _or([_analyze(node.this, ctx), _analyze(node.expression, ctx)])
    if isinstance(node, exp.Not):
        return _all_unprunable(ctx, node, R_NOT)
    if isinstance(node, exp.NEQ):
        return _all_unprunable(ctx, node, R_NEQ)
    if isinstance(node, exp.EQ | exp.GT | exp.GTE | exp.LT | exp.LTE):
        return _compare(node, ctx)
    if isinstance(node, exp.Between):
        return _between(node, ctx)
    if isinstance(node, exp.In):
        return _in(node, ctx)
    if isinstance(node, exp.Like | exp.ILike):
        return _like(node, ctx)
    if isinstance(node, exp.Is):
        return _all_unprunable(ctx, node, R_NULL)
    return _all_unprunable(ctx, node, R_FUNC if isinstance(node, exp.Func) else R_OTHER)


def _conjuncts(node):
    node = _unparen(node)
    if isinstance(node, exp.And):
        yield from _conjuncts(node.this)
        yield from _conjuncts(node.expression)
    else:
        yield node


def _referenced_columns(select: exp.Select, ctx: _Ctx) -> list[str]:
    names: set[str] = set()
    for item in select.expressions:
        if isinstance(item, exp.Star) or (
            isinstance(item, exp.Column) and isinstance(item.this, exp.Star)
        ):
            return list(FACT_COLUMNS)
    for col in select.find_all(exp.Column):
        if not isinstance(col.this, exp.Star) and ctx.table_of(col) == FACT:
            names.add(col.name)
    for join in ctx.joins:
        if join["fk"]:
            names.add(join["fk"])
    return [c for c in FACT_COLUMNS if c in names] or [FACT_KEY]


def _strip_qualifiers(node: exp.Expression) -> exp.Expression:
    return node.copy().transform(lambda n: exp.column(n.name) if isinstance(n, exp.Column) else n)


def extract_query(sql: str, resolver: DimensionResolver) -> QueryModel:
    select = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(select, exp.Select):
        raise ValueError("only SELECT statements are supported")
    ctx = _Ctx(select)
    parts: list[Cond] = []
    dim_conds: dict[str, list[exp.Expression]] = {}
    where = select.args.get("where")
    for conj in _conjuncts(where.this) if where is not None else []:
        tables = ctx.tables_in(conj)
        dims = tables - {FACT}
        if not dims:
            parts.append(_analyze(conj, ctx))
        elif not (tables & {FACT}) and len(dims) == 1:
            dim_conds.setdefault(next(iter(dims)), []).append(conj)
        else:
            parts.append(_all_unprunable(ctx, conj, R_SPANS))
    semi: list[SemiJoin] = []
    for join in ctx.joins:
        conds = dim_conds.get(join["dim"])
        if not conds or join["fk"] is None:
            continue
        if not join["pushable"]:
            cond = Cond()
            cond.note(join["fk"], R_OUTER if join["outer"] else R_OTHER)
            parts.append(cond)
            continue
        condition = exp.and_(*[_strip_qualifiers(c) for c in conds]).sql(dialect="duckdb")
        keys = resolver.keys(join["dim"], condition)
        parts.append(Cond({join["fk"]: IntervalSet.points(True, keys)}))
        semi.append(SemiJoin(join["dim"], join["fk"], condition, len(keys)))
    merged = _and(parts)
    return QueryModel(
        sql=sql,
        restrictions=merged.restrictions,
        unprunable={c: "; ".join(r) for c, r in merged.unprunable.items()},
        referenced_columns=_referenced_columns(select, ctx),
        semi_joins=semi,
    )


def truth_sql(sql: str) -> str:
    """The query rewritten to return the ``sale_id`` of every fact row it reads (ground truth)."""
    select = sqlglot.parse_one(sql, read="duckdb")
    ctx = _Ctx(select)
    select.set("expressions", [exp.column(FACT_KEY, table=ctx.fact_alias)])
    for key in ("group", "having", "order", "limit", "qualify", "distinct"):
        select.set(key, None)
    return select.sql(dialect="duckdb")
