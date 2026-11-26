"""Interval sets over a totally ordered domain: the shape of every per-column predicate.

An equality set, an IN set, a closed or open range, a conjunction (intersection) and a disjunction
on one column (union) are all sets of intervals. A row group with min/max ``[mn, mx]`` can contain a
matching row only if ``[mn, mx]`` intersects the set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# (lo, lo_inclusive, hi, hi_inclusive); None bound means unbounded on that side.
Interval = tuple[Any, bool, Any, bool]


def _lo_key(iv: Interval):
    # unbounded sorts first; an inclusive bound starts before an exclusive one at the same value
    return (0, 0, 0) if iv[0] is None else (1, iv[0], 0 if iv[1] else 1)


def _hi_max(a_hi, a_inc, b_hi, b_inc):
    if a_hi is None or b_hi is None:
        return None, True
    if a_hi > b_hi:
        return a_hi, a_inc
    if b_hi > a_hi:
        return b_hi, b_inc
    return a_hi, a_inc or b_inc


def _empty(lo, lo_inc, hi, hi_inc) -> bool:
    if lo is None or hi is None:
        return False
    return lo > hi or (lo == hi and not (lo_inc and hi_inc))


@dataclass(frozen=True)
class IntervalSet:
    discrete: bool
    intervals: tuple[Interval, ...]

    # ---- construction -------------------------------------------------
    @classmethod
    def build(cls, discrete: bool, parts: list[Interval]) -> IntervalSet:
        norm: list[Interval] = []
        for lo, lo_inc, hi, hi_inc in parts:
            if discrete:
                if lo is not None and not lo_inc:
                    lo, lo_inc = lo + 1, True
                if hi is not None and not hi_inc:
                    hi, hi_inc = hi - 1, True
            if not _empty(lo, lo_inc, hi, hi_inc):
                norm.append((lo, lo_inc, hi, hi_inc))
        norm.sort(key=_lo_key)
        merged: list[Interval] = []
        for iv in norm:
            if merged and cls._touches(discrete, merged[-1], iv):
                prev = merged[-1]
                hi, hi_inc = _hi_max(prev[2], prev[3], iv[2], iv[3])
                merged[-1] = (prev[0], prev[1], hi, hi_inc)
            else:
                merged.append(iv)
        return cls(discrete, tuple(merged))

    @staticmethod
    def _touches(discrete: bool, a: Interval, b: Interval) -> bool:
        if a[2] is None or b[0] is None:
            return True
        if b[0] < a[2]:
            return True
        if b[0] == a[2]:
            return a[3] or b[1]
        return discrete and b[0] == a[2] + 1

    @classmethod
    def full(cls, discrete: bool) -> IntervalSet:
        return cls(discrete, ((None, True, None, True),))

    @classmethod
    def empty(cls, discrete: bool) -> IntervalSet:
        return cls(discrete, ())

    @classmethod
    def points(cls, discrete: bool, values) -> IntervalSet:
        return cls.build(discrete, [(v, True, v, True) for v in values])

    @classmethod
    def range(cls, discrete: bool, lo=None, lo_inc=True, hi=None, hi_inc=True) -> IntervalSet:
        return cls.build(discrete, [(lo, lo_inc, hi, hi_inc)])

    # ---- algebra ------------------------------------------------------
    def union(self, other: IntervalSet) -> IntervalSet:
        return IntervalSet.build(self.discrete, list(self.intervals) + list(other.intervals))

    def intersect(self, other: IntervalSet) -> IntervalSet:
        parts: list[Interval] = []
        for a in self.intervals:
            for b in other.intervals:
                lo, lo_inc = a[0], a[1]
                if b[0] is not None and (lo is None or b[0] > lo or (b[0] == lo and not b[1])):
                    lo, lo_inc = b[0], b[1]
                hi, hi_inc = a[2], a[3]
                if b[2] is not None and (hi is None or b[2] < hi or (b[2] == hi and not b[3])):
                    hi, hi_inc = b[2], b[3]
                parts.append((lo, lo_inc, hi, hi_inc))
        return IntervalSet.build(self.discrete, parts)

    # ---- queries ------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.intervals

    def contains(self, v) -> bool:
        return self.intersects(v, v)

    def intersects(self, mn, mx) -> bool:
        """True if some value in the closed range [mn, mx] may belong to the set.

        Missing statistics (None) can never rule a row group out.
        """
        if mn is None or mx is None:
            return True
        for lo, lo_inc, hi, hi_inc in self.intervals:
            if lo is not None and (mx < lo or (mx == lo and not lo_inc)):
                continue
            if hi is not None and (mn > hi or (mn == hi and not hi_inc)):
                continue
            return True
        return False

    def values(self, limit: int = 100_000) -> list | None:
        """The finite members of an integer-valued set, or None if unbounded or too large."""
        if not self.discrete:
            return [iv[0] for iv in self.intervals] if self._all_points() else None
        out: list = []
        for lo, _, hi, _ in self.intervals:
            if lo is None or hi is None or len(out) + (hi - lo + 1) > limit:
                return None
            out.extend(range(lo, hi + 1))
        return out

    def _all_points(self) -> bool:
        return all(iv[0] is not None and iv[0] == iv[2] for iv in self.intervals)

    def to_list(self) -> list[list]:
        return [list(iv) for iv in self.intervals]

    @classmethod
    def from_list(cls, discrete: bool, items: list[list]) -> IntervalSet:
        return cls(discrete, tuple((i[0], i[1], i[2], i[3]) for i in items))
