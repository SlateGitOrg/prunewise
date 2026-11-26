from prunewise.intervals import IntervalSet


def test_points_and_membership():
    s = IntervalSet.points(True, [3, 9, 5])
    assert s.intervals == ((3, True, 3, True), (5, True, 5, True), (9, True, 9, True))
    assert s.contains(5) and not s.contains(4)
    assert s.values() == [3, 5, 9]


def test_discrete_adjacent_points_merge_into_a_range():
    s = IntervalSet.points(True, [4, 5, 6, 7])
    assert s.intervals == ((4, True, 7, True),)
    assert s.values() == [4, 5, 6, 7]


def test_discrete_exclusive_bounds_become_inclusive():
    s = IntervalSet.range(True, lo=5, lo_inc=False, hi=10, hi_inc=False)
    assert s.intervals == ((6, True, 9, True),)


def test_continuous_exclusive_bounds_are_kept():
    s = IntervalSet.range(False, lo="b", lo_inc=False, hi="d", hi_inc=True)
    assert not s.contains("b") and s.contains("c") and s.contains("d")
    assert s.intersects("a", "b") is False
    assert s.intersects("a", "bb") is True


def test_intersection_and_union():
    a = IntervalSet.range(True, 0, True, 10, True)
    b = IntervalSet.range(True, 5, True, 20, True)
    assert a.intersect(b).intervals == ((5, True, 10, True),)
    assert a.union(IntervalSet.points(True, [30])).intervals == (
        (0, True, 10, True),
        (30, True, 30, True),
    )
    assert a.intersect(IntervalSet.points(True, [50])).is_empty


def test_zone_map_intersection_at_boundaries():
    s = IntervalSet.range(True, lo=10, hi=20)
    assert s.intersects(0, 10) and s.intersects(20, 30)
    assert not s.intersects(0, 9) and not s.intersects(21, 30)
    assert s.intersects(None, None)  # no statistics: cannot skip


def test_unbounded_and_values_limit():
    s = IntervalSet.range(True, lo=5)
    assert s.values() is None
    assert IntervalSet.full(True).intersect(s).intervals == s.intervals
    assert IntervalSet.range(True, 0, True, 10**9, True).values(limit=10) is None
