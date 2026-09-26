"""Unit tests for the separation domain logic."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.separation import (
    DEFAULT_MATRIX,
    Capacity,
    ClosureWindow,
    ExclusiveGroup,
    Flight,
    best_adjacent_swap,
    build_matrix,
    delay_stats,
    find_capacity_violations,
    find_closure_conflicts,
    find_violations,
    joint_trial_schedule,
    required_gap,
    sort_flights,
    sort_windows,
    trial_reschedule,
)

UTC = timezone.utc


def dt(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def make(no: str, t: int, wake: str, locked: bool = False) -> Flight:
    return Flight(flight_no=no, scheduled_off=dt(t), wake=wake, locked=locked)


def window(start: int, end: int) -> ClosureWindow:
    return ClosureWindow(start=dt(start), end=dt(end))


def test_stable_sort_by_time_then_flight_no():
    flights = [
        make("BBB", 200, "M"),
        make("AAA", 100, "H"),
        make("CCC", 100, "L"),  # same time as AAA -> flight_no tiebreak
    ]
    assert [f.flight_no for f in sort_flights(flights)] == ["AAA", "CCC", "BBB"]
    # input order unchanged (immutable objects, new list)
    assert flights[0].flight_no == "BBB"


def test_default_matrix_contents():
    matrix = build_matrix(None)
    assert matrix[("J", "M")] == DEFAULT_MATRIX["J"]["M"] == 180
    assert required_gap(matrix, "H", "M") == 120


def test_custom_matrix_unknown_pair_is_zero():
    matrix = build_matrix({("H", "M"): 90})
    assert required_gap(matrix, "H", "M") == 90
    assert required_gap(matrix, "J", "L") == 0


def test_find_violations_flags_only_adjacent_breaches():
    # H leader (100-> next M needs 120s); M->M needs 60s
    flights = [
        make("CA101", 0, "H"),
        make("CA102", 60, "M"),    # gap 60 < 120 -> violation
        make("CA103", 130, "M"),   # gap 70 >= 60 -> ok
    ]
    violations = find_violations(flights, build_matrix(None))
    assert len(violations) == 1
    v = violations[0]
    assert (v.leader_no, v.follower_no) == ("CA101", "CA102")
    assert v.required_seconds == 120
    assert v.actual_seconds == 60
    assert v.shortfall_seconds == 60
    assert (v.leader_position, v.follower_position) == (1, 2)


def test_no_violation_when_exact_gap_met():
    flights = [make("CA101", 0, "H"), make("CA102", 120, "M")]
    assert find_violations(flights, build_matrix(None)) == []


def test_trial_cascades_delays_without_mutating_originals():
    # J(0) -> M needs 180; M -> M needs 60
    flights = [
        make("CA1", 0, "J"),
        make("CA2", 60, "M"),    # pushed to 180 (triggered by CA1), delay 120
        make("CA3", 200, "M"),   # must be >= 240 (triggered by CA2), delay 40
    ]
    items = trial_reschedule(flights, build_matrix(None)).items
    assert [i.flight_no for i in items] == ["CA1", "CA2", "CA3"]
    assert items[0].delay_seconds == 0
    assert items[0].triggered_by is None
    assert items[1].adjusted_time == dt(180)
    assert items[1].delay_seconds == 120
    assert items[1].triggered_by == "CA1"
    assert items[2].adjusted_time == dt(240)
    assert items[2].delay_seconds == 40
    assert items[2].triggered_by == "CA2"

    # original schedules are untouched
    assert flights[1].scheduled_off == dt(60)
    assert flights[2].scheduled_off == dt(200)


def test_trial_never_moves_flight_earlier():
    flights = [make("CA1", 1000, "M"), make("CA2", 5000, "M")]
    items = trial_reschedule(flights, build_matrix(None)).items
    assert items[1].delay_seconds == 0
    assert items[1].adjusted_time == dt(5000)


def test_trial_with_custom_matrix():
    matrix = build_matrix({("H", "H"): 300})
    flights = [make("CA1", 0, "H"), make("CA2", 100, "H")]
    items = trial_reschedule(flights, matrix).items
    assert items[1].adjusted_time == dt(300)
    assert items[1].delay_seconds == 200
    assert items[1].triggered_by == "CA1"


def test_single_flight_has_no_violation_and_no_delay():
    flights = [make("SOLO", 42, "J")]
    assert find_violations(flights, build_matrix(None)) == []
    (item,) = trial_reschedule(flights, build_matrix(None)).items
    assert item.delay_seconds == 0 and item.triggered_by is None


# ---------- closure windows ----------


def test_sort_windows_orders_by_start():
    wins = sort_windows([window(300, 400), window(0, 100), window(100, 200)])
    assert [(w.s, w.e) for w in wins] == [(0, 100), (100, 200), (300, 400)]


def test_closure_conflict_window_is_half_open():
    flights = [
        make("AT_START", 100, "M"),  # start is inclusive -> conflict
        make("AT_END", 200, "M"),    # end is exclusive -> no conflict
        make("BEFORE", 99, "M"),
        make("AFTER", 201, "M"),
    ]
    conflicts = find_closure_conflicts(flights, [window(100, 200)])
    assert [c.flight_no for c in conflicts] == ["AT_START"]
    c = conflicts[0]
    assert c.position == 2  # sorted: BEFORE(99), AT_START(100), ...
    assert c.window_index == 1
    assert c.window_start == dt(100) and c.window_end == dt(200)


def test_closure_conflicts_sorted_positions_and_window_index():
    flights = [
        make("LATE", 500, "H"),   # unsorted input on purpose
        make("EARLY", 50, "M"),
        make("MID", 350, "L"),
    ]
    wins = [window(300, 400), window(0, 100)]  # unsorted input on purpose
    conflicts = find_closure_conflicts(flights, wins)
    assert [(c.flight_no, c.position, c.window_index) for c in conflicts] == [
        ("EARLY", 1, 1),   # window [0,100) is index 1 after sorting
        ("MID", 2, 2),     # window [300,400) is index 2
    ]


def test_trial_first_flight_pushed_out_of_window():
    flights = [make("CA1", 30, "M"), make("CA2", 200, "M")]
    items = trial_reschedule(flights, build_matrix(None), [window(0, 90)]).items
    assert items[0].adjusted_time == dt(90)
    assert items[0].delay_seconds == 60
    assert items[0].triggered_by is None
    assert items[0].closure_hit is True
    assert items[0].closure_window == window(0, 90)
    # follower unaffected: 200 >= 90 + 60
    assert items[1].delay_seconds == 0
    assert items[1].closure_hit is False
    assert items[1].closure_window is None


def test_trial_wake_push_into_window_iterates_to_window_end():
    # J->M requires 180s: wake pushes CA2 from 60 to 180, which lands inside
    # the [150, 210) window, so it must move on to 210.
    flights = [make("CA1", 0, "J"), make("CA2", 60, "M")]
    items = trial_reschedule(flights, build_matrix(None), [window(150, 210)]).items
    assert items[1].adjusted_time == dt(210)
    assert items[1].delay_seconds == 150
    assert items[1].triggered_by == "CA1"
    assert items[1].closure_hit is True
    assert items[1].closure_window == window(150, 210)


def test_trial_iterates_through_touching_windows():
    # window end == next window start: landing on the boundary is still closed
    flights = [make("CA1", 50, "M")]
    items = trial_reschedule(
        flights, build_matrix(None), [window(0, 100), window(100, 160)]
    ).items
    assert items[0].adjusted_time == dt(160)
    assert items[0].closure_hit is True
    assert items[0].closure_window == window(100, 160)


def test_trial_closure_delay_cascades_to_successors():
    # CA2 is closure-pushed to 210; CA3 must cascade to 210 + 60 (M->M)
    flights = [
        make("CA1", 0, "J"),
        make("CA2", 60, "M"),
        make("CA3", 220, "M"),
    ]
    items = trial_reschedule(flights, build_matrix(None), [window(150, 210)]).items
    assert items[1].adjusted_time == dt(210)
    assert items[2].adjusted_time == dt(270)
    assert items[2].delay_seconds == 50
    assert items[2].triggered_by == "CA2"
    assert items[2].closure_hit is False


def test_trial_wake_push_past_window_is_not_closure_hit():
    # wake earliest (300) already clears the [100, 200) window entirely
    flights = [make("CA1", 0, "H"), make("CA2", 150, "H")]
    matrix = build_matrix({("H", "H"): 300})
    items = trial_reschedule(flights, matrix, [window(100, 200)]).items
    assert items[1].adjusted_time == dt(300)
    assert items[1].triggered_by == "CA1"
    assert items[1].closure_hit is False
    assert items[1].closure_window is None


def test_trial_without_windows_keeps_legacy_flags():
    flights = [make("CA1", 0, "M"), make("CA2", 10, "M")]
    items = trial_reschedule(flights, build_matrix(None)).items
    assert all(i.closure_hit is False and i.closure_window is None for i in items)


# ---------- rolling capacity constraint ----------


def test_capacity_none_yields_no_violations_and_legacy_trial_fields():
    flights = [make("CA1", 0, "M"), make("CA2", 10, "M")]
    assert find_capacity_violations(flights, None) == []
    items = trial_reschedule(flights, build_matrix(None), capacity=None).items
    assert all(
        i.capacity_hit is False
        and i.capacity_anchor is None
        and i.capacity_window_start is None
        and i.capacity_window_seconds is None
        for i in items
    )


def test_capacity_violation_half_open_boundary():
    # max 2 departures per 60 s. [0,60) may hold F1,F2; F3 at 59 overflows
    # (anchor F1). F4 at 60 clears F1's window (end excluded), but its own
    # k-th predecessor is F2@30: [30,90) still holds F2,F3,F4 -> overflows.
    flights = [make("F1", 0, "M"), make("F2", 30, "M"),
              make("F3", 59, "M"), make("F4", 60, "M")]
    violations = find_capacity_violations(flights, Capacity(60, 2))
    assert [(v.flight_no, v.anchor_flight_no) for v in violations] == [
        ("F3", "F1"),
        ("F4", "F2"),
    ]
    v = violations[0]
    assert v.position == 3
    assert v.window_seconds == 60 and v.max_departures == 2
    assert v.departures_in_window == 3
    assert v.anchor_position == 1
    assert v.anchor_off == dt(0)
    assert v.window_end == dt(60)
    assert v.earliest_compliant_off == dt(60)
    v2 = violations[1]
    assert v2.anchor_off == dt(30)
    assert v2.window_end == dt(90)
    assert v2.earliest_compliant_off == dt(90)
    assert v2.departures_in_window == 3


def test_capacity_violations_follow_stable_order_with_ties():
    # three flights at the same instant with k=1: 2nd and 3rd overflow;
    # stable (flight_no) order decides the anchors.
    flights = [make("B", 100, "M"), make("A", 100, "M"), make("C", 100, "M")]
    flights = sort_flights(flights)
    violations = find_capacity_violations(flights, Capacity(60, 1))
    assert [v.flight_no for v in violations] == ["B", "C"]
    assert violations[0].anchor_flight_no == "A"
    assert violations[0].earliest_compliant_off == dt(160)
    assert violations[1].anchor_flight_no == "B"
    assert violations[0].departures_in_window == 2


def test_capacity_exactly_filled_is_compliant():
    flights = [make("F1", 0, "M"), make("F2", 60, "M"), make("F3", 120, "M")]
    assert find_capacity_violations(flights, Capacity(60, 1)) == []


def test_capacity_sliding_window_catches_non_adjacent_overflow():
    # spacing 40 s, window 90 s, k=2: any window may hold at most 2 flights.
    # F0..F2 span 80 s -> F2 overflows (anchor F0); F1..F3 span 80 s -> F3
    # overflows (anchor F1) — the window slides, it is not anchored to the
    # first flight.
    flights = [make(f"F{i}", 40 * i, "M") for i in range(4)]
    violations = find_capacity_violations(flights, Capacity(90, 2))
    assert [(v.flight_no, v.anchor_flight_no) for v in violations] == [
        ("F2", "F0"),
        ("F3", "F1"),
    ]
    assert violations[1].anchor_off == dt(40)
    assert violations[1].window_end == dt(130)
    assert violations[0].departures_in_window == 3
    assert violations[1].departures_in_window == 3


def test_trial_capacity_pushes_overflowing_flight_to_window_end():
    flights = [make("F1", 0, "M"), make("F2", 30, "M"), make("F3", 59, "M")]
    matrix = {("M", "M"): 0}
    items = trial_reschedule(flights, matrix, capacity=Capacity(60, 2)).items
    assert items[0].delay_seconds == 0 and items[0].capacity_hit is False
    assert items[1].delay_seconds == 0 and items[1].capacity_hit is False
    # F3 pushed to anchor(F1).off + 60 = 60
    assert items[2].adjusted_time == dt(60)
    assert items[2].delay_seconds == 1
    assert items[2].capacity_hit is True
    assert items[2].capacity_anchor == "F1"
    assert items[2].capacity_window_seconds == 60
    assert items[2].capacity_max_departures == 2
    assert items[2].capacity_window_start == dt(0)
    assert items[2].capacity_window_end == dt(60)
    # capacity does not masquerade as a wake trigger
    assert items[2].triggered_by is None


def test_trial_capacity_delay_cascades_through_successors():
    # k=1 (one departure per 60 s): F2 30 -> 60, F3 70 -> 120
    flights = [make("F1", 0, "M"), make("F2", 30, "M"), make("F3", 70, "M")]
    matrix = {("M", "M"): 0}
    items = trial_reschedule(flights, matrix, capacity=Capacity(60, 1)).items
    assert items[1].adjusted_time == dt(60)
    assert items[2].adjusted_time == dt(120)
    assert items[2].delay_seconds == 50
    assert items[2].capacity_hit is True
    assert items[2].capacity_anchor == "F2"


def test_trial_capacity_push_into_closure_iterates_back_to_capacity():
    # F3's capacity push lands on the closure [60,90) -> moved to 90;
    # F4 at 90 (exactly 60 s after F2) then: closure excludes 90? end excluded
    # so [60,90) does not contain 90; F4 anchor F2@30 earliest 90, OK.
    flights = [make("F1", 0, "M"), make("F2", 30, "M"),
              make("F3", 59, "M"), make("F4", 90, "M")]
    matrix = {("M", "M"): 0}
    items = trial_reschedule(
        flights, matrix, [window(60, 90)], Capacity(60, 2)
    ).items
    assert items[2].adjusted_time == dt(90)
    assert items[2].capacity_hit is True
    assert items[2].closure_hit is True
    assert items[2].closure_window == window(60, 90)
    assert items[3].adjusted_time == dt(90)  # ties allowed: [30,90) excludes F4
    # final schedule satisfies the rolling limit
    adjusted = [i.adjusted_time for i in items]
    for idx in range(2, len(adjusted)):
        assert (adjusted[idx] - adjusted[idx - 2]).total_seconds() >= 60


def test_trial_closure_push_then_capacity_push():
    # F2 scheduled inside the closure is pushed to its end (90). F3's wake
    # earliest follows F2 to 90, jumping over the window, so F3 itself never
    # lands inside the closure; capacity (anchor F1@0, earliest 60) also holds.
    flights = [make("F1", 0, "M"), make("F2", 40, "M"), make("F3", 60, "M")]
    matrix = {("M", "M"): 0}
    items = trial_reschedule(
        flights, matrix, [window(30, 90)], Capacity(60, 2)
    ).items
    assert items[1].adjusted_time == dt(90)
    assert items[1].closure_hit is True
    assert items[2].adjusted_time == dt(90)
    assert items[2].closure_hit is False
    assert items[2].triggered_by == "F2"
    adjusted = [i.adjusted_time for i in items]
    assert (adjusted[2] - adjusted[0]).total_seconds() >= 60


def test_trial_closure_boundary_requires_capacity_further_push():
    # Capacity pushes F3 to 60, which is exactly a closure start -> the
    # closure moves it on to 100; re-checked capacity (anchor F1@0 needs 60)
    # already holds at 100.
    flights = [make("F1", 0, "M"), make("F2", 30, "M"), make("F3", 59, "M")]
    matrix = {("M", "M"): 0}
    items = trial_reschedule(
        flights, matrix, [window(60, 100)], Capacity(60, 2)
    ).items
    assert items[2].adjusted_time == dt(100)
    assert items[2].capacity_hit is True
    assert items[2].closure_hit is True
    assert items[2].capacity_anchor == "F1"
    assert items[2].closure_window == window(60, 100)


# ---------- immovable (locked) flights ----------


def test_locked_feasible_flight_keeps_scheduled_time_and_is_marked():
    # J@0 -> M needs 180, but the follower is locked at 200 (>= 180): kept.
    flights = [make("CA1", 0, "J"), make("CA2", 200, "M", locked=True)]
    result = trial_reschedule(flights, build_matrix(None))
    assert result.feasible is True and result.blocked is None
    assert len(result.items) == 2
    locked_item = result.items[1]
    assert locked_item.locked is True
    assert locked_item.adjusted_time == dt(200)
    assert locked_item.delay_seconds == 0
    assert result.items[0].locked is False


def test_locked_flight_blocked_by_wake_returns_prefix_and_required_time():
    # J@0 -> M needs 180; locked follower at 60 cannot move -> blocked.
    flights = [make("CA1", 0, "J"), make("CA2", 60, "M", locked=True),
              make("CA3", 300, "M")]
    result = trial_reschedule(flights, build_matrix(None))
    assert result.feasible is False
    # feasible prefix only (CA1); blocked flight and successors get no items
    assert [i.flight_no for i in result.items] == ["CA1"]
    b = result.blocked
    assert b is not None
    assert b.flight_no == "CA2"
    assert b.position == 2
    assert b.constraint == "wake"
    assert b.required_earliest_off == dt(180)
    assert b.triggered_by == "CA1"
    assert b.required_gap_seconds == 180
    assert b.closure_hit is False and b.closure_window is None
    assert b.capacity_hit is False and b.capacity_anchor is None


def test_locked_flight_blocked_by_closure_window():
    # locked single flight scheduled inside the closure: must not be moved.
    flights = [make("CA1", 30, "M", locked=True)]
    result = trial_reschedule(flights, build_matrix(None), [window(0, 90)])
    assert result.feasible is False
    assert result.items == []
    b = result.blocked
    assert b.flight_no == "CA1" and b.position == 1
    assert b.constraint == "closure"
    assert b.required_earliest_off == dt(90)
    assert b.triggered_by is None and b.required_gap_seconds is None
    assert b.closure_hit is True
    assert b.closure_window == window(0, 90)


def test_locked_flight_blocked_by_capacity():
    # 3 flights in a 60 s window limited to 2; the third is locked at 59.
    flights = [make("F1", 0, "M"), make("F2", 30, "M"),
              make("F3", 59, "M", locked=True)]
    matrix = {("M", "M"): 0}
    result = trial_reschedule(flights, matrix, capacity=Capacity(60, 2))
    assert result.feasible is False
    assert [i.flight_no for i in result.items] == ["F1", "F2"]
    b = result.blocked
    assert b.flight_no == "F3" and b.position == 3
    assert b.constraint == "capacity"
    assert b.required_earliest_off == dt(60)
    assert b.capacity_hit is True
    assert b.capacity_anchor == "F1"
    assert b.capacity_window_seconds == 60
    assert b.capacity_max_departures == 2
    assert b.capacity_window_start == dt(0)
    assert b.capacity_window_end == dt(60)
    assert b.closure_hit is False


def test_locked_flight_after_pushed_successor_is_checked_against_prefix():
    # CA2 is pushed to 180; locked CA3 at 200 needs M->M 60 after 180 (240),
    # so the locked flight blocks the runway at CA3, not at CA2.
    flights = [make("CA1", 0, "J"), make("CA2", 60, "M"),
              make("CA3", 200, "M", locked=True)]
    result = trial_reschedule(flights, build_matrix(None))
    assert result.feasible is False
    assert result.items[1].adjusted_time == dt(180)  # CA2 pushed by CA1
    b = result.blocked
    assert b.flight_no == "CA3" and b.position == 3
    assert b.constraint == "wake"
    assert b.required_earliest_off == dt(240)
    assert b.triggered_by == "CA2"
    assert b.required_gap_seconds == 60


def test_locked_flight_after_pushed_successor_kept_when_gap_allows():
    # CA2 is pushed to 180; locked CA3 at 240 clears the 60 s gap -> kept.
    flights = [make("CA1", 0, "J"), make("CA2", 60, "M"),
              make("CA3", 240, "M", locked=True)]
    result = trial_reschedule(flights, build_matrix(None))
    assert result.feasible is True
    assert result.items[1].adjusted_time == dt(180)  # CA2 pushed by CA1
    locked_item = result.items[2]
    assert locked_item.locked is True
    assert locked_item.adjusted_time == dt(240)
    assert locked_item.delay_seconds == 0


def test_first_infeasible_locked_flight_stops_later_locked_checks():
    # CA2 locked is infeasible (wake); CA3 would also be moot and gets no item.
    flights = [make("CA1", 0, "J"),
              make("CA2", 60, "M", locked=True),
              make("CA3", 100, "M", locked=True)]
    result = trial_reschedule(flights, build_matrix(None))
    assert result.feasible is False
    assert result.blocked.flight_no == "CA2"
    assert [i.flight_no for i in result.items] == ["CA1"]


def test_unlocked_flights_still_only_move_later_when_locks_present():
    # an earlier locked flight never pulls a follower forward
    flights = [make("CA1", 0, "M", locked=True), make("CA2", 1000, "M")]
    result = trial_reschedule(flights, build_matrix(None))
    assert result.feasible is True
    assert result.items[0].adjusted_time == dt(0)
    assert result.items[1].adjusted_time == dt(1000)
    assert result.items[1].delay_seconds == 0


# ---------- best adjacent swap ----------


def test_best_adjacent_swap_picks_improving_swap():
    # J->M needs 180 s but M->J only 60 s: swapping the pair saves 20 s.
    flights = [make("J1", 0, "J"), make("M1", 50, "M")]
    result = best_adjacent_swap(flights, build_matrix(None))
    assert [f.flight_no for f in result.baseline_order] == ["J1", "M1"]
    assert delay_stats(result.baseline) == (130, 130)
    chosen = result.chosen_swap
    assert chosen is not None
    assert chosen.position == 1
    assert chosen.first_flight_no == "J1"
    assert chosen.second_flight_no == "M1"
    assert [f.flight_no for f in result.optimized_order] == ["M1", "J1"]
    assert delay_stats(result.optimized) == (110, 110)
    assert result.saved_seconds == 20
    # scheduled_off values are never rewritten
    assert flights[0].scheduled_off == dt(0)
    assert flights[1].scheduled_off == dt(50)


def test_best_adjacent_swap_keeps_baseline_on_tie():
    # the swap yields the same (total, max) -> "no swap" wins the tie
    flights = [make("J1", 0, "J"), make("M1", 60, "M")]
    result = best_adjacent_swap(flights, build_matrix(None))
    assert result.chosen_swap is None
    assert result.optimized_order == result.baseline_order
    assert result.optimized is result.baseline
    assert result.saved_seconds == 0


def test_best_adjacent_swap_single_flight_has_no_candidates():
    result = best_adjacent_swap([make("SOLO", 42, "J")], build_matrix(None))
    assert result.chosen_swap is None
    assert delay_stats(result.optimized) == (0, 0)
    assert result.saved_seconds == 0


def test_best_adjacent_swap_max_delay_tiebreak():
    # baseline and swap(1,2) tie on total delay (220 s); the swap wins on a
    # lower max single-flight delay (120 < 140), so saved_seconds is 0.
    flights = [make("J1", 0, "J"), make("M1", 40, "M"), make("M2", 160, "M")]
    result = best_adjacent_swap(flights, build_matrix(None))
    assert delay_stats(result.baseline) == (220, 140)
    chosen = result.chosen_swap
    assert chosen is not None
    assert chosen.position == 1
    assert (chosen.first_flight_no, chosen.second_flight_no) == ("J1", "M1")
    assert delay_stats(result.optimized) == (220, 120)
    assert result.saved_seconds == 0


def test_best_adjacent_swap_prefers_smallest_position_on_tie():
    # two independent identical pairs: swapping either pair saves the same
    # (240, 130); the lowest swap position wins.
    flights = [
        make("J1", 0, "J"),
        make("M1", 50, "M"),
        make("J2", 1000, "J"),
        make("M2", 1050, "M"),
    ]
    result = best_adjacent_swap(flights, build_matrix(None))
    assert delay_stats(result.baseline) == (260, 130)
    chosen = result.chosen_swap
    assert chosen is not None
    assert chosen.position == 1
    assert (chosen.first_flight_no, chosen.second_flight_no) == ("J1", "M1")
    assert delay_stats(result.optimized) == (240, 130)
    assert result.saved_seconds == 20


def test_best_adjacent_swap_respects_closures_and_capacity():
    # every candidate is rescheduled under the same closure/capacity rules;
    # here the swap is worse, so the baseline order is kept.
    flights = [make("J1", 0, "J"), make("M1", 50, "M")]
    result = best_adjacent_swap(
        flights, build_matrix(None), [window(110, 200)], Capacity(60, 1)
    )
    # baseline: M1 wake-pushed to 180 -> inside [110,200) -> 200 (delay 150)
    assert delay_stats(result.baseline) == (150, 150)
    # swapped: J1 wake-pushed to 110 -> closure -> 200 (delay 200): worse
    assert result.chosen_swap is None
    assert result.optimized.items[1].adjusted_time == dt(200)
    assert result.optimized.items[1].closure_hit is True
    assert result.saved_seconds == 0


# ---------- joint trial (mutually exclusive runway groups) ----------


def group(index: int, runways: list[str], gap: int) -> ExclusiveGroup:
    return ExclusiveGroup(
        index=index, runways=frozenset(runways), min_gap_seconds=gap
    )


def test_joint_global_stable_order_merges_runways():
    # flights of every runway are merged into one (time, flight_no) order;
    # positions refer to that global order, wake applies only per runway
    flights_by_runway = {
        "36L": [make("CA2", 50, "M"), make("CA1", 0, "J")],
        "18R": [make("MU2", 400, "M"), make("MU1", 50, "M")],  # tie with CA2
    }
    result = joint_trial_schedule(
        flights_by_runway, build_matrix(None), {}, {}, []
    )
    # CA2 sorts before MU1 on the flight_no tiebreak at t=50
    assert [i.flight_no for i in result.items] == ["CA1", "CA2", "MU1", "MU2"]
    assert [i.position for i in result.items] == [1, 2, 3, 4]
    assert [i.runway for i in result.items] == ["36L", "36L", "18R", "18R"]
    by_no = {i.flight_no: i for i in result.items}
    # CA2 wake-pushed behind CA1 (J->M 180); MU1 leads its own runway
    assert by_no["CA2"].adjusted_time == dt(180)
    assert by_no["CA2"].triggered_by == "CA1"
    assert by_no["CA2"].constraint == "wake"
    assert by_no["MU1"].delay_seconds == 0
    assert by_no["MU1"].triggered_by is None
    assert result.feasible and result.blocked is None


def test_joint_group_gap_pushes_cross_runway_departure():
    flights_by_runway = {
        "36L": [make("CA1", 0, "M")],
        "18R": [make("MU1", 60, "M")],
    }
    result = joint_trial_schedule(
        flights_by_runway,
        build_matrix(None),
        {},
        {},
        [group(1, ["36L", "18R"], 90)],
    )
    by_no = {i.flight_no: i for i in result.items}
    mu1 = by_no["MU1"]
    # 60 < 0 + 90 -> pushed to the end of CA1's occupancy span
    assert mu1.adjusted_time == dt(90)
    assert mu1.delay_seconds == 30
    assert mu1.constraint == "group"
    assert mu1.group_hit is True
    assert mu1.group_index == 1
    assert mu1.group_min_gap_seconds == 90
    assert mu1.group_triggered_by == "CA1"
    assert mu1.group_triggered_runway == "36L"
    assert mu1.group_triggered_off == dt(0)
    assert mu1.group_required_off == dt(90)
    assert mu1.group_window_start == dt(0)
    assert mu1.group_window_end == dt(90)
    assert by_no["CA1"].group_hit is False


def test_joint_group_gap_span_is_half_open():
    # exactly min_gap after the other departure is compliant; one second
    # less is not
    base = {"36L": [make("CA1", 0, "M")]}
    ok = joint_trial_schedule(
        {**base, "18R": [make("MU1", 90, "M")]},
        build_matrix(None), {}, {}, [group(1, ["36L", "18R"], 90)],
    )
    assert ok.items[1].delay_seconds == 0
    tight = joint_trial_schedule(
        {**base, "18R": [make("MU1", 89, "M")]},
        build_matrix(None), {}, {}, [group(1, ["36L", "18R"], 90)],
    )
    assert tight.items[1].adjusted_time == dt(90)
    assert tight.items[1].delay_seconds == 1


def test_joint_group_gap_against_later_same_group_departure():
    # CA2 is wake-pushed to 180; MU1 (scheduled 100) would sit only 80 s
    # before it -> pushed to 180 + 90, the *later* departure's span end
    flights_by_runway = {
        "36L": [make("CA1", 0, "J"), make("CA2", 50, "M")],
        "18R": [make("MU1", 100, "M")],
    }
    result = joint_trial_schedule(
        flights_by_runway,
        build_matrix(None),
        {},
        {},
        [group(1, ["36L", "18R"], 90)],
    )
    by_no = {i.flight_no: i for i in result.items}
    assert by_no["CA2"].adjusted_time == dt(180)
    mu1 = by_no["MU1"]
    assert mu1.adjusted_time == dt(270)
    assert mu1.group_triggered_by == "CA2"
    assert mu1.group_triggered_off == dt(180)
    assert mu1.group_window_end == dt(270)


def test_joint_group_gap_applies_within_same_runway_too():
    # two departures on one member runway are also "two departures in the
    # group": with a zero wake matrix the group gap still separates them
    flights_by_runway = {
        "36L": [make("CA1", 0, "M"), make("CA2", 100, "M")],
        "18R": [make("MU1", 500, "M")],
    }
    matrix = build_matrix({("M", "M"): 0})
    result = joint_trial_schedule(
        flights_by_runway, matrix, {}, {}, [group(1, ["36L", "18R"], 120)]
    )
    by_no = {i.flight_no: i for i in result.items}
    assert by_no["CA2"].adjusted_time == dt(120)
    assert by_no["CA2"].group_triggered_by == "CA1"
    assert by_no["CA2"].group_triggered_runway == "36L"


def test_joint_group_push_into_closure_iterates_until_converged():
    # group push lands MU1 on 90, inside its own runway's [90, 200) closure;
    # the closure then pushes it to 200 and every constraint is re-checked
    flights_by_runway = {
        "36L": [make("CA1", 0, "M")],
        "18R": [make("MU1", 30, "M")],
    }
    result = joint_trial_schedule(
        flights_by_runway,
        build_matrix(None),
        {"18R": [window(90, 200)]},
        {},
        [group(1, ["36L", "18R"], 90)],
    )
    mu1 = result.items[1]
    assert mu1.adjusted_time == dt(200)
    assert mu1.group_hit is True
    assert mu1.closure_hit is True
    assert mu1.closure_window == window(90, 200)
    # closure outranks group when naming the first pushing constraint
    assert mu1.constraint == "closure"


def test_joint_capacity_push_into_group_span_iterates():
    # CA3 is capacity-pushed from 150 to 180 (anchor CA1 + 180 s), but 180 is
    # only 60 s ahead of MU1's adjusted 120; the group constraint then pushes
    # it to 210 and capacity is re-checked against the new candidate.
    flights_by_runway = {
        "36L": [make("CA1", 0, "M"), make("CA3", 150, "M")],
        "18R": [make("MU1", 120, "M")],
    }
    matrix = build_matrix({("M", "M"): 0})
    result = joint_trial_schedule(
        flights_by_runway,
        matrix,
        {},
        {"36L": Capacity(180, 1)},
        [group(1, ["36L", "18R"], 90)],
    )
    by_no = {i.flight_no: i for i in result.items}
    # MU1 stays: 120 is exactly 120 s after CA1, outside the 90 s band
    assert by_no["MU1"].adjusted_time == dt(120)
    ca3 = by_no["CA3"]
    assert ca3.adjusted_time == dt(210)
    assert ca3.delay_seconds == 60
    assert ca3.capacity_hit is True
    assert ca3.capacity_anchor == "CA1"
    assert ca3.group_hit is True
    assert ca3.group_triggered_by == "MU1"
    assert ca3.group_window_end == dt(210)
    assert ca3.constraint == "capacity"  # capacity outranks group


def test_joint_multiple_groups_are_independent():
    flights_by_runway = {
        "01": [make("A1", 0, "M")],
        "02": [make("B1", 30, "M")],
        "03": [make("C1", 40, "M")],
        "04": [make("D1", 45, "M")],
    }
    matrix = build_matrix({("M", "M"): 0})
    result = joint_trial_schedule(
        flights_by_runway,
        matrix,
        {},
        {},
        [group(1, ["01", "02"], 90), group(2, ["03", "04"], 30)],
    )
    by_no = {i.flight_no: i for i in result.items}
    # B1 pushed by group 1 (A1 + 90); C1/D1 belong to group 2 only
    assert by_no["B1"].adjusted_time == dt(90)
    assert by_no["B1"].group_index == 1
    assert by_no["C1"].delay_seconds == 0
    assert by_no["D1"].adjusted_time == dt(70)  # C1 + 30
    assert by_no["D1"].group_index == 2


def test_joint_without_groups_matches_independent_trials():
    flights_by_runway = {
        "36L": [make("CA1", 0, "J"), make("CA2", 60, "M"), make("CA3", 200, "M")],
        "18R": [make("MU1", 30, "H"), make("MU2", 100, "L")],
    }
    matrix = build_matrix(None)
    windows = {"36L": [window(150, 210)]}
    capacities = {"18R": Capacity(50, 1)}
    result = joint_trial_schedule(
        flights_by_runway, matrix, windows, capacities, []
    )
    for runway, flights in flights_by_runway.items():
        solo = trial_reschedule(
            flights, matrix, windows.get(runway), capacities.get(runway)
        )
        joint = [i for i in result.items if i.runway == runway]
        assert [i.flight_no for i in joint] == [i.flight_no for i in solo.items]
        assert [i.adjusted_time for i in joint] == [
            i.adjusted_time for i in solo.items
        ]


def test_joint_locked_flight_blocked_by_group_stops_everything():
    flights_by_runway = {
        "36L": [make("CA1", 0, "J")],
        "18R": [make("MU1", 60, "M", locked=True), make("MU2", 500, "M")],
    }
    result = joint_trial_schedule(
        flights_by_runway,
        build_matrix(None),
        {},
        {},
        [group(1, ["36L", "18R"], 90)],
    )
    assert not result.feasible
    # feasible prefix only: MU2 (later in the global order) gets no proposal
    assert [i.flight_no for i in result.items] == ["CA1"]
    blocked = result.blocked
    assert blocked.flight_no == "MU1"
    assert blocked.runway == "18R"
    assert blocked.position == 2
    assert blocked.constraint == "group"
    assert blocked.required_earliest_off == dt(90)
    assert blocked.group_required_off == dt(90)
    assert blocked.wake_required_off is None
    assert blocked.closure_required_off is None
    assert blocked.capacity_required_off is None
    assert blocked.group_hit is True
    assert blocked.group_index == 1
    assert blocked.group_min_gap_seconds == 90
    assert blocked.group_triggered_by == "CA1"
    assert blocked.group_triggered_runway == "36L"
    assert blocked.group_triggered_off == dt(0)
    assert blocked.group_window_start == dt(0)
    assert blocked.group_window_end == dt(90)


def test_joint_locked_blocker_reports_every_constraint_required_time():
    # wake pushes the locked CA2 to 180, inside closure [150, 300) -> 300;
    # the blocker names wake first but reports each constraint's instant
    flights_by_runway = {
        "36L": [make("CA1", 0, "J"), make("CA2", 60, "M", locked=True)],
    }
    result = joint_trial_schedule(
        flights_by_runway, build_matrix(None), {"36L": [window(150, 300)]}, {}, []
    )
    blocked = result.blocked
    assert blocked.constraint == "wake"
    assert blocked.triggered_by == "CA1"
    assert blocked.required_gap_seconds == 180
    assert blocked.wake_required_off == dt(180)
    assert blocked.closure_required_off == dt(300)
    assert blocked.required_earliest_off == dt(300)
    assert blocked.closure_hit is True
    assert blocked.closure_window == window(150, 300)


def test_joint_locked_flight_feasible_keeps_scheduled_off():
    flights_by_runway = {
        "36L": [make("CA1", 0, "J")],
        "18R": [make("MU1", 300, "M", locked=True)],
    }
    result = joint_trial_schedule(
        flights_by_runway,
        build_matrix(None),
        {},
        {},
        [group(1, ["36L", "18R"], 90)],
    )
    assert result.feasible
    mu1 = result.items[1]
    assert mu1.locked is True
    assert mu1.adjusted_time == dt(300)
    assert mu1.delay_seconds == 0
