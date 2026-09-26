"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from .separation import DEFAULT_MATRIX, Wake


class FlightIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flight_no: str = Field(..., min_length=1, examples=["CA1234"])
    scheduled_off: datetime = Field(
        ..., description="计划离场时刻，ISO 8601，必须带时区，例如 2026-09-18T08:00:00+08:00"
    )
    wake: Wake = Field(..., description="尾流等级: L 轻型 / M 中型 / H 重型 / J 超重型")
    locked: bool = Field(
        default=False,
        strict=True,
        description="是否为不可移动航班：试排中必须保持 scheduled_off，"
        "缺省为 false。validate 仍按原计划校验并回显锁定状态。",
    )

    @field_validator("flight_no")
    @classmethod
    def _strip_flight_no(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("flight_no 不能为空")
        return v

    @field_validator("scheduled_off")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("scheduled_off 必须携带时区信息（ISO 8601 offset 或 Z）")
        return v


class ClosureWindowIn(BaseModel):
    """A temporary runway closure window ``[start, end)`` (end excluded)."""

    model_config = ConfigDict(extra="forbid")

    start: datetime = Field(
        ..., description="关闭开始时刻，ISO 8601，必须带时区，例如 2026-09-18T08:10:00+08:00"
    )
    end: datetime = Field(
        ..., description="关闭结束时刻（不含），ISO 8601，必须带时区"
    )

    @field_validator("start", "end")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("关闭窗口时刻必须携带时区信息（ISO 8601 offset 或 Z）")
        return v

    @model_validator(mode="after")
    def _end_after_start(self) -> ClosureWindowIn:
        if self.end <= self.start:
            raise ValueError("关闭窗口结束时刻必须晚于开始时刻（[start, end) 左闭右开）")
        return self


class CapacityIn(BaseModel):
    """A per-runway rolling departure-capacity limit.

    In any half-open span of ``window_seconds`` seconds, at most
    ``max_departures`` flights may depart.
    """

    model_config = ConfigDict(extra="forbid")

    window_seconds: int = Field(
        ...,
        ge=1,
        strict=True,
        description="滚动窗口长度（秒），正整数",
    )
    max_departures: int = Field(
        ...,
        ge=1,
        strict=True,
        description="窗口内最多离场架次，正整数",
    )


class RunwayIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runway: str = Field(..., min_length=1, examples=["36L"])
    flights: list[FlightIn] = Field(..., min_length=1)
    closures: list[ClosureWindowIn] = Field(
        default_factory=list,
        description="本跑道的临时关闭窗口列表，[start, end) 左闭右开，彼此不得重叠。",
    )
    capacity: CapacityIn | None = Field(
        default=None,
        description="本跑道可选的滚动离场容量约束：任意 window_seconds 秒的半开区间内"
        "离场不得超过 max_departures 架；缺省表示不施加容量约束。",
    )

    @field_validator("runway")
    @classmethod
    def _strip_runway(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("runway 不能为空")
        return v

    @model_validator(mode="after")
    def _validate_runway(self) -> RunwayIn:
        seen: set[str] = set()
        for f in self.flights:
            key = f.flight_no
            if key in seen:
                raise ValueError(f"跑道 {self.runway} 内航班号重复: {key}")
            seen.add(key)
        # closures are half-open [start, end): touching is allowed, overlap is not
        ordered = sorted(self.closures, key=lambda w: (w.start, w.end))
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start < prev.end:
                raise ValueError(
                    f"跑道 {self.runway} 的关闭窗口重叠: "
                    f"[{prev.start.isoformat()}, {prev.end.isoformat()}) 与 "
                    f"[{nxt.start.isoformat()}, {nxt.end.isoformat()})"
                )
        return self


class MatrixEntryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leader: Wake
    follower: Wake
    seconds: int = Field(..., ge=0, description="前机—后机最小雷达间隔（秒）")


class WakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runways: list[RunwayIn] = Field(..., min_length=1)
    matrix: list[MatrixEntryIn] | None = Field(
        default=None,
        description="前机—后机最小间隔矩阵；缺省使用内置 ICAO 风格默认矩阵。"
        "提供时仅列出的组合生效，未列出的组合视为无最小间隔(0 秒)。",
    )

    @model_validator(mode="after")
    def _validate_request(self) -> WakeRequest:
        runway_names: set[str] = set()
        global_flights: set[str] = set()
        for rw in self.runways:
            if rw.runway in runway_names:
                raise ValueError(f"跑道标识重复: {rw.runway}")
            runway_names.add(rw.runway)
            for f in rw.flights:
                if f.flight_no in global_flights:
                    raise ValueError(f"航班号在整个请求中必须唯一: {f.flight_no}")
                global_flights.add(f.flight_no)
        if self.matrix is not None:
            pairs: set[tuple[str, str]] = set()
            for entry in self.matrix:
                pair = (entry.leader.value, entry.follower.value)
                if pair in pairs:
                    raise ValueError(
                        f"间隔矩阵组合重复: leader={pair[0]} follower={pair[1]}"
                    )
                pairs.add(pair)
        return self


class ExclusiveGroupIn(BaseModel):
    """A mutually exclusive runway group (e.g. intersecting runways): any two
    departures on the member runways must be at least ``min_gap_seconds``
    apart."""

    model_config = ConfigDict(extra="forbid")

    runways: list[str] = Field(
        ...,
        min_length=2,
        description="互斥跑道标识列表（至少两条，须在 runways 中定义；"
        "同组内不得重复，且一条跑道最多属于一个组）",
    )
    min_gap_seconds: int = Field(
        ...,
        ge=1,
        strict=True,
        description="组内任意两条跑道的两次离场之间的最小占用间隔（秒），正整数",
    )

    @field_validator("runways")
    @classmethod
    def _strip_and_dedupe_runways(cls, v: list[str]) -> list[str]:
        cleaned = [name.strip() for name in v]
        if any(not name for name in cleaned):
            raise ValueError("互斥组的跑道标识不能为空")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("互斥组内跑道标识重复")
        return cleaned


class JointTrialRequest(WakeRequest):
    """Joint-trial request: the usual runway plan plus mutually exclusive
    runway groups with per-group minimum occupancy gaps."""

    exclusive_groups: list[ExclusiveGroupIn] = Field(
        default_factory=list,
        description="互斥跑道组列表；缺省或空列表表示无组间占用间隔约束。",
    )

    @model_validator(mode="after")
    def _validate_groups(self) -> JointTrialRequest:
        defined = {rw.runway for rw in self.runways}
        grouped: dict[str, int] = {}
        for index, group in enumerate(self.exclusive_groups, start=1):
            for name in group.runways:
                if name not in defined:
                    raise ValueError(
                        f"互斥组 {index} 引用了未定义的跑道: {name}"
                    )
                if name in grouped:
                    raise ValueError(
                        f"跑道 {name} 同时出现在互斥组 {grouped[name]} 与 {index}，"
                        "一条跑道最多属于一个互斥组"
                    )
                grouped[name] = index
        return self


# ---------- response models ----------


class ClosureWindowOut(BaseModel):
    start: datetime
    end: datetime


class CapacityOut(BaseModel):
    window_seconds: int
    max_departures: int


class CapacityViolationOut(BaseModel):
    flight_no: str
    wake: Wake
    scheduled_off: datetime
    position: int
    window_seconds: int
    max_departures: int
    departures_in_window: int
    anchor_flight_no: str
    anchor_position: int
    window_start: datetime
    window_end: datetime
    earliest_compliant_off: datetime


class ClosureConflictOut(BaseModel):
    flight_no: str
    wake: Wake
    scheduled_off: datetime
    position: int
    window_index: int
    window_start: datetime
    window_end: datetime


class ViolationOut(BaseModel):
    leader: str
    follower: str
    leader_wake: Wake
    follower_wake: Wake
    leader_scheduled_off: datetime
    follower_scheduled_off: datetime
    required_seconds: int
    actual_seconds: int
    shortfall_seconds: int
    leader_position: int
    follower_position: int


class LockedFlightOut(BaseModel):
    position: int
    flight_no: str
    scheduled_off: datetime
    wake: Wake


class TrialBlockedOut(BaseModel):
    """First locked flight the trial cannot keep on its scheduled time."""

    flight_no: str
    wake: Wake
    scheduled_off: datetime
    position: int
    constraint: Literal["wake", "closure", "capacity"]
    required_earliest_off: datetime
    # wake detail
    triggered_by: str | None
    required_gap_seconds: int | None
    # closure detail
    closure_hit: bool
    closure_window: ClosureWindowOut | None
    # capacity detail (present only when the runway carried a capacity config)
    has_capacity: bool = Field(default=False, exclude=True)
    capacity_hit: bool = False
    capacity_window_seconds: int | None = None
    capacity_max_departures: int | None = None
    capacity_anchor: str | None = None
    capacity_window: ClosureWindowOut | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data = {
            "flight_no": self.flight_no,
            "wake": self.wake,
            "scheduled_off": self.scheduled_off,
            "position": self.position,
            "constraint": self.constraint,
            "required_earliest_off": self.required_earliest_off,
            "triggered_by": self.triggered_by,
            "required_gap_seconds": self.required_gap_seconds,
            "closure_hit": self.closure_hit,
            "closure_window": (
                self.closure_window.model_dump()
                if self.closure_window is not None
                else None
            ),
        }
        if self.has_capacity:
            data["capacity_hit"] = self.capacity_hit
            data["capacity_window_seconds"] = self.capacity_window_seconds
            data["capacity_max_departures"] = self.capacity_max_departures
            data["capacity_anchor"] = self.capacity_anchor
            data["capacity_window"] = (
                self.capacity_window.model_dump()
                if self.capacity_window is not None
                else None
            )
        return data


class RunwayReportOut(BaseModel):
    runway: str
    order: list[str]
    total: int
    closures: list[ClosureWindowOut]
    violation_count: int
    violations: list[ViolationOut]
    closure_conflict_count: int
    closure_conflicts: list[ClosureConflictOut]
    # Capacity fields are emitted only when the runway carried a capacity
    # config, so legacy request/response shapes stay byte-compatible.
    has_capacity: bool = Field(default=False, exclude=True)
    capacity: CapacityOut | None = None
    capacity_violation_count: int = 0
    capacity_violations: list[CapacityViolationOut] = Field(default_factory=list)
    # Lock detail is emitted only when the request explicitly carried the
    # ``locked`` field on at least one flight of this runway.
    has_locks: bool = Field(default=False, exclude=True)
    locked_flights: list[LockedFlightOut] = Field(default_factory=list)

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "runway": self.runway,
            "order": self.order,
            "total": self.total,
            "closures": [w.model_dump() for w in self.closures],
            "violation_count": self.violation_count,
            "violations": [v.model_dump(mode="json") for v in self.violations],
            "closure_conflict_count": self.closure_conflict_count,
            "closure_conflicts": [
                c.model_dump(mode="json") for c in self.closure_conflicts
            ],
        }
        if self.has_capacity:
            data["capacity"] = (
                self.capacity.model_dump() if self.capacity is not None else None
            )
            data["capacity_violation_count"] = self.capacity_violation_count
            data["capacity_violations"] = [
                v.model_dump(mode="json") for v in self.capacity_violations
            ]
        if self.has_locks:
            data["locked_flights"] = [
                f.model_dump(mode="json") for f in self.locked_flights
            ]
        return data


class ValidateResponse(BaseModel):
    batch_id: str
    mode: str = "validate"
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_violations: int
    total_closure_conflicts: int
    runways: list[RunwayReportOut]
    # Same conditional rule as the per-runway capacity block: the summary is
    # present only when at least one runway in the batch had a capacity config.
    has_capacity: bool = Field(default=False, exclude=True)
    total_capacity_violations: int = 0
    # Present only when at least one flight in the batch explicitly carried
    # the ``locked`` field.
    has_locks: bool = Field(default=False, exclude=True)
    total_locked: int = 0

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "batch_id": self.batch_id,
            "mode": self.mode,
            "matrix": self.matrix,
            "total_runways": self.total_runways,
            "total_flights": self.total_flights,
            "total_violations": self.total_violations,
            "total_closure_conflicts": self.total_closure_conflicts,
            "runways": [rw.model_dump(mode="json") for rw in self.runways],
        }
        if self.has_capacity:
            data["total_capacity_violations"] = self.total_capacity_violations
        if self.has_locks:
            data["total_locked"] = self.total_locked
        return data


class TrialItemOut(BaseModel):
    position: int
    flight_no: str
    wake: Wake
    scheduled_off: datetime
    adjusted_off: datetime
    delay_seconds: int
    triggered_by: str | None
    closure_hit: bool
    closure_window: ClosureWindowOut | None
    # Echoed only when the request explicitly carried the ``locked`` field.
    has_locks: bool = Field(default=False, exclude=True)
    locked: bool = False
    # Capacity detail is present only when the runway carried a capacity config.
    has_capacity: bool = Field(default=False, exclude=True)
    capacity_hit: bool = False
    capacity_window_seconds: int | None = None
    capacity_max_departures: int | None = None
    capacity_anchor: str | None = None
    capacity_window: ClosureWindowOut | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "position": self.position,
            "flight_no": self.flight_no,
            "wake": self.wake,
            "scheduled_off": self.scheduled_off,
            "adjusted_off": self.adjusted_off,
            "delay_seconds": self.delay_seconds,
            "triggered_by": self.triggered_by,
            "closure_hit": self.closure_hit,
            "closure_window": (
                self.closure_window.model_dump()
                if self.closure_window is not None
                else None
            ),
        }
        if self.has_capacity:
            data["capacity_hit"] = self.capacity_hit
            data["capacity_window_seconds"] = self.capacity_window_seconds
            data["capacity_max_departures"] = self.capacity_max_departures
            data["capacity_anchor"] = self.capacity_anchor
            data["capacity_window"] = (
                self.capacity_window.model_dump()
                if self.capacity_window is not None
                else None
            )
        if self.has_locks:
            data["locked"] = self.locked
        return data


class RunwayTrialOut(BaseModel):
    runway: str
    order: list[str]
    total: int
    closures: list[ClosureWindowOut]
    delayed_count: int
    total_delay_seconds: int
    max_delay_seconds: int
    items: list[TrialItemOut]
    has_capacity: bool = Field(default=False, exclude=True)
    capacity: CapacityOut | None = None
    has_locks: bool = Field(default=False, exclude=True)
    locked_flights: list[LockedFlightOut] = Field(default_factory=list)
    # Set only when the first locked flight that cannot keep its scheduled
    # time was reached; ``items`` then holds just the feasible prefix.
    blocked: TrialBlockedOut | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "runway": self.runway,
            "order": self.order,
            "total": self.total,
            "closures": [w.model_dump() for w in self.closures],
            "delayed_count": self.delayed_count,
            "total_delay_seconds": self.total_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "items": [i.model_dump(mode="json") for i in self.items],
        }
        if self.has_capacity:
            data["capacity"] = (
                self.capacity.model_dump() if self.capacity is not None else None
            )
        if self.has_locks:
            data["locked_flights"] = [
                f.model_dump(mode="json") for f in self.locked_flights
            ]
            data["blocked"] = (
                self.blocked.model_dump(mode="json")
                if self.blocked is not None
                else None
            )
        return data


class TrialResponse(BaseModel):
    batch_id: str
    mode: str = "trial"
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_delayed: int
    total_delay_seconds: int
    runways: list[RunwayTrialOut]
    has_locks: bool = Field(default=False, exclude=True)
    total_locked: int = 0

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "batch_id": self.batch_id,
            "mode": self.mode,
            "matrix": self.matrix,
            "total_runways": self.total_runways,
            "total_flights": self.total_flights,
            "total_delayed": self.total_delayed,
            "total_delay_seconds": self.total_delay_seconds,
            "runways": [rw.model_dump(mode="json") for rw in self.runways],
        }
        if self.has_locks:
            data["total_locked"] = self.total_locked
        return data


class SwapChoiceOut(BaseModel):
    """The chosen adjacent swap; positions refer to the baseline order."""

    position: int
    first_flight_no: str
    second_flight_no: str


class BaselineMetricsOut(BaseModel):
    """Delay metrics of the baseline (stable-order) schedule."""

    delayed_count: int
    total_delay_seconds: int
    max_delay_seconds: int


class RunwaySwapOut(BaseModel):
    """One runway's best-adjacent-swap optimisation result."""

    runway: str
    order: list[str]  # optimised departure order; item positions refer to it
    baseline_order: list[str]  # stable order the enumeration started from
    total: int
    closures: list[ClosureWindowOut]
    baseline: BaselineMetricsOut
    chosen_swap: SwapChoiceOut | None  # None -> baseline order already best
    saved_seconds: int
    delayed_count: int
    total_delay_seconds: int
    max_delay_seconds: int
    items: list[TrialItemOut]
    has_capacity: bool = Field(default=False, exclude=True)
    capacity: CapacityOut | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "runway": self.runway,
            "order": self.order,
            "baseline_order": self.baseline_order,
            "total": self.total,
            "closures": [w.model_dump() for w in self.closures],
            "baseline": self.baseline.model_dump(mode="json"),
            "chosen_swap": (
                self.chosen_swap.model_dump()
                if self.chosen_swap is not None
                else None
            ),
            "saved_seconds": self.saved_seconds,
            "delayed_count": self.delayed_count,
            "total_delay_seconds": self.total_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "items": [i.model_dump(mode="json") for i in self.items],
        }
        if self.has_capacity:
            data["capacity"] = (
                self.capacity.model_dump() if self.capacity is not None else None
            )
        return data


class BestAdjacentSwapResponse(BaseModel):
    batch_id: str
    mode: str = "best_adjacent_swap"
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_baseline_delay_seconds: int
    total_delay_seconds: int
    total_saved_seconds: int
    runways: list[RunwaySwapOut]


class ExclusiveGroupOut(BaseModel):
    """Echo of one mutually exclusive runway group (1-based index)."""

    index: int
    runways: list[str]
    min_gap_seconds: int


class JointTrialItemOut(BaseModel):
    """One flight's joint-trial proposal; positions follow the global order."""

    position: int
    flight_no: str
    wake: Wake
    runway: str
    scheduled_off: datetime
    adjusted_off: datetime
    delay_seconds: int
    constraint: Literal["wake", "closure", "capacity", "group"] | None
    triggered_by: str | None
    closure_hit: bool
    closure_window: ClosureWindowOut | None
    # Echoed only when the request explicitly carried the ``locked`` field.
    has_locks: bool = Field(default=False, exclude=True)
    locked: bool = False
    # Capacity detail is present only when the flight's runway carried a
    # capacity config.
    has_capacity: bool = Field(default=False, exclude=True)
    capacity_hit: bool = False
    capacity_window_seconds: int | None = None
    capacity_max_departures: int | None = None
    capacity_anchor: str | None = None
    capacity_window: ClosureWindowOut | None = None
    # Exclusive-group detail (all None when no group pushed this flight).
    group_hit: bool
    group_index: int | None
    group_min_gap_seconds: int | None
    group_triggered_by: str | None
    group_triggered_runway: str | None
    group_triggered_off: datetime | None
    group_required_off: datetime | None
    group_window: ClosureWindowOut | None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "position": self.position,
            "flight_no": self.flight_no,
            "wake": self.wake,
            "runway": self.runway,
            "scheduled_off": self.scheduled_off,
            "adjusted_off": self.adjusted_off,
            "delay_seconds": self.delay_seconds,
            "constraint": self.constraint,
            "triggered_by": self.triggered_by,
            "closure_hit": self.closure_hit,
            "closure_window": (
                self.closure_window.model_dump()
                if self.closure_window is not None
                else None
            ),
            "group_hit": self.group_hit,
            "group_index": self.group_index,
            "group_min_gap_seconds": self.group_min_gap_seconds,
            "group_triggered_by": self.group_triggered_by,
            "group_triggered_runway": self.group_triggered_runway,
            "group_triggered_off": self.group_triggered_off,
            "group_required_off": self.group_required_off,
            "group_window": (
                self.group_window.model_dump()
                if self.group_window is not None
                else None
            ),
        }
        if self.has_capacity:
            data["capacity_hit"] = self.capacity_hit
            data["capacity_window_seconds"] = self.capacity_window_seconds
            data["capacity_max_departures"] = self.capacity_max_departures
            data["capacity_anchor"] = self.capacity_anchor
            data["capacity_window"] = (
                self.capacity_window.model_dump()
                if self.capacity_window is not None
                else None
            )
        if self.has_locks:
            data["locked"] = self.locked
        return data


class JointBlockedOut(BaseModel):
    """First locked flight the joint trial cannot keep on its scheduled time."""

    flight_no: str
    wake: Wake
    runway: str
    scheduled_off: datetime
    position: int  # slot in the global stable order
    constraint: Literal["wake", "closure", "capacity", "group"]
    required_earliest_off: datetime
    # per-constraint required instants (None when that constraint did not push)
    wake_required_off: datetime | None
    closure_required_off: datetime | None
    capacity_required_off: datetime | None
    group_required_off: datetime | None
    # wake detail
    triggered_by: str | None
    required_gap_seconds: int | None
    # closure detail
    closure_hit: bool
    closure_window: ClosureWindowOut | None
    # capacity detail (present only when the runway carried a capacity config)
    has_capacity: bool = Field(default=False, exclude=True)
    capacity_hit: bool = False
    capacity_window_seconds: int | None = None
    capacity_max_departures: int | None = None
    capacity_anchor: str | None = None
    capacity_window: ClosureWindowOut | None = None
    # exclusive-group detail
    group_hit: bool
    group_index: int | None
    group_min_gap_seconds: int | None
    group_triggered_by: str | None
    group_triggered_runway: str | None
    group_triggered_off: datetime | None
    group_window: ClosureWindowOut | None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data = {
            "flight_no": self.flight_no,
            "wake": self.wake,
            "runway": self.runway,
            "scheduled_off": self.scheduled_off,
            "position": self.position,
            "constraint": self.constraint,
            "required_earliest_off": self.required_earliest_off,
            "wake_required_off": self.wake_required_off,
            "closure_required_off": self.closure_required_off,
            "capacity_required_off": self.capacity_required_off,
            "group_required_off": self.group_required_off,
            "triggered_by": self.triggered_by,
            "required_gap_seconds": self.required_gap_seconds,
            "closure_hit": self.closure_hit,
            "closure_window": (
                self.closure_window.model_dump()
                if self.closure_window is not None
                else None
            ),
            "group_hit": self.group_hit,
            "group_index": self.group_index,
            "group_min_gap_seconds": self.group_min_gap_seconds,
            "group_triggered_by": self.group_triggered_by,
            "group_triggered_runway": self.group_triggered_runway,
            "group_triggered_off": self.group_triggered_off,
            "group_window": (
                self.group_window.model_dump()
                if self.group_window is not None
                else None
            ),
        }
        if self.has_capacity:
            data["capacity_hit"] = self.capacity_hit
            data["capacity_window_seconds"] = self.capacity_window_seconds
            data["capacity_max_departures"] = self.capacity_max_departures
            data["capacity_anchor"] = self.capacity_anchor
            data["capacity_window"] = (
                self.capacity_window.model_dump()
                if self.capacity_window is not None
                else None
            )
        return data


class JointRunwayOut(BaseModel):
    """Per-runway summary inside a joint-trial response."""

    runway: str
    order: list[str]  # this runway's flights in the global stable order
    total: int
    closures: list[ClosureWindowOut]
    has_capacity: bool = Field(default=False, exclude=True)
    capacity: CapacityOut | None = None
    has_locks: bool = Field(default=False, exclude=True)
    locked_flights: list[LockedFlightOut] = Field(default_factory=list)

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "runway": self.runway,
            "order": self.order,
            "total": self.total,
            "closures": [w.model_dump() for w in self.closures],
        }
        if self.has_capacity:
            data["capacity"] = (
                self.capacity.model_dump() if self.capacity is not None else None
            )
        if self.has_locks:
            data["locked_flights"] = [
                f.model_dump(mode="json") for f in self.locked_flights
            ]
        return data


class JointTrialResponse(BaseModel):
    batch_id: str
    mode: str = "joint_trial"
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_delayed: int
    total_delay_seconds: int
    max_delay_seconds: int
    order: list[str]  # global stable order of every flight
    exclusive_groups: list[ExclusiveGroupOut]
    items: list[JointTrialItemOut]
    blocked: JointBlockedOut | None
    runways: list[JointRunwayOut]
    has_locks: bool = Field(default=False, exclude=True)
    total_locked: int = 0

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data: dict = {
            "batch_id": self.batch_id,
            "mode": self.mode,
            "matrix": self.matrix,
            "total_runways": self.total_runways,
            "total_flights": self.total_flights,
            "total_delayed": self.total_delayed,
            "total_delay_seconds": self.total_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "order": self.order,
            "exclusive_groups": [g.model_dump() for g in self.exclusive_groups],
            "items": [i.model_dump(mode="json") for i in self.items],
            "blocked": (
                self.blocked.model_dump(mode="json")
                if self.blocked is not None
                else None
            ),
            "runways": [rw.model_dump(mode="json") for rw in self.runways],
        }
        if self.has_locks:
            data["total_locked"] = self.total_locked
        return data


class MatrixOut(BaseModel):
    matrix: dict[str, dict[str, int]]


class HealthOut(BaseModel):
    status: str


class BatchResponse(BaseModel):
    """Stored batch (either mode) fetched by id; summary fields are mode-specific."""

    batch_id: str
    mode: str
    created_at: datetime
    matrix: dict[str, dict[str, int]]
    total_runways: int
    total_flights: int
    total_violations: int | None = None
    total_closure_conflicts: int | None = None
    total_capacity_violations: int | None = None
    total_delayed: int | None = None
    total_delay_seconds: int | None = None
    total_locked: int | None = None
    total_baseline_delay_seconds: int | None = None
    total_saved_seconds: int | None = None
    max_delay_seconds: int | None = None
    order: list[str] | None = None
    exclusive_groups: list[dict] | None = None
    items: list[dict] | None = None
    blocked: dict | None = None
    runways: list[dict]

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        data = {
            "batch_id": self.batch_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "matrix": self.matrix,
            "total_runways": self.total_runways,
            "total_flights": self.total_flights,
            "runways": self.runways,
        }
        for name in (
            "total_violations",
            "total_closure_conflicts",
            "total_capacity_violations",
            "total_delayed",
            "total_delay_seconds",
            "total_locked",
            "total_baseline_delay_seconds",
            "total_saved_seconds",
            "max_delay_seconds",
            "order",
            "exclusive_groups",
            "items",
            "blocked",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.mode == "joint_trial":
            # joint-trial batches always carry the mode-defining keys, even
            # when the schedule was feasible (blocked = null)
            data.setdefault("blocked", None)
        return data


def matrix_to_nested(flat: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    """Convert the internal flat matrix to a leader->follower nested dict."""
    nested: dict[str, dict[str, int]] = {}
    for (leader, follower), seconds in flat.items():
        nested.setdefault(leader, {})[follower] = seconds
    return nested


def default_matrix_nested() -> dict[str, dict[str, int]]:
    return {k: dict(v) for k, v in DEFAULT_MATRIX.items()}
