"""FastAPI application: runway departure wake-separation validation."""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .database import get_db, init_db
from .repository import load_batch, save_batch
from .schemas import (
    BaselineMetricsOut,
    BatchResponse,
    BestAdjacentSwapResponse,
    ClosureWindowOut,
    HealthOut,
    JointBlockedOut,
    JointRequiredTimesOut,
    JointTrialItemOut,
    JointTrialRequest,
    JointTrialResponse,
    LockedFlightOut,
    MatrixOut,
    MutexGroupOut,
    RequiredCapacityOut,
    RequiredClosureOut,
    RequiredMutexGroupOut,
    RequiredWakeOut,
    RunwayReportOut,
    RunwaySwapOut,
    RunwayTrialOut,
    RunwayJointOut,
    SwapChoiceOut,
    TrialBlockedOut,
    TrialItemOut,
    TrialResponse,
    ValidateResponse,
    WakeRequest,
    default_matrix_nested,
)
from .separation import ClosureWindow, delay_stats
from .service import (
    nested_matrix,
    run_best_adjacent_swap,
    run_joint_trial,
    run_trial,
    run_validate,
)
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="跑道离场尾流间隔校验 API",
    description=(
        "提交按跑道分组的航班计划（可附带跑道临时关闭窗口与滚动容量约束），"
        "校验相邻离场航班的尾流间隔违规、落入关闭窗口的航班与滚动容量超限，"
        "并支持在不改动原计划的前提下试排（向后级联延误，自动满足容量并"
        "避开关闭窗口）。航班可选 locked=true 标记为不可移动：试排中必须"
        "保持 scheduled_off，否则在跑道级返回首个不可行航班、冲突约束与"
        "所需最早时刻，仅持久化可行前缀。best-adjacent-swap 在试排基础上"
        "枚举每一对相邻航班的交换并择优（总延误 → 最大单航班延误 → 交换位置），"
        "给出基线指标、选中的交换与节省秒数（含 locked=true 的请求返回 422）。"
        "trial/joint 联合试排额外接受互斥跑道组：先按 (计划时刻, 航班号) "
        "确定跨跑道全局稳定顺序，再逐项求同时满足本跑道尾流/关闭窗口/滚动"
        "容量与组内最小占用间隔的最早时刻（任一约束推迟后全部约束重新检查"
        "直至收敛），锁定航班需要后移时返回首个阻塞航班及各约束所需时刻。"
    ),
    version="1.5.0",
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Pydantic 之外可能抛出的领域校验错误统一为 400。"""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "runway-wake-separation",
        "docs": "/docs",
        "health": "/health",
        "default_matrix": "/matrix/default",
        "endpoints": {
            "validate": "POST /api/v1/validate",
            "trial": "POST /api/v1/trial",
            "best_adjacent_swap": "POST /api/v1/trial/best-adjacent-swap",
            "joint_trial": "POST /api/v1/trial/joint",
            "batch": "GET /api/v1/batches/{batch_id}",
        },
    }


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    return HealthOut(status="ok")


@app.get("/matrix/default", response_model=MatrixOut, tags=["meta"])
def default_matrix() -> MatrixOut:
    return MatrixOut(matrix=default_matrix_nested())


def _window_out(window: ClosureWindow) -> dict:
    return {"start": window.start, "end": window.end}


def _capacity_out(capacity) -> dict:
    return {
        "window_seconds": capacity.window_seconds,
        "max_departures": capacity.max_departures,
    }


def _trial_item_out(item, *, has_capacity: bool, has_locks: bool = False) -> TrialItemOut:
    """Serialisable trial item; capacity/lock detail follows the runway flags."""
    return TrialItemOut(
        position=item.position,
        flight_no=item.flight_no,
        wake=item.wake,
        scheduled_off=item.original_time,
        adjusted_off=item.adjusted_time,
        delay_seconds=item.delay_seconds,
        triggered_by=item.triggered_by,
        closure_hit=item.closure_hit,
        closure_window=(
            ClosureWindowOut(
                start=item.closure_window.start, end=item.closure_window.end
            )
            if item.closure_window
            else None
        ),
        has_locks=has_locks,
        locked=item.locked,
        has_capacity=has_capacity,
        capacity_hit=item.capacity_hit,
        capacity_window_seconds=item.capacity_window_seconds,
        capacity_max_departures=item.capacity_max_departures,
        capacity_anchor=item.capacity_anchor,
        capacity_window=(
            ClosureWindowOut(
                start=item.capacity_window_start, end=item.capacity_window_end
            )
            if item.capacity_window_start is not None
            else None
        ),
    )


@app.post("/api/v1/validate", response_model=ValidateResponse, tags=["wake"])
def validate(
    req: WakeRequest, db: sqlite3.Connection = Depends(get_db)
) -> ValidateResponse:
    """按（计划时刻, 航班号）稳定排序，列出各跑道相邻航班的间隔违规、
    落入关闭窗口的航班及其命中窗口，以及滚动容量约束的超限航班、
    触发窗口与最早合规时刻。"""
    (
        payloads,
        matrix,
        windows,
        capacities,
        violations,
        conflicts,
        capacity_violations,
        locked_runways,
        locked_flights,
    ) = run_validate(req)
    batch_id = save_batch(
        db,
        mode="validate",
        matrix=matrix,
        runway_payloads=payloads,
        closures_by_runway=windows,
        capacities_by_runway=capacities,
        violations_by_runway=violations,
        closure_conflicts_by_runway=conflicts,
        capacity_violations_by_runway=capacity_violations,
        locked_runways=locked_runways,
    )

    runway_reports: list[RunwayReportOut] = []
    total_locked = 0
    for runway, flights in payloads.items():
        vlist = violations[runway]
        clist = conflicts[runway]
        capacity = capacities.get(runway)
        cvlist = capacity_violations[runway]
        runway_has_locks = runway in locked_runways
        runway_locked = locked_flights.get(runway, [])
        if runway_has_locks:
            total_locked += len(runway_locked)
        runway_reports.append(
            RunwayReportOut(
                runway=runway,
                order=[f.flight_no for f in flights],
                total=len(flights),
                closures=[_window_out(w) for w in windows[runway]],
                violation_count=len(vlist),
                violations=[
                    {
                        "leader": v.leader_no,
                        "follower": v.follower_no,
                        "leader_wake": v.leader_wake,
                        "follower_wake": v.follower_wake,
                        "leader_scheduled_off": v.leader_time,
                        "follower_scheduled_off": v.follower_time,
                        "required_seconds": v.required_seconds,
                        "actual_seconds": v.actual_seconds,
                        "shortfall_seconds": v.shortfall_seconds,
                        "leader_position": v.leader_position,
                        "follower_position": v.follower_position,
                    }
                    for v in vlist
                ],
                closure_conflict_count=len(clist),
                closure_conflicts=[
                    {
                        "flight_no": c.flight_no,
                        "wake": c.wake,
                        "scheduled_off": c.scheduled_off,
                        "position": c.position,
                        "window_index": c.window_index,
                        "window_start": c.window_start,
                        "window_end": c.window_end,
                    }
                    for c in clist
                ],
                has_capacity=capacity is not None,
                capacity=_capacity_out(capacity) if capacity is not None else None,
                capacity_violation_count=len(cvlist),
                capacity_violations=[
                    {
                        "flight_no": cv.flight_no,
                        "wake": cv.wake,
                        "scheduled_off": cv.scheduled_off,
                        "position": cv.position,
                        "window_seconds": cv.window_seconds,
                        "max_departures": cv.max_departures,
                        "departures_in_window": cv.departures_in_window,
                        "anchor_flight_no": cv.anchor_flight_no,
                        "anchor_position": cv.anchor_position,
                        "window_start": cv.anchor_off,
                        "window_end": cv.window_end,
                        "earliest_compliant_off": cv.earliest_compliant_off,
                    }
                    for cv in cvlist
                ],
                has_locks=runway_has_locks,
                locked_flights=[
                    LockedFlightOut(
                        position=pos,
                        flight_no=lf.flight_no,
                        scheduled_off=lf.scheduled_off,
                        wake=lf.wake,
                    )
                    for pos, lf in enumerate(flights, start=1)
                    if lf.flight_no in set(runway_locked)
                ],
            )
        )

    return ValidateResponse(
        batch_id=batch_id,
        matrix=nested_matrix(matrix),
        total_runways=len(payloads),
        total_flights=sum(len(f) for f in payloads.values()),
        total_violations=sum(len(v) for v in violations.values()),
        total_closure_conflicts=sum(len(c) for c in conflicts.values()),
        has_capacity=bool(capacities),
        total_capacity_violations=sum(
            len(cv) for cv in capacity_violations.values()
        ),
        has_locks=bool(locked_runways),
        total_locked=total_locked,
        runways=runway_reports,
    )


def _blocked_out(blocked, *, has_capacity: bool) -> TrialBlockedOut:
    capacity_window = None
    if blocked.capacity_window_start is not None:
        capacity_window = {
            "start": blocked.capacity_window_start,
            "end": blocked.capacity_window_end,
        }
    return TrialBlockedOut(
        flight_no=blocked.flight_no,
        wake=blocked.wake,
        scheduled_off=blocked.scheduled_off,
        position=blocked.position,
        constraint=blocked.constraint,
        required_earliest_off=blocked.required_earliest_off,
        triggered_by=blocked.triggered_by,
        required_gap_seconds=blocked.required_gap_seconds,
        closure_hit=blocked.closure_hit,
        closure_window=(
            _window_out(blocked.closure_window) if blocked.closure_window else None
        ),
        has_capacity=has_capacity,
        capacity_hit=blocked.capacity_hit,
        capacity_window_seconds=blocked.capacity_window_seconds,
        capacity_max_departures=blocked.capacity_max_departures,
        capacity_anchor=blocked.capacity_anchor,
        capacity_window=capacity_window,
    )


@app.post("/api/v1/trial", response_model=TrialResponse, tags=["wake"])
def trial(
    req: WakeRequest, db: sqlite3.Connection = Depends(get_db)
) -> TrialResponse:
    """保持原排序逐项推迟到最早安全时刻（满足尾流间隔、滚动容量约束且
    避开关闭窗口）；延误向后级联，原计划不变。

    锁定航班（locked=true）不可移动：必须保持 scheduled_off。当稳定顺序
    计算中尾流/关闭/容量约束会把首个锁定航班推迟时，该跑道在此停止，仅
    返回可行前缀与该航班（冲突约束、所需最早时刻），其后不再给出建议。
    """
    payloads, matrix, windows, capacities, trials, locked_runways, locked_flights = (
        run_trial(req)
    )
    batch_id = save_batch(
        db,
        mode="trial",
        matrix=matrix,
        runway_payloads=payloads,
        closures_by_runway=windows,
        capacities_by_runway=capacities,
        trial_by_runway=trials,
        locked_runways=locked_runways,
    )

    runway_trials: list[RunwayTrialOut] = []
    total_delayed = 0
    total_delay = 0
    total_locked = 0
    for runway, result in trials.items():
        items = result.items
        delayed = [i for i in items if i.delay_seconds > 0]
        rw_delay = sum(i.delay_seconds for i in items)
        total_delayed += len(delayed)
        total_delay += rw_delay
        capacity = capacities.get(runway)
        flights = payloads[runway]
        runway_has_locks = runway in locked_runways
        locked_no_set = set(locked_flights.get(runway, []))
        if runway_has_locks:
            total_locked += len(locked_no_set)
        runway_trials.append(
            RunwayTrialOut(
                runway=runway,
                order=[f.flight_no for f in flights],
                total=len(flights),
                closures=[_window_out(w) for w in windows[runway]],
                delayed_count=len(delayed),
                total_delay_seconds=rw_delay,
                max_delay_seconds=max((i.delay_seconds for i in items), default=0),
                has_capacity=capacity is not None,
                capacity=_capacity_out(capacity) if capacity is not None else None,
                items=[
                    _trial_item_out(
                        i,
                        has_capacity=capacity is not None,
                        has_locks=runway_has_locks,
                    )
                    for i in items
                ],
                has_locks=runway_has_locks,
                locked_flights=[
                    {
                        "position": pos,
                        "flight_no": f.flight_no,
                        "scheduled_off": f.scheduled_off,
                        "wake": f.wake,
                    }
                    for pos, f in enumerate(flights, start=1)
                    if f.flight_no in locked_no_set
                ],
                blocked=(
                    _blocked_out(result.blocked, has_capacity=capacity is not None)
                    if runway_has_locks and result.blocked is not None
                    else None
                ),
            )
        )

    return TrialResponse(
        batch_id=batch_id,
        matrix=nested_matrix(matrix),
        total_runways=len(trials),
        total_flights=sum(len(f) for f in payloads.values()),
        total_delayed=total_delayed,
        total_delay_seconds=total_delay,
        has_locks=bool(locked_runways),
        total_locked=total_locked,
        runways=runway_trials,
    )


@app.post(
    "/api/v1/trial/best-adjacent-swap",
    response_model=BestAdjacentSwapResponse,
    tags=["wake"],
)
def best_adjacent_swap_endpoint(
    req: WakeRequest, db: sqlite3.Connection = Depends(get_db)
) -> BestAdjacentSwapResponse:
    """以稳定排序为基线，枚举「不交换」与每一对相邻航班交换后的顺序，
    分别按现有尾流、关闭窗口与滚动容量规则从头试排（scheduled_off 不改），
    先最小化总延误秒数，再最小化最大单航班延误，仍并列时取交换位置最小者，
    不交换并列时优先不交换。各跑道独立求解。

    请求含任意 locked=true 的不可移动航班时拒绝（422）：交换顺序与
    不可移动语义冲突。
    """
    if any(f.locked for rw in req.runways for f in rw.flights):
        raise HTTPException(
            status_code=422,
            detail="best-adjacent-swap 不支持 locked=true 的不可移动航班",
        )
    payloads, matrix, windows, capacities, swaps = run_best_adjacent_swap(req)
    batch_id = save_batch(
        db,
        mode="best_adjacent_swap",
        matrix=matrix,
        runway_payloads={
            runway: result.optimized_order for runway, result in swaps.items()
        },
        closures_by_runway=windows,
        capacities_by_runway=capacities,
        trial_by_runway={runway: result.optimized for runway, result in swaps.items()},
        swap_by_runway=swaps,
    )

    runway_outs: list[RunwaySwapOut] = []
    total_baseline_delay = 0
    total_delay = 0
    for runway, result in swaps.items():
        capacity = capacities.get(runway)
        base_total, base_max = delay_stats(result.baseline)
        opt_total, opt_max = delay_stats(result.optimized)
        total_baseline_delay += base_total
        total_delay += opt_total
        chosen = result.chosen_swap
        runway_outs.append(
            RunwaySwapOut(
                runway=runway,
                order=[f.flight_no for f in result.optimized_order],
                baseline_order=[f.flight_no for f in result.baseline_order],
                total=len(result.optimized_order),
                closures=[_window_out(w) for w in windows[runway]],
                baseline=BaselineMetricsOut(
                    delayed_count=sum(
                        1 for i in result.baseline.items if i.delay_seconds > 0
                    ),
                    total_delay_seconds=base_total,
                    max_delay_seconds=base_max,
                ),
                chosen_swap=(
                    SwapChoiceOut(
                        position=chosen.position,
                        first_flight_no=chosen.first_flight_no,
                        second_flight_no=chosen.second_flight_no,
                    )
                    if chosen is not None
                    else None
                ),
                saved_seconds=result.saved_seconds,
                delayed_count=sum(
                    1 for i in result.optimized.items if i.delay_seconds > 0
                ),
                total_delay_seconds=opt_total,
                max_delay_seconds=opt_max,
                items=[
                    _trial_item_out(i, has_capacity=capacity is not None)
                    for i in result.optimized.items
                ],
                has_capacity=capacity is not None,
                capacity=_capacity_out(capacity) if capacity is not None else None,
            )
        )

    return BestAdjacentSwapResponse(
        batch_id=batch_id,
        matrix=nested_matrix(matrix),
        total_runways=len(swaps),
        total_flights=sum(len(f) for f in payloads.values()),
        total_baseline_delay_seconds=total_baseline_delay,
        total_delay_seconds=total_delay,
        total_saved_seconds=total_baseline_delay - total_delay,
        runways=runway_outs,
    )


def _closure_out(window) -> ClosureWindowOut | None:
    return ClosureWindowOut(start=window.start, end=window.end) if window else None


def _joint_item_out(item, *, has_capacity: bool, in_group: bool, has_locks: bool):
    return JointTrialItemOut(
        position=item.position,
        runway_position=item.runway_position,
        runway=item.runway,
        flight_no=item.flight_no,
        wake=item.wake,
        scheduled_off=item.original_time,
        adjusted_off=item.adjusted_time,
        delay_seconds=item.delay_seconds,
        constraint=item.constraint,
        triggered_by=item.triggered_by,
        closure_hit=item.closure_hit,
        closure_window=_closure_out(item.closure_window),
        has_locks=has_locks,
        locked=item.locked,
        has_capacity=has_capacity,
        capacity_hit=item.capacity_hit,
        capacity_window_seconds=item.capacity_window_seconds,
        capacity_max_departures=item.capacity_max_departures,
        capacity_anchor=item.capacity_anchor,
        capacity_window=(
            ClosureWindowOut(
                start=item.capacity_window_start, end=item.capacity_window_end
            )
            if item.capacity_window_start is not None
            else None
        ),
        in_group=in_group,
        mutex_group_hit=item.mutex_group_hit,
        mutex_group=item.mutex_group_name,
        mutex_group_anchor=item.mutex_group_anchor,
        mutex_group_min_interval_seconds=item.mutex_group_min_interval_seconds,
        mutex_group_window=(
            ClosureWindowOut(
                start=item.mutex_group_window_start,
                end=item.mutex_group_window_end,
            )
            if item.mutex_group_window_start is not None
            else None
        ),
    )


def _required_times_out(required: dict) -> JointRequiredTimesOut:
    def wake_payload(data):
        if data is None:
            return None
        return RequiredWakeOut(
            required_off=data["required_off"],
            triggered_by=data["triggered_by"],
            required_gap_seconds=data["required_gap_seconds"],
        )

    def closure_payload(data):
        if data is None:
            return None
        return RequiredClosureOut(
            required_off=data["required_off"],
            window=_closure_out(data["window"]),
        )

    def capacity_payload(data):
        if data is None:
            return None
        start, end = data["window"]
        return RequiredCapacityOut(
            required_off=data["required_off"],
            anchor_flight_no=data["anchor_flight_no"],
            window_seconds=data["window_seconds"],
            max_departures=data["max_departures"],
            window=ClosureWindowOut(start=start, end=end),
        )

    def group_payload(data):
        if data is None:
            return None
        start, end = data["window"]
        return RequiredMutexGroupOut(
            required_off=data["required_off"],
            triggered_by=data["triggered_by"],
            group=data["group"],
            min_interval_seconds=data["min_interval_seconds"],
            window=ClosureWindowOut(start=start, end=end),
        )

    return JointRequiredTimesOut(
        wake=wake_payload(required.get("wake")),
        closure=closure_payload(required.get("closure")),
        capacity=capacity_payload(required.get("capacity")),
        mutex_group=group_payload(required.get("mutex_group")),
    )


def _joint_blocked_out(blocked, *, has_capacity: bool, in_group: bool):
    return JointBlockedOut(
        position=blocked.position,
        runway_position=blocked.runway_position,
        runway=blocked.runway,
        flight_no=blocked.flight_no,
        wake=blocked.wake,
        scheduled_off=blocked.scheduled_off,
        constraint=blocked.constraint,
        required_earliest_off=blocked.required_earliest_off,
        triggered_by=blocked.triggered_by,
        required_gap_seconds=blocked.required_gap_seconds,
        closure_hit=blocked.closure_hit,
        closure_window=_closure_out(blocked.closure_window),
        required_times=_required_times_out(blocked.required_times),
        has_capacity=has_capacity,
        capacity_hit=blocked.capacity_hit,
        capacity_window_seconds=blocked.capacity_window_seconds,
        capacity_max_departures=blocked.capacity_max_departures,
        capacity_anchor=blocked.capacity_anchor,
        capacity_window=(
            ClosureWindowOut(
                start=blocked.capacity_window_start,
                end=blocked.capacity_window_end,
            )
            if blocked.capacity_window_start is not None
            else None
        ),
        in_group=in_group,
        mutex_group_hit=blocked.mutex_group_hit,
        mutex_group=blocked.mutex_group_name,
        mutex_group_anchor=blocked.mutex_group_anchor,
        mutex_group_min_interval_seconds=blocked.mutex_group_min_interval_seconds,
        mutex_group_window=(
            ClosureWindowOut(
                start=blocked.mutex_group_window_start,
                end=blocked.mutex_group_window_end,
            )
            if blocked.mutex_group_window_start is not None
            else None
        ),
    )


@app.post("/api/v1/trial/joint", response_model=JointTrialResponse, tags=["wake"])
def joint_trial_endpoint(
    req: JointTrialRequest, db: sqlite3.Connection = Depends(get_db)
) -> JointTrialResponse:
    """相交跑道联合试排。

    请求可定义若干互斥跑道组及组内两次离场的最小占用间隔。先按
    (计划时刻, 航班号) 确定跨跑道全局稳定顺序，再逐项求不早于原计划、
    同时满足本跑道尾流间隔、关闭窗口避让、滚动容量与互斥组间隔的最早
    时刻；任一约束把时刻推迟后，其余约束全部重新检查，循环直至收敛。

    锁定航班（locked=true）必须保持 scheduled_off：首个需要后移的锁定
    航班使整个联合试排停止——返回该阻塞航班、首个触发约束、最终最早合规
    时刻与各约束分别要求的时刻（required_times），该航班及其后不再生成
    建议，各跑道仅返回可行全局前缀。
    """
    (
        payloads,
        matrix,
        windows,
        capacities,
        groups,
        result,
        locked_runways,
        locked_flights,
    ) = run_joint_trial(req)

    batch_id = save_batch(
        db,
        mode="joint_trial",
        matrix=matrix,
        runway_payloads=payloads,
        closures_by_runway=windows,
        capacities_by_runway=capacities,
        joint_groups=groups,
        joint_result=result,
        locked_runways=locked_runways,
    )

    group_of_runway = {
        runway: group.name for group in groups for runway in group.runways
    }
    items_by_runway: dict[str, list] = {}
    for item in result.items:
        items_by_runway.setdefault(item.runway, []).append(item)

    runway_outs: list[RunwayJointOut] = []
    total_delayed = 0
    total_delay = 0
    max_delay = 0
    total_locked = 0
    for runway, flights in payloads.items():
        rw_items = items_by_runway.get(runway, [])
        delayed = [i for i in rw_items if i.delay_seconds > 0]
        rw_delay = sum(i.delay_seconds for i in rw_items)
        rw_max = max((i.delay_seconds for i in rw_items), default=0)
        total_delayed += len(delayed)
        total_delay += rw_delay
        max_delay = max(max_delay, rw_max)
        capacity = capacities.get(runway)
        in_group = runway in group_of_runway
        runway_has_locks = runway in locked_runways
        locked_no_set = set(locked_flights.get(runway, []))
        if runway_has_locks:
            total_locked += len(locked_no_set)
        blocked = result.blocked
        runway_outs.append(
            RunwayJointOut(
                runway=runway,
                total=len(flights),
                closures=[_window_out(w) for w in windows[runway]],
                delayed_count=len(delayed),
                total_delay_seconds=rw_delay,
                max_delay_seconds=rw_max,
                items=[
                    _joint_item_out(
                        i,
                        has_capacity=capacity is not None,
                        in_group=in_group,
                        has_locks=runway_has_locks,
                    )
                    for i in rw_items
                ],
                has_capacity=capacity is not None,
                capacity=_capacity_out(capacity) if capacity is not None else None,
                in_group=in_group,
                mutex_group=group_of_runway.get(runway),
                has_locks=runway_has_locks,
                locked_flights=[
                    LockedFlightOut(
                        position=pos,
                        flight_no=f.flight_no,
                        scheduled_off=f.scheduled_off,
                        wake=f.wake,
                    )
                    for pos, f in enumerate(flights, start=1)
                    if f.flight_no in locked_no_set
                ],
                blocked=(
                    _joint_blocked_out(
                        blocked,
                        has_capacity=capacity is not None,
                        in_group=in_group,
                    )
                    if runway_has_locks
                    and blocked is not None
                    and blocked.runway == runway
                    else None
                ),
            )
        )

    return JointTrialResponse(
        batch_id=batch_id,
        matrix=nested_matrix(matrix),
        total_runways=len(payloads),
        total_flights=sum(len(f) for f in payloads.values()),
        groups=[
            MutexGroupOut(
                name=group.name,
                runways=list(group.runways),
                min_interval_seconds=group.min_interval_seconds,
            )
            for group in groups
        ],
        order=[
            {"runway": runway, "flight_no": flight.flight_no}
            for runway, flight in result.order
        ],
        total_delayed=total_delayed,
        total_delay_seconds=total_delay,
        max_delay_seconds=max_delay,
        runways=runway_outs,
        has_locks=bool(locked_runways),
        total_locked=total_locked,
    )


@app.get(
    "/api/v1/batches/{batch_id}",
    response_model=BatchResponse,
    tags=["wake"],
)
def get_batch(
    batch_id: str, db: sqlite3.Connection = Depends(get_db)
) -> BatchResponse:
    """读取已保存的批次（validate / trial / best-adjacent-swap / joint_trial）。"""
    data = load_batch(db, batch_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"批次不存在: {batch_id}")
    return BatchResponse(**data)
