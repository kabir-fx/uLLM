"""Semantic checks on a telemetry trace, beyond what the schema can say.

The schema says each frame is well formed. This says the frames tell a
consistent story: ordered, gap-free, every request follows the life cycle in
catalog E4, steps only touch requests that can be in them, and token counts
in ``req.finished`` agree with the steps that produced them. The synthetic
fixtures must pass it, and real recordings will be checked with it from M2.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ullm.telemetry import events as ev

# Times are serialised to microseconds; allow for rounding on both sides.
_TOLERANCE_MS = 0.002

Problem = Callable[[str], None]


@dataclass
class _Req:
    arrived_t: float
    state: str = "waiting"  # waiting -> admitted -> running -> finished | aborted; preempted
    prefilled: bool = False
    had_first_token: bool = False
    sampled: int = 0  # tokens sampled for it: one per prefill step, one per decode step


@dataclass
class TraceReport:
    problems: list[str] = field(default_factory=list)
    requests: int = 0
    steps: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def check_trace(frames: Sequence[ev.Frame], *, complete: bool = True) -> TraceReport:
    """Check a whole trace. ``complete`` also requires every request to have ended."""
    report = TraceReport()
    problem = report.problems.append
    if not frames:
        problem("trace is empty")
        return report

    first = frames[0].events[0] if frames[0].events else None
    if not isinstance(first, ev.Hello):
        problem("the first event is not hello")
    hello = first if isinstance(first, ev.Hello) else None

    reqs: dict[str, _Req] = {}
    last_t = 0.0
    for i, frame in enumerate(frames):
        if frame.seq != frames[0].seq + i:
            problem(f"frame {i}: seq {frame.seq}, expected {frames[0].seq + i}")
        if frame.replica != frames[0].replica:
            problem(f"frame {i}: replica {frame.replica!r} differs from the first frame")
        for event in frame.events:
            where = f"frame {frame.seq} {event.type} t={event.t}"
            if event.t + _TOLERANCE_MS < last_t:
                problem(f"{where}: time goes backwards (previous {last_t})")
            if event.t > frame.t + _TOLERANCE_MS:
                problem(f"{where}: event is later than its frame (t={frame.t})")
            last_t = max(last_t, event.t)
            _apply(event, reqs, hello, where, problem, report)

    if complete:
        unfinished = sorted(
            rid for rid, r in reqs.items() if r.state not in {"finished", "aborted"}
        )
        if unfinished:
            problem(f"{len(unfinished)} requests never ended, e.g. {unfinished[:3]}")
    report.requests = len(reqs)
    return report


def _apply(
    event: ev.Event,
    reqs: dict[str, _Req],
    hello: ev.Hello | None,
    where: str,
    problem: Problem,
    report: TraceReport,
) -> None:
    match event:
        case ev.Hello():
            if hello is not None and event.replica != hello.replica:
                problem(f"{where}: hello for another replica")
        case ev.ReqArrived():
            if event.rid in reqs:
                problem(f"{where}: {event.rid} arrived twice")
            reqs[event.rid] = _Req(arrived_t=event.t)
        case ev.ReqAdmitted():
            r = _get(reqs, event.rid, where, problem)
            if r and r.state not in {"waiting", "preempted"}:
                problem(f"{where}: {event.rid} admitted while {r.state}")
            if r:
                r.state, r.prefilled = "admitted", False
        case ev.ReqFirstToken():
            r = _get(reqs, event.rid, where, problem)
            if r and (r.had_first_token or not r.prefilled):
                problem(f"{where}: {event.rid} first token out of order")
            if r:
                r.had_first_token = True
                _near(event.ttft_ms, event.t - r.arrived_t, f"{where}: ttft_ms", problem)
        case ev.ReqPreempted():
            r = _get(reqs, event.rid, where, problem)
            if r and r.state not in {"admitted", "running"}:
                problem(f"{where}: {event.rid} preempted while {r.state}")
            if r:
                r.state = "preempted"
        case ev.ReqFinished():
            r = _get(reqs, event.rid, where, problem)
            if r and (r.state != "running" or not r.had_first_token):
                problem(f"{where}: {event.rid} finished while {r.state}")
            if r:
                r.state = "finished"
                if event.usage.completion_tokens != r.sampled:
                    problem(
                        f"{where}: {event.rid} reports {event.usage.completion_tokens} tokens,"
                        f" steps sampled {r.sampled}"
                    )
                _near(event.timings.e2e_ms, event.t - r.arrived_t, f"{where}: e2e_ms", problem)
        case ev.ReqAborted():
            r = _get(reqs, event.rid, where, problem)
            if r and r.state in {"finished", "aborted"}:
                problem(f"{where}: {event.rid} aborted after it ended")
            if r:
                r.state = "aborted"
        case ev.Step():
            report.steps += 1
            _apply_step(event, reqs, where, problem)
        case ev.KvSnapshot():
            if (
                hello
                and hello.kv.num_blocks is not None
                and event.num_blocks != hello.kv.num_blocks
            ):
                problem(f"{where}: {event.num_blocks} blocks, hello said {hello.kv.num_blocks}")
            for rid in event.owners:
                r = reqs.get(rid)
                if r is None or r.state not in {"admitted", "running"}:
                    problem(f"{where}: owner {rid} is not admitted")
        case ev.RouterDecision():
            pass


def _apply_step(event: ev.Step, reqs: dict[str, _Req], where: str, problem: Problem) -> None:
    if len(set(event.rids)) != len(event.rids):
        problem(f"{where}: a request appears twice in one step")
    if event.kind == "prefill":
        if len(event.rids) != 1:
            problem(f"{where}: a prefill step has {len(event.rids)} rows, expected 1")
        for rid in event.rids:
            r = _get(reqs, rid, where, problem)
            if r and (r.state != "admitted" or r.prefilled):
                problem(f"{where}: prefill of {rid} while {r.state}")
            if r:
                r.prefilled, r.state = True, "running"
                r.sampled += 1
        return
    if event.running != len(event.rids):
        problem(f"{where}: running={event.running} but {len(event.rids)} rows")
    for rid in event.rids:
        r = _get(reqs, rid, where, problem)
        if r and r.state != "running":
            problem(f"{where}: decode of {rid} while {r.state}")
        if r:
            r.sampled += 1


def _get(reqs: dict[str, _Req], rid: str, where: str, problem: Problem) -> _Req | None:
    r = reqs.get(rid)
    if r is None:
        problem(f"{where}: {rid} was never announced")
    return r


def _near(reported: float, expected: float, what: str, problem: Problem) -> None:
    if abs(reported - expected) > 2 * _TOLERANCE_MS:
        problem(f"{what} is {reported}, expected {expected:.3f}")
