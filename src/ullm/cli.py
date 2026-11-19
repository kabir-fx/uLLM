"""The ``ullm`` command. ``ullm serve`` runs the API server (A1, decision 0011); ``ullm sim`` runs
the real engine on FakeRunner (DEC-3, decision 0010)."""

import argparse
import asyncio
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from ullm.core.client import EngineClient, Unavailable
from ullm.core.engine import Delta, EngineConfig, Failed, Finished
from ullm.models import DEFAULT_MODEL, MODELS
from ullm.runner.base import Runner, RunnerInfo, SamplingParams, Tokenizer
from ullm.runner.fake_runner import FakeDetokenizer, FakeRunner, FakeTokenizer, Latency
from ullm.server.app import DEFAULT_MAX_TOKENS, ServeConfig, create_app
from ullm.telemetry import events as ev
from ullm.telemetry import trace
from ullm.telemetry.check import check_trace

SIM_INFO = RunnerInfo(
    model_id="fake", vocab_size=1000, eos_token_ids=frozenset({0}), max_model_len=4096
)
SIM_CONFIG = EngineConfig(source="sim")
# A latency model for watching the engine, not a measurement of any machine.
SIM_LATENCY = Latency(
    prefill_base_ms=2.0, prefill_per_token_ms=0.01, decode_base_ms=1.0, decode_per_row_ms=0.2
)


async def _client(engine: EngineClient, rng: random.Random, disconnect: float) -> str:
    """One simulated client; returns how its request ended."""
    await asyncio.sleep(rng.uniform(0, 0.5))
    leaves = rng.random() < disconnect
    prompt = tuple(rng.randrange(1, SIM_INFO.vocab_size) for _ in range(rng.randint(1, 256)))
    sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=rng.randrange(2**32))
    stop = (f" t{rng.randrange(SIM_INFO.vocab_size)} ",) if rng.random() < 0.2 else ()
    try:
        stream = engine.submit(
            prompt, sampling, max_tokens=64 if leaves else rng.randint(1, 64), stop=stop
        )
    except Unavailable as e:
        return f"refused ({e.why})"
    outcome = "no end"
    async with stream:
        async for out in stream:
            match out:
                case Delta() if leaves:
                    return "left early"
                case Finished(reason=reason):
                    outcome = f"finished ({reason})"
                case Failed(reason=reason):
                    outcome = f"failed ({reason})"
    return outcome


async def _sim(clients: int, seed: int, disconnect: float) -> tuple[list[ev.Event], Counter[str]]:
    events: list[ev.Event] = []
    engine = EngineClient.start(
        FakeRunner(SIM_INFO, SIM_LATENCY),
        FakeDetokenizer,
        SIM_CONFIG,
        events.append,
    )
    outcomes = await asyncio.gather(
        *(_client(engine, random.Random(f"{seed}-{i}"), disconnect) for i in range(clients))
    )
    await engine.close()  # the engine thread has exited, so events is complete
    return events, Counter(outcomes)


def _run_sim(args: argparse.Namespace) -> int:
    events, outcomes = asyncio.run(_sim(args.clients, args.seed, args.disconnect))
    frames = trace.frames_from(events, SIM_CONFIG.replica)
    if args.trace:
        trace.write(args.trace, frames)
    report = check_trace(frames)
    print(f"{args.clients} clients, {report.steps} steps, {len(events)} events")
    for outcome, n in sorted(outcomes.items()):
        print(f"  {n:4d}  {outcome}")
    # A refusal past max_in_flight is the bound working; the trace check covers the rest.
    bad = [o for o in outcomes if not o.startswith(("finished", "left early", "refused (full)"))]
    for problem in report.problems:
        print(f"trace problem: {problem}", file=sys.stderr)
    if bad:
        print(f"clients that did not end well: {bad}", file=sys.stderr)
    if report.ok and not bad:
        print("trace check: ok")
        return 0
    return 1


def _serve(args: argparse.Namespace) -> int:
    runner: Runner
    tokenizer: Tokenizer
    if args.fake:
        runner, tokenizer = FakeRunner(SIM_INFO, SIM_LATENCY), FakeTokenizer()
        served = args.served_model_name or SIM_INFO.model_id
    else:
        from ullm.runner.mlx_runner import load_mlx  # MLX exists only on Apple Silicon

        spec = MODELS[args.model]
        runner, tokenizer = load_mlx(spec)
        served = args.served_model_name or spec.repo
    config = ServeConfig(
        served_model=served,
        default_max_tokens=args.default_max_tokens,
        engine=EngineConfig(
            max_num_seqs=args.max_num_seqs,
            max_kv_slots=args.max_kv_slots,
            max_in_flight=args.max_in_flight,
        ),
    )
    # Local by default (rule 14): no option binds another address.
    uvicorn.run(create_app(runner, tokenizer, config), host="127.0.0.1", port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ullm")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the OpenAI-compatible API server")
    serve.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL.key)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", help="the model id clients send; default the repo")
    serve.add_argument("--default-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    serve.add_argument("--max-num-seqs", type=int, default=EngineConfig().max_num_seqs)
    serve.add_argument(
        "--max-kv-slots",
        type=int,
        default=EngineConfig().max_kv_slots,
        help="padded KV slots the batch may reserve; at least the model's max_model_len",
    )
    serve.add_argument("--max-in-flight", type=int, default=EngineConfig().max_in_flight)
    serve.add_argument("--fake", action="store_true", help="FakeRunner and FakeTokenizer: no model")
    sim = commands.add_parser("sim", help="run the real engine on FakeRunner with fake clients")
    sim.add_argument("--clients", type=int, default=50)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--disconnect", type=float, default=0.2, help="share that leave early")
    sim.add_argument("--trace", type=Path, help="also write the frames to this .jsonl.gz")
    args = parser.parse_args(argv)
    return _serve(args) if args.command == "serve" else _run_sim(args)


if __name__ == "__main__":
    sys.exit(main())
