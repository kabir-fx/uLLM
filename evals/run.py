"""One eval run for one pinned model (Q5, decision 0035): perplexity on WikiText-2, then exact
match on GSM8K through ``ullm serve``. Writes ``result.json``, ``items.jsonl``, ``env.json``
and ``server.log`` into a new results directory.

    uv run python -m evals.run --model qwen3-0.6b-bf16 --out evals/results/m4/qwen3-0.6b-bf16

Exits 0 when the run finished, and 2 when it refuses to start.
"""

import argparse
import socket
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from bench.conditions import environment
from evals import data, exact_match, perplexity
from evals.schema import EvalResult

from ullm.models import DEFAULT_MODEL, MODELS


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Q5 evals for one model.")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL.key)
    parser.add_argument("--out", type=Path, required=True, help="a new results directory")
    args = parser.parse_args(argv)
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        print(f"{out} already holds files: choose a new directory", file=sys.stderr)
        return 2
    spec = MODELS[args.model]
    # Before any output exists, so `dirty` describes the code that ran.
    env = environment(used_powermetrics=False, started=datetime.now(UTC))

    model, tokenizer = perplexity.load(spec)
    ppl = perplexity.measure(
        model, perplexity.windows(tokenizer.encode(data.wikitext()), perplexity.WINDOW)
    )
    del model

    out.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    launch = (
        f"{sys.executable} -m ullm.cli serve --model {spec.key} --port {port}"
        " --max-num-seqs 1 --telemetry off"
    )
    summary, items = exact_match.run(launch, port, data.gsm8k(), out / "server.log")

    result = EvalResult(
        model=spec.key,
        repo=spec.repo,
        revision=spec.revision,
        dtype=spec.dtype,
        datasets=dict(data.SHA256),
        prompt_suffix=exact_match.PROMPT_SUFFIX,
        max_tokens=exact_match.MAX_TOKENS,
        chat_template_kwargs=exact_match.CHAT_TEMPLATE_KWARGS,
        perplexity=ppl,
        exact_match=summary,
    )
    (out / "items.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in items))
    (out / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    ended = datetime.now(UTC).isoformat(timespec="seconds")
    (out / "env.json").write_text(
        env.model_copy(update={"ended": ended}).model_dump_json(indent=2) + "\n"
    )
    print(
        f"{spec.key}: perplexity {ppl.perplexity:.3f} ± {ppl.standard_error:.3f},"
        f" exact match {summary.correct}/{summary.items},"
        f" {summary.output_tok_s:.1f} output tok/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
