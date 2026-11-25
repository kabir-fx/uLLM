# M2-T12: README v1, claims ledger, `make verify`

**State:** in build. You enable GitHub Pages and run `pages.yml` before `/ship`, so the replay link works.
**Decision record:** [0030](../decisions/0030-readme-claims-verify.md) (the full design, for agents) · **Spec:** roadmap M2-T12 (P1, P2)

## The goal

A stranger reads the README and sees what uLLM does now. It shows a demo, a benchmark table and a way to check each claim. `make verify` reproduces the claims on any Apple Silicon Mac with one command.

## The problem

- The README says that nothing serves a model yet. That is not true after M1 and M2.
- `make verify` exits 2 and names M2-T12.
- Gate A asks for a benchmark table that a script generates from committed runs. Rule 8 forbids a number that a person types by hand.
- `make goldens-check` compares the goldens exactly. On a different Apple chip, a near-tie can flip a token. That check can then fail on a correct Mac (decision 0006).
- The hosted replay has no public address. GitHub Pages is off.

## How it works

```
bench/results/m2/sweep/c1..c8 ──▶ bench.readme ──▶ README.md, between markers
                                       ▲
              tests/integration/test_readme_tables.py compares the two

terminal/dist-hosted ──▶ Playwright project "demo" (video) ──▶ ffmpeg ──▶ docs/assets/demo.gif

make verify ──▶ models · equivalence at 128 tokens · property tests · sim · bench-smoke
```

1. `bench/readme.py` loads the c1, c2, c4 and c8 sweep sessions. It writes one Sustained table of uLLM against batched `mlx_lm.server` into the README, with the machine and the commit.
2. `tests/integration/test_readme_tables.py` fails when the README table differs from the table that the script makes. `make ci-core` runs it, so CI runs it.
3. `make readme` runs the script. `make bench-sweep` runs it at the end.
4. `terminal/demo/record.spec.ts` plays the hosted replay of the real M2 recording and records a video. `make demo-gif` turns the video into `docs/assets/demo.gif`.
5. `make verify` runs the equivalence tests at 128 tokens. These tests accept a near-tie on another chip, but never on the chip that made the goldens.
6. `README.md` gets the sections that P1 names. `docs/CLAIMS.md` gets a row for each new claim.

## What you get at the end

- Run `make verify` on an Apple Silicon Mac. It downloads the model and runs the checks in the claims rows that it names.
- Run `make readme` after a new sweep. The README table changes with the committed results.
- Run `make demo-gif` to make the GIF again.

New tests:

- The README table equals the table that `bench.readme` makes from the committed sessions.

What you do not get yet:

- Benchmark tables for workloads B, C and D: M3 and later.
- The final README and the full claims ledger: M7-T3.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 █  T5 █  T6 █  T7 █  T8 █  T9 █  T10 █  T11 █  T12 ▓  T13 ░
```

After this slice, M2-T13 is open for you, and then Gate A.

## Choices the agent made

You chose two items: a script records the GIF, and you publish the replay before `/ship`. You can still change these choices before `/ship`:

- The README table shows Sustained only. Peak has an interval only at c1.
- The freshness test is in `tests/integration`, not `tests/unit`. It needs about 30 s, so `make check` stays fast.
- The README gives the line count of `core` and `runner` at one commit, with the command. A script does not keep it current.
- `make verify` replaces `make goldens-check` in claims row 8, because the exact check fails on another chip.

## What can go wrong

- A new sweep changes the results, but no one runs `make readme`. CI fails on the freshness test.
- The replay link is dead until you run `pages.yml`.
- `make verify` needs about 1.2 GB for the model, and more time than `make check`.

## Words

| Word | Meaning |
|---|---|
| Sustained | The regime where the GPU clock is capped for at least half of the busy time (decision 0016). |
| freshness test | A test that fails when a generated file differs from what its script makes now. |

## Learn from this slice

A number in a README goes stale when the results change. A generator and a freshness test make a stale number fail CI.

Test yourself: why does `make verify` not run `make goldens-check`?

<details><summary>Answer</summary>

`make goldens-check` needs the exact tokens of the goldens. On another Apple chip, bf16 arithmetic can flip a near-tied token. The equivalence tests accept that flip only at a near-tie, so they pass on a correct Mac.

</details>

## Result
