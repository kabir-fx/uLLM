# M2-T9: Replay mode and the hosted build

**State:** DONE at `c8dceb7`. Type `/ship` to open the pull request.
**Decision record:** [0026](../decisions/0026-replay-and-hosted-build.md) (the full design, for agents) · **Spec:** roadmap M2-T9 (T4, DEP3)

## The goal

Anyone can open a public page and watch a recorded uLLM run, with play, pause, speed and a scrubber. The page always says that it is a replay, and it names the machine and the commit of the recording.

## The problem

- The replay plays forward only, at a speed that `?speed=` sets before the page loads.
- In replay mode, the side column is empty. T4 puts a run picker there.
- The one real recording has `commit: null` and `machine: null` in its `hello`. `ullm serve` does not fill them.
- Nothing is hosted. GitHub Pages is not enabled on the repo.
- Each publish needs your approval (DEP3, rule 14). Lane B may not change a contract (roadmap 3.5).

## How it works

```
runs.ts ──▶ run picker ──?run=id──▶ sourceFromQuery ──▶ ReplaySource ──frames──▶ TelemetryStore ──▶ panels
                                                            ▲
                                      replay panel: play, pause, speed, seek
```

1. `terminal/src/sources/replay.ts`: `ReplaySource` adds `play`, `pause`, `setSpeed` and `seek`. A seek back sends the trace again from its `hello`, and the store resets itself.
2. `terminal/src/runs.ts`: the list of bundled runs, with the machine and the commit of each real recording.
3. `terminal/src/runs.test.ts`: fails when the bundled copy or its machine and commit differ from the bench session.
4. `terminal/src/config.ts`: `sourceFromQuery` gives a live mode or a replay mode. The hosted build ignores `?live=` and `?trace=`.
5. `terminal/src/ui/ReplayPanel.tsx`: the run picker, the transport and the scrubber, in place of the launcher.
6. `terminal/src/ui/Header.tsx`: a REPLAY badge, and the machine and commit of the run.
7. `.github/workflows/pages.yml`: builds `make terminal-hosted` and deploys it to Pages. It runs only when you start it.

## What you get at the end

Run `pnpm -C terminal dev` and open `http://127.0.0.1:5173/`. Pick a run, then play, pause, scrub, and change the speed. Run `make terminal-hosted` for the static bundle in `terminal/dist-hosted/`.

To publish, enable Pages once with the source "GitHub Actions". Then, after the merge, run `gh workflow run pages.yml`. The page will be at `https://kabir-fx.github.io/uLLM/`.

New tests:

- Pause stops the frames, and play continues from the same position.
- A new speed changes the rate with no jump in position.
- A seek forward sends the frames between. A seek back rebuilds the store from the `hello`.
- Play at the end starts again from 0.
- The bundled copy and its labels equal the bench session.
- The hosted mode refuses `?live=` and `?trace=`.
- The header shows REPLAY, the machine, and the commit.

What you do not get yet:

- The Playwright smoke test and keyboard basics: M2-T10.
- Runs for workloads B and C: M3-T8.
- The machine and commit in the engine's own `hello`: a Lane A change.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 █  T5 █  T6 ░  T7 █  T8 █  T9 ▓  T10 ░  T11 ░  T12 ░  T13 ░
```

After this slice, M2-T10 is open in Lane B.

## What you approve

1. The machine and commit come from a list in the terminal, held equal to `env.json` by a test. You chose this over a change to `ullm serve`.
2. The host is GitHub Pages, and a workflow publishes only when you start it. You chose this over a build with no host.
3. The hosted page plays only its own runs. You chose this over a page that keeps `?live=` and `?trace=`.

Choices the agent made, which you can still change before `/ship`:

- A test holds the list equal to `env.json`. A script that writes the list is the other option.
- The real recording is copied into `terminal/fixtures/runs/`. A test holds the copy equal to the bench file.
- `?trace=` still works in the local build, for your own recordings.

## What can go wrong

- The real recording comes from a dirty tree at `fab197c`. The header shows "dirty" next to the commit.
- A seek back on a long trace sends every frame again. With 287 to about 700 frames, this is fast.
- A host that does not decode `.gz` files still works, because the terminal reads gzip itself.

## Words

| Word | Meaning |
|---|---|
| run | One bundled trace, with a title and, for a real recording, its machine and commit. |
| hosted build | The static terminal from `vite build --mode hosted`, which plays only bundled runs. |
| seek | A move of the replay position to a new trace time. |

## Learn from this slice

A replay clock needs a base position and the wall time of that base. Each pause, speed change and seek takes a new base. Then the position never jumps, and timer lateness never accumulates.

The store already resets on a `hello`, because each live connection starts with one. A seek back reuses that rule. The replay sends the trace again from frame 0, and the store rebuilds itself.

Test yourself: why does a seek back send the trace again from frame 0?

<details><summary>Answer</summary>The store keeps state that frames add to, such as request rows and windows. It cannot take frames away. Only a reset from the `hello` and a new pass gives the state at an earlier time.</details>

## Result

**State:** DONE. The proof is VERIFIED at `41f0c7a`: 11 of 11 claims. The review is READY at `c8dceb7`. The re-check at `9d1533b` found one bug. You accepted its fix, `c8dceb7`, as covered. `make check` passes there, and the change used 3 agent runs.

**Against the plan:** `sourceFromQuery` now takes Vite's `import.meta.env` whole, and not a base and a flag. Then the tests reach the choice of the hosted mode.

**What proof and review found:**

- No test covered the hosted wiring in `main.tsx`. Now `sourceFromQuery` reads `MODE` itself, and a mutation of it fails a test.
- The scrubber had `step=100`, so it stopped 40 ms short of the end of the real run. It now has `step="any"`.
- A pause pressed while the trace loaded was lost. A test now holds it.
- The re-check found that a play after that early pause was then lost. A test failed before the fix in `c8dceb7` and passes after it.

**What the verifier saw:** the hosted build ran under `/uLLM/` from a server that does not decode gzip. Every run played to the end, and `?live=` and `?trace=` were ignored. After a seek back from the end, the table went from 32 finished requests to 0.

**What is not proven:** the Pages workflow did not run. The public URL does not exist until you enable Pages and start the workflow after the merge.

**Read these yourself:**

- `terminal/src/sources/replay.ts`: `seek`, and the reset to frame 0 when the target is before the last frame sent.
- `terminal/src/config.ts`: `sourceFromQuery`, and the two places where `hosted` drops `?live=` and `?trace=`.
- `.github/workflows/pages.yml`: only `workflow_dispatch`, and the two Pages actions pinned by commit.

