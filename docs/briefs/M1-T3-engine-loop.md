# M1-T3: The engine loop

**State:** DONE at `1d2d9e8`. You can now type `/ship`.
**Decision record:** [0010](../decisions/0010-engine-loop-and-thread-bridge.md) (the full design, for agents) · **Roadmap:** M1-T3 (E4, E5, E6)

## The goal

This slice connects the parts that we have into one engine. The engine takes many requests, runs them step by step, and streams the text back to each client.

## The problem

- The API server (M1-T4) runs on one thread, the asyncio event loop. One slow call on that thread stops every client.
- A prefill step keeps its thread busy for a long time. If the model runs on the event loop, the server cannot answer anyone during that time.
- Rule 2 says that only one thread, the engine thread, touches the model, the scheduler, and the requests.
- Thus we need two threads that talk through queues, with no locks.

## How it works

```
API thread (asyncio event loop)                Engine thread
+-------------------------------+              +-------------------------------+
| EngineClient                  |   tickets    | Engine loop, one turn:        |
|   submit()  -> a stream       | -----------> |   1. read new tickets         |
|   health()  never waits       |              |   2. Scheduler: plan one step |
|   close()                     |   outputs    |   3. Runner: do the step      |
|   one queue for each request  | <----------- |   4. Request: add the token   |
+-------------------------------+  one batch   |   5. send the outputs, once   |
                                   per turn    +-------------------------------+
```

The agent will build four parts. All four are in `src/ullm/core`, and none of them imports MLX.

1. **Request** (`request.py`). It holds the state of one request. The state moves along one fixed list of legal moves: `WAITING`, then `PREFILLING`, then `RUNNING`, then `FINISHED`. A live request can also move to `ABORTED`. An illegal move raises an error and changes nothing. Each move writes one event to the trace.
2. **Scheduler** (`scheduler.py`). It selects the work for the next step. The oldest request goes first. A step does one prefill or one decode, not both. For now, only one request runs at a time.
3. **Engine** (`engine.py`). This is the loop on the engine thread. The picture shows the five things that it does in each turn.
4. **Client** (`client.py`). This is the side of the API thread. `submit` checks the request, puts it on the queue, and gives back a stream. The stream gives zero or more `Delta` texts, then exactly one `Finished` or `Failed`. If the caller leaves early, the client tells the engine to abort the request.

Then `ullm sim` uses all four parts with the fake model and many fake clients.

**A fact that the agent measured first.** MLX says that its GPU work queue works only on the thread that made it. Thus the agent tested if the model can load on one thread and run on another. On your M4, it generated 32 tokens in three ways, and all three gave the same tokens. Thus the server can load the model at startup, before the engine thread starts. The agent did not keep the test code.

## What you get at the end

- You can type `make sim`. It starts the real engine with the fake model and many fake clients. Some clients leave early on purpose. Then the trace checker reads every event. The command fails if one request did not end correctly.
- New tests cover the state machine, the scheduler, the engine loop, and 50 clients at the same time.
- One test on the real model makes sure that the loop still gives the same tokens as mlx-lm (rule 1).

What you do not get yet:

- No HTTP. You cannot send `curl` to the engine until M1-T4.
- One request at a time. Batches of many requests come in M2-T2.

## Where this puts the project

```
     done █   this slice ▓   not started ░

M0   █████████       9 of 9
M1   ██▓░░░░░        2 of 8, then 3 of 8 after this slice
M2   ░░░░░░░░░░░░░   0 of 13    <- Gate A, tag v0.1: 12 of 30 tasks after this slice
M3   ░░░░░░░░░       0 of 9
M4   ░░░             0 of 3
M5   ░░░░░░          0 of 6
M6   ░░░░░           0 of 5
M7   ░░░░░           0 of 5     <- Gate B, tag v1.0
M8   ░░░░░░          0 of 6
M9   ░░░             0 of 3     All: 12 of 67 tasks after this slice
```

This slice opens the next tasks:

- M1-T4, the API server, calls `EngineClient` directly. It changes `ValueError` to HTTP 400 and `Unavailable` to HTTP 503.
- M2-T2 changes only the limit in the scheduler. The loop and the bridge between the threads stay the same.

## What you approve

These are the choices that you can disagree with. Each one names the option that the agent did not select.

1. **The model gets its own thread.** The other option was a coroutine on the event loop, with only the model step on a helper thread. That option has one thread less. But the scheduler then runs on the API thread, and that breaks rule 2.
2. **The client sets the limit on load, not the queue.** Not more than 64 requests can be open at the same time. Request 65 gets `Unavailable`, and the server sends HTTP 503. A limit on the queue does not work. The engine empties the queue at each turn, so the real waiting line has no limit. **This changes the roadmap text for E5 in the same pull request.**
3. **One wake-up for each engine step, not one for each token.** The engine sends all outputs of a turn in one call. Thus the event loop wakes one time for each step, not one time for each token of each request.
4. **Each state change writes its own event.** A state change cannot occur without its event, so the trace is always complete. The other option was to give the events back to the caller. But then a caller can forget one.

## What can go wrong

- A new request can wait one full step before the engine reads it. Thus a long prefill makes the wait of a new request longer.
- If the engine has a bug, the engine thread stops. Every open stream then gets `Failed("error")`, and `health()` shows `alive=False`. No request waits forever.
- `ullm sim` uses a real clock, so two runs give different traces. The checker checks each trace. It does not compare it with a saved trace.

## Words

| Word | Meaning |
|---|---|
| Request | One prompt and its reply, from arrival to end. |
| Ticket | A message from the API thread to the engine thread: `Submit`, `Abort`, or `Shutdown`. |
| Step | One run of the model: one prefill or one decode. |
| Prefill | The step that reads the full prompt and gives the first token. |
| Decode | A step that gives one more token to each request that runs. |
| Event loop | The one asyncio thread that runs the server. A slow call on it stops every client. |
| Trace | The list of events from the engine. The trace checker reads it to find errors. |

## Learn from this slice

1. **Why the model must not run on the event loop.** asyncio does one thing at a time on one thread. A coroutine gives control back only at an `await`. A long model step has no `await`, so every client waits.
2. **A state machine with one list of legal moves.** `TRANSITIONS` is the only list. One private method, `_move`, makes every change and checks the list. Thus a bad state cannot occur, and the tests can try every illegal pair.
3. **Two threads with no locks.** The API thread puts tickets on a `SimpleQueue`. The engine thread sends outputs back with `call_soon_threadsafe`, which runs a function on the event loop. All client data stays on one thread, so no lock is necessary.

**Test yourself.** A client calls `submit` at the moment the engine thread stops. Can that request wait forever?

<details><summary>Answer</summary>

No. `submit` and the exit callback both run on the event loop thread, so they cannot run at the same time. If `submit` runs first, the exit callback fails the request. If the exit callback runs first, `submit` refuses the request with `Unavailable("stopped")`.

</details>

## Result

**State.** DONE. The proof is VERIFIED at `126622e`. The review is READY and covers the fix commits to `1d2d9e8`. `make ci` passed at `1d2d9e8`.

**Changes against the plan.**

- `Finished.text` holds the last piece of the reply. The `Delta` texts plus this piece are the full reply.
- The engine sends `hello` as its first event.
- A stream that the caller leaves stays open until the engine ends it. Thus the limit counts what the engine still holds.
- The client never sends `Failed("shutdown")` at exit. After a `Shutdown`, the engine already ended every request that it got.

**Bugs that proof and review found.** The agent fixed each one with a test first.

1. A bug after the end of a reply hid the first error. The engine also did not end the other requests in the trace.
2. A duplicate `rid` made the engine wait forever. Now the engine stops with an error.
3. `ullm sim` failed when the client limit refused clients. A refusal is now a good result.
4. `health()` could show `alive=True` for a short time after the exit callback.
5. No test failed a prefill step. A new test does.

**Not proven.** The agent did not measure a real prefill on the event loop. The fake latency model gave the 500 ms step.

**For later slices.**

- M1-T4 must set a limit on the length of each stop string. The schema limits only the number of stop strings.
- M1-T4 must not accept an abort by `rid` from a client. The ids are easy to guess.
- `Submit` errors repeat the full prompt. M1-T4 must not send that text back in a 400 reply.
- `tests/recording.py` matches tokens to requests through the engine's own step events. M2-T2 must match them in a different way.

**Read these yourself.**

- `src/ullm/core/engine.py:168`: the engine waits on the queue only when no request is live.
- `src/ullm/core/engine.py:235`: an ended request leaves `_live` before the runner releases it.
- `src/ullm/core/client.py:168`: the exit callback fails every open stream. Then no stream can wait forever.
