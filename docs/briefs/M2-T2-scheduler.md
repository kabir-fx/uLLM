# M2-T2: Scheduler with continuous batching

**State:** in build. You chose the recommended answer to each of the four questions. Read the Result when `/build` reports.
**Decision record:** [0020](../decisions/0020-continuous-batching-admission.md) (the full design, for agents) · **Spec:** roadmap E6, task M2-T2

## The goal

The engine will serve many requests at the same time, in one decode batch. A new request joins the batch while others decode.

## The problem

- `EngineConfig` refuses `max_num_seqs` above 1. Thus the engine never uses the batch KV from M2-T1.
- A batch holds as many slots as its row count times its longest row. One long prompt at batch 8 can need about 37.6 GB of KV.
- Nothing can remove a running request until preemption arrives in M3. Thus the scheduler must refuse a request that can overfill the batch later.
- Rule 2: only the engine thread touches the scheduler. Rule 3: `ullm.core` stays free of MLX.
- E6: admission is first come, first served. A request that does not fit stops every request behind it.

## How it works

```
waiting deque (FCFS)                      running list
[ d  c  b ] --- head fits? ----yes----->  [ a  b ]   prefill b alone this step
                   |                        |
                   no: nobody admitted      next step: decode a and b together
```

1. `Scheduler`, in `src/ullm/core/scheduler.py`, looks at the head of the waiting queue on each step.
2. Each request reserves `prompt + max_tokens` slots. This is the most KV its row can ever hold.
3. The head is admitted only when two limits hold with it. The row count stays at or below `max_num_seqs`. The row count times the largest reservation stays at or below `max_kv_slots`.
4. `Engine`, in `src/ullm/core/engine.py`, refuses a `max_kv_slots` below the model's `max_model_len`. Thus one request alone always fits.
5. `hello` reports both limits. `check_trace` reports a step with more rows than `max_num_seqs`.

One slot costs 114,688 bytes on the Qwen3 models. The agent read this from the model's `config.json`. The default of 40,960 slots is about 4.7 GB of KV.

## What you get at the end

- `ullm serve --max-num-seqs N --max-kv-slots S` serves up to N requests in one batch. The default N is 16.
- `make sim` runs 50 clients through batched steps and checks the trace.
- Unit tests prove the two limits, the FCFS order with no skipping, and the refusal of a small budget.
- An engine test proves that a failed batch fails each request in it, and that the waiting requests still finish.

What you do not get yet:

- Equivalence at batch 2, 4 and 8, and the scheduler property tests: M2-T3.
- The benchmark sweep across concurrency: M2-T6.
- Admission by free blocks: M3-T4.

## Where this puts the project

```
M0  █████████
M1  ████████
M2  █▓░░░░░░░░░░░
```

This slice opens M2-T3, the batched equivalence tests, and M2-T6, the concurrency sweep.

## Choices the agent made

You can still change these before `/ship`.

- The reservation counts `prompt + max_tokens`, not the current length. Without preemption, a count of current lengths can grow past the budget.
- The budget is never below `max_model_len`, so the bound has no exception.
- The ordering policy stays the deque order. E16 adds an interface in M8, with its second policy.
- The engine-level equivalence test at batch size 1 sets `max_num_seqs=1`, because its requests arrive together.

## What can go wrong

- A request with a large `max_tokens` holds room that it may not use. Fewer requests then share a batch.
- The default budget does not count the weights or the copy during a repack. At the default, the peak KV is about 9.4 GB.
- Batched greedy output can differ from batch-size-1 output at near-ties, as decision 0019 says. M2-T3 sets the threshold.

## Words

| Word | Meaning |
|---|---|
| reservation | The slots a request can hold at most: its prompt length plus its `max_tokens`. |
| padded KV slots | Rows times the longest row: the slots that the batch KV holds. |
| head of line | The oldest waiting request. When it does not fit, no request behind it is admitted. |

## Learn from this slice

An admission check is only as safe as the worst growth after it. With no preemption, the scheduler must count the most that each row can grow to.

First come, first served means that a large request can hold back small ones. This is the price of a simple order with no starvation.

Test yourself: why does the engine refuse a `max_kv_slots` below `max_model_len`?

<details><summary>Answer</summary>

A legal request can reserve up to `max_model_len` slots. With a smaller budget, that request never fits, even alone. It then blocks the head of the queue for ever.

</details>

## Result

