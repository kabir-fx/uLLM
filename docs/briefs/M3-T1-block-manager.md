# M3-T1: Block manager

**State:** in build. Read the Result when `/build` reports.
**Decision record:** [0032](../decisions/0032-block-manager.md) (the full design, for agents) · **Spec:** roadmap E8 and Q2, task M3-T1

## The goal

The engine will count KV memory in blocks of 16 tokens, not in padded slots. This slice adds the bookkeeping for those blocks, with no change to what the server does.

## The problem

- M2 reserves `prompt + max_tokens` padded slots per row (decision 0020). Thus short requests hold memory that they never use.
- Prefix caching (M3-T5) and preemption (M3-T6) need blocks that requests can share, keep and give back.
- Rule 2: only the engine thread touches the block state, so the manager has no locks.
- Rule 3: `ullm.core` never imports MLX. The manager counts blocks. It does not hold KV tensors.

## How it works

```
            allocate / append_slot
   Free  ------------------------->  In use  (ref count >= 1)
    ^                                 |   ^
    | evicted when the                |   | shared: a prefix hit
    | free list is empty         free |   |
    |                                 v   |
    +-------------------------------  Cached  (ref count 0, key kept)
```

1. `BlockManager`, in `src/ullm/core/block_manager.py`, holds a fixed pool of blocks and one table per sequence.
2. `allocate` gives a sequence one block per 16 tokens, rounded up. It can start the table with `shared` blocks from a prefix hit.
3. `append_slot` gives room for one decoded token. It takes a new block only when the last block is full.
4. `register` gives a full block its key. M3-T5 will compute the key as a prefix hash. `lookup` finds the block for a key.
5. `free` lowers the ref count of each block. A block at 0 with a key becomes Cached. A block at 0 without a key becomes Free.
6. When the free list is empty, the manager evicts the oldest Cached block and forgets its key.

## What you get at the end

- `make check` runs `tests/unit/test_block_manager.py` and `tests/property/test_block_manager.py`.
- The unit tests prove the rounding, the append rule, the Cached state, sharing, and the eviction order.
- The property test runs random sequences of allocate, share, append, register, free and preempt. It checks every invariant of Q2 after each step.

What you do not get yet:

- A pool size from a memory budget, and the block metrics: M3-T4.
- The physical KV store behind the blocks: M3-T2 and M3-T3.
- Prefix hashes: M3-T5. The choice of which request to preempt: M3-T6.

## Where this puts the project

```
M3  ▓░░░░░░░░
```

This slice opens M3-T3, the KV store, and M3-T4, admission by free blocks. M3-T2, the layout spike, does not depend on it.

## Choices the agent made

You can still change these before `/ship`.

- The manager owns the tables. A request reads its table from the manager. E4 lists the table as a request field. One owner keeps the ref counts and the tables in one place.
- Sharing is part of `allocate`. The manager revives the shared blocks first, so it never evicts a block that it is about to share.
- Eviction has no public call. It happens only when the manager needs a block and the free list is empty.
- `free` gives back the last block of a table first. Thus the end of a cached prefix is evicted before its start.
- A second block with a key that another block already holds stays without a key.

## What can go wrong

- `num_in_use` counts every block on each call. For a large pool, a metric scrape will cost one pass. M3-T4 can keep a counter.
- The manager trusts the caller for keys. Two different prefixes with one key will share wrong KV. M3-T5 must make the hash collision-safe.

## Words

| Word | Meaning |
|---|---|
| block | 16 token slots of KV, the unit that the manager counts. |
| block table | The list of blocks that holds one sequence's KV, in token order. |
| ref count | The number of block tables that hold a block. |
| key | The prefix hash of a full block. A Cached block keeps its key for a later hit. |
| Cached | A block with ref count 0 that keeps its contents and its key. |

## Learn from this slice

A block is useful to a prefix hit only after every block before it in the chain. Thus `free` gives back the tail first, and the oldest Cached block is evicted first.

A shared block must be read-only. Only a full block can take a key, and only a block with a key is shared. Thus no request can write into a block that another request reads.

Test yourself: why does `allocate` revive the shared blocks before it takes new blocks?

<details><summary>Answer</summary>

A new block can come from the Cached blocks when the free list is empty. If a shared block is the oldest Cached block, the manager can evict it to make room. The same sequence then holds it twice, with the wrong contents in one place.

</details>

## Result
