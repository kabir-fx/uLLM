# 0032: The block manager owns the block tables; keys are opaque; eviction is implicit

**Status:** accepted · **Task:** M3-T1 (E8, Q2)

## Context

E8 asks for a logical, MLX-free block manager: a fixed pool of 16-token blocks, a free list, a block table per request, a ref count per block, and three block states (Free, In use, Cached). M3-T3 maps blocks to physical KV, M3-T4 admits by free blocks, M3-T5 adds prefix hashing, M3-T6 preempts. Nothing calls the manager in M3-T1. E4 lists the block table among a request's fields.

## Decision

1. `ullm.core.block_manager.BlockManager(num_blocks, block_size=16)` owns every table, keyed by `SeqId`. A request reads its table through `table(seq)`. It does not hold a copy (a change from E4's field list).
2. The interface: `can_allocate(num_tokens, shared=())`, `allocate(seq, num_tokens, shared=())`, `can_append(seq)`, `append_slot(seq)`, `register(seq, index, key)`, `lookup(key)`, `free(seq)`, `table(seq)`, `ref_count(block)`, and the counts `num_free`, `num_in_use`, `num_cached`, `evictions`.
3. Sharing goes through `allocate`: the `shared` blocks are the table's first blocks, then new blocks cover the rest. Each shared block must have a key, which only a full block can take, so a shared block is full and read-only by construction. The shared blocks are revived before any new block is taken, so making room never evicts one of them. A Cached block being revived does not count as room.
4. A key (`BlockKey = int`) is opaque here. E10 computes the chained prefix hash in M3-T5. `register` gives a full block a key. It returns False when another block already holds that key: the newer block stays unkeyed and becomes Free on release.
5. Eviction has no public call. `_take` uses the free list first, then the least recently cached block, and drops its key. `free` releases a table tail first, so a chain's later blocks are evicted before its head, which a later prefix hit needs first.
6. Preemption is `free` at this layer. Which request to preempt, and re-admission, are E11's (M3-T6).
7. `OutOfBlocks` means nothing changed. Misuse (a sequence allocated twice, a shared block without a key, a block keyed twice, a non-full block registered) raises `ValueError`. An unknown sequence raises `KeyError`.
8. Sizing the pool from a memory budget, and E8's metrics, land in M3-T4 (the author's call): sizing needs the bytes per block from the KV layout (M3-T2, M3-T3), and T4 is where the engine first builds a manager.

## Alternatives considered

- **A shallow pool (`take(n)`, `retain`, `release`) with the table on `Request`.** The ceil arithmetic, the append-when-full rule and the share-before-take order would live in every caller, and Q2 could not test them in one place.
- **A public `evict()`.** No caller needs to evict except to make room, and a separate call lets a caller evict a block it is about to share.
- **Hashing inside the manager now.** E10's chain needs token IDs and the cache salt (E12). Keeping keys opaque leaves the manager free of both, and M3-T5 adds the hashing without changing this interface.

## Consequences

- `tests/property/test_block_manager.py` runs allocate, share, append, register, free, preempt and fill-the-pool at random. After every operation it checks: each table has `ceil(tokens / block_size)` distinct blocks, so waste stays below one block; each ref count equals the number of tables holding the block, and none is negative; free + in use + cached equals the pool; a block in use keeps its key; a key finds its own contents or a Cached block; and the whole free and cached room can be allocated at once.
- When every prompt block hits, E10 still recomputes the last prompt token, and a shared block is read-only. M3-T5 must therefore pass one block fewer as `shared` and let the last prompt block be the sequence's own.
- `num_in_use` scans the ref counts, O(`num_blocks`). M3-T4 can keep a counter if a metric scrape needs one.
