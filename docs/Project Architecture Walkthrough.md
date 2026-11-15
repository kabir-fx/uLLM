# Brigade: LLM inference engine on Apple Silicon (architecture, Sections 0-13)

Text version of `brigade-architecture.excalidraw`, extracted from the file and checked so that every label in the drawing appears here.
Use it as context for Claude Code; the drawing has the same content laid out visually.

Panels: System overview (master) | 0 one request inside the model | 0b one token's journey through the hardware | 1 model runner and sampler | 2 KV cache | 3 prefix caching | 4 scheduler | 5 engine loop and concurrency | 6 API server | 7 tool calling | 8 Go router | 9 metrics and tracing | 10 live terminal | 11 testing and evals | 12 benchmarking | 13 later additions.

Every label, note and connection is included. Conventions:
- `Box: text` = a labelled box (lines inside a box are joined with ` / `).
- `A -> B [label: ...]` = an arrow from A to B; `<->` = two-way; `(dashed)` = dashed line.
- `(dashed box)` = future work or a placeholder; `[Name]` = a sub-section frame.
- Parts that are only drawn (grids, timelines, bars, chart curves) are written out as tables or short descriptions.
- Colours: S0 blue, S1 violet, S2 brown, S3 teal, S4 orange, S5 pink, S6 indigo, S7 lime, S8 grape, S9 yellow, S10 mint, S11 green, S12 dark blue; red = problems/warnings, green = correct/success.

---

## System overview (master)

- Box: Legend / solid box = explained so far (S0-S13) / dashed grey box = coming in later sections / (S#) = section
- Circle: Clients / curl / OpenAI SDK / OpenCode agent
- Box: Go router (S8) / prefix-aware routing / health checks, retries
- Box: Observability (S9-S10) / Prometheus + Grafana / OpenTelemetry traces / live terminal
- Box: Quality (S11-S12) / tests, evals, CI gate / benchmarks A-D vs baselines
- Box: Later (S13): rate limits, MCP, chunked prefill, k8s... (dashed box)
- Arrow: Go router (S8) / prefix-aware routing / health checks, retries <-> API server (S6-S7) / OpenAI-compatible, SSE / tool-call parser (S7)  [label: HTTP + SSE]
- Arrow: [Engine thread (S5): one loop, one step at a time; the only thread that touches MLX] -> [Apple M4 Air (S0)]  [label: MLX / calls]  (dashed)
- Arrow: Clients / curl / OpenAI SDK / OpenCode agent <-> Go router (S8) / prefix-aware routing / health checks, retries

### Engine replica x2-3 (Python process, runs natively on the Mac)

- Box: API server (S6-S7) / OpenAI-compatible, SSE / tool-call parser (S7)
- Box: Request queue (S5) / thread-safe
- Box: Output queues (S5) / one per request
- Arrow: Output queues (S5) / one per request -> API server (S6-S7) / OpenAI-compatible, SSE / tool-call parser (S7)
- Arrow: API server (S6-S7) / OpenAI-compatible, SSE / tool-call parser (S7) -> Request queue (S5) / thread-safe
- Arrow: Request queue (S5) / thread-safe -> Scheduler (S4) / continuous batching, FCFS / preemption  [label: new / requests]
- Arrow: Stop check + / streaming detokenizer / (S1) -> Output queues (S5) / one per request  [label: text chunks]

### Apple M4 Air (S0)

- Box: GPU (Metal) / ~120 GB/s memory bandwidth
- Box: Metal -> MLX -> mlx-lm -> Brigade / MLX is lazy: work runs at mx.eval / mlx-lm is a pip dependency, not a fork

### Inside the engine replica: Engine thread (S5): one loop, one step at a time; the only thread that touches MLX

- Box: Scheduler (S4) / continuous batching, FCFS / preemption
- Box: Model runner (S1) / Qwen3, 28 layers / mlx-lm model unchanged
- Box: Sampler (S1) / greedy, temperature / top-k, top-p
- Box: Block manager (S2-S3) / block pool, free list / block tables, ref counts / prefix hash table (S3) / LRU eviction (S3)
- Box: Paged KV cache (S2) / plug point / write into block, gather
- Box: Stop check + / streaming detokenizer / (S1)
- Arrow: Scheduler (S4) / continuous batching, FCFS / preemption -> Model runner (S1) / Qwen3, 28 layers / mlx-lm model unchanged  [label: batch / [B, L]]
- Arrow: Model runner (S1) / Qwen3, 28 layers / mlx-lm model unchanged -> Sampler (S1) / greedy, temperature / top-k, top-p  [label: logits]
- Arrow: Scheduler (S4) / continuous batching, FCFS / preemption <-> Block manager (S2-S3) ...
- Arrow: Model runner (S1) / Qwen3, 28 layers / mlx-lm model unchanged <-> Paged KV cache (S2) / plug point / write into block, gather
- Arrow: Sampler (S1) / greedy, temperature / top-k, top-p -> Stop check + / streaming detokenizer / (S1)  [label: 1 ID per / request]
- Arrow: Paged KV cache (S2) / plug point / write into block, gather <-> Block manager (S2-S3) ...  [label: block / tables]
- Note: can fit? allocate / append, free
- Note: new K,V in / all K,V out

### Inside the Apple M4 Air: Unified RAM, 16 GB

- Box: Weights ~1.2 GB (bf16) / read fully every step
- Box: KV block pool (~4 GB) / shared by all users
- Box: macOS + apps / real budget for weights / + KV: ~10-11 GB

---

## Panel 0: one request inside the model (S0)

- Circle: Text / "The cat sat"
- Box: Tokenizer / (CPU)
- Box: Token IDs / [785, 8251, 7578]
- Box: Prefill / all prompt tokens, 1 forward pass / 28 layers, compute-bound
- Box: Decode loop / last token -> forward pass -> next / memory-bandwidth-bound
- Box: KV cache / ~112 KB per token / K,V never change
- Box: Sampler / pick 1 token
- Decision: Stop? / EOS or / max_tokens
- Box: KV size per token / = 2 x layers x KV heads x head dim x bytes / = 2 x 28 x 8 x 128 x 2 = ~112 KB / 4k-token chat = ~450 MB
- Box: Software stack / Metal (Apple's GPU API, like CUDA) / -> MLX (arrays, lazy, fast attention) / -> mlx-lm (Qwen3 code, loader) / -> Brigade (scheduler, paged KV, API)
- Box: Batching / weights read ONCE per step, / shared by N users' tokens. / Each user's KV still read separately. / B=1 -> 1 token per weight read / B=8 -> 8 tokens per weight read
- Box: Done / finish_reason
- Arrow: Text / "The cat sat" -> Tokenizer / (CPU)
- Arrow: Tokenizer / (CPU) -> Token IDs / [785, 8251, 7578]
- Arrow: Token IDs / [785, 8251, 7578] -> Prefill / all prompt tokens, 1 forward pass / 28 layers, compute-bound
- Arrow: Prefill / all prompt tokens, 1 forward pass / 28 layers, compute-bound -> Decode loop / last token -> forward pass -> next / memory-bandwidth-bound
- Arrow: Prefill / all prompt tokens, 1 forward pass / 28 layers, compute-bound -> KV cache / ~112 KB per token / K,V never change
- Arrow: Decode loop / last token -> forward pass -> next / memory-bandwidth-bound -> Sampler / pick 1 token
- Arrow: Decode loop / last token -> forward pass -> next / memory-bandwidth-bound <-> KV cache / ~112 KB per token / K,V never change
- Arrow: Sampler / pick 1 token -> Stop? / EOS or / max_tokens
- Arrow: Stop? / EOS or / max_tokens -> Decode loop / last token -> forward pass -> next / memory-bandwidth-bound  [label: no: stream text, / feed token back / (each repeat = TPOT)]
- Arrow: Stop? / EOS or / max_tokens -> Done / finish_reason  [label: yes]
- Note: first token = TTFT
- Note: writes K,V for / every prompt token
- Note: reads all past K,V / appends 1 new
- Note: TTFT = wait for the first token   TPOT = time per later token   p50 = typical   p99 = worst 1% (SLOs use p99)

---

## Panel 0b: one token's journey through the hardware (S0)

- Box: 1 / request
- Box: 2 / tokenize
- Box: 3 / embed
- Box: 4 / stream / weights
- Box: 5 / attention / + MLP
- Box: 6 / layers / 2-28
- Box: 7 / output / scores
- Box: 8 / sample / (mx.eval)
- Box: 9 / ID back
- Box: 10 / stream / text
- Arrow: Weights ~1.2 GB / streamed to GPU every step -> On-chip cache / few MB, can't keep weights  (THICK RED arrow: ~1.2 GB every step, the bottleneck)
- Arrow: Tokenizer / steps 2 and 10 <-> Token IDs / vectors  [label: IDs / (bytes)]
- Arrow: Compute units / attention + MLP x 28 <-> KV cache / one per user  [label: past K,V in / new K,V out]
- Arrow: Sampler / scores -> 1 token ID -> Token IDs / vectors  [label: 4 bytes]
- Bracket (two-way arrow under steps 3 to 8): steps 3-8 repeat every decode step (= 1 TPOT)
- Note: TTFT = steps 1-10 for the first token (prefill: 3 prompt tokens go through steps 3-8 together)
- Note: Decode = steps 3-10 again with 1 token. Batched decode = same steps for B users' tokens,
- Note: weights streamed in once and shared; each user's KV read separately.
- Note: Only the thick red arrow is expensive. Every other arrow carries KB or bytes.

### CPU (Python)

- Box: API server / step 1: request in
- Box: Tokenizer / steps 2 and 10
- Box: SSE stream / step 10: text out
- Arrow: API server / step 1: request in -> Tokenizer / steps 2 and 10
- Arrow: Tokenizer / steps 2 and 10 -> SSE stream / step 10: text out

### Unified RAM

- Box: Weights ~1.2 GB / streamed to GPU every step
- Box: KV cache / one per user
- Box: Token IDs / vectors

### GPU

- Box: On-chip cache / few MB, can't keep weights
- Box: Compute units / attention + MLP x 28
- Box: Sampler / scores -> 1 token ID
- Arrow: On-chip cache / few MB, can't keep weights -> Compute units / attention + MLP x 28
- Arrow: Compute units / attention + MLP x 28 -> Sampler / scores -> 1 token ID

---

## Panel 1: model runner and sampler (S1)

### Startup lane: once per engine process, never per request

- Circle: Hugging Face / mlx-community/Qwen3
- Box: config.json / 28 layers, vocab 151,936
- Box: *.safetensors / weights ~1.2 GB (bf16)
- Box: Tokenizer files / vocabulary + chat template
- Arrow: Hugging Face / mlx-community/Qwen3 -> config.json / 28 layers, vocab 151,936
- Arrow: Hugging Face / mlx-community/Qwen3 -> *.safetensors / weights ~1.2 GB (bf16)
- Arrow: Hugging Face / mlx-community/Qwen3 -> Tokenizer files / vocabulary + chat template
- Note: loaded once with mlx_lm.load()

### Every engine step

- Decision: Stop?
- Circle: Messages / (JSON)
- Box: Chat template / messages -> 1 string
- Box: Tokenizer / string -> IDs
- Box: Model runner / forward pass, 28 layers / prefill: L = prompt length / decode: L = 1 (same box)
- Box: Take last / position
- Box: Sampler (per row) / greedy, temperature / top-k, top-p
- Box: Streaming detokenizer / one per request / holds half-characters
- Box: MLX is lazy: all GPU work runs at / mx.eval, once per step. / Time only after it.
- Box: Wrong template = silent quality loss. / Thinking mode set the same / for Brigade and every baseline.
- Circle: SSE out
- Box: KV cache / plug point -> paged
- Box: Correctness anchor (CI): greedy output / == mlx_lm.generate, token for token (dashed box)
- Box: finish_reason / stop / length
- Arrow: Stop? -> Model runner ...  [label: not finished: feed 1 new token back (L = 1)]
- Arrow: Messages / (JSON) -> Chat template / messages -> 1 string
- Arrow: Chat template / messages -> 1 string -> Tokenizer / string -> IDs  [label: 1 string]
- Arrow: Tokenizer / string -> IDs -> Model runner ...  [label: IDs / [B, L]]
- Arrow: Model runner ... -> Take last / position  [label: logits / [B, L, 151,936]]
- Arrow: Take last / position -> Sampler (per row) / greedy, temperature / top-k, top-p  [label: [B, 151,936]]
- Arrow: Sampler (per row) / greedy, temperature / top-k, top-p -> Stop?  [label: B IDs]
- Arrow: Stop? -> Streaming detokenizer / one per request / holds half-characters  [label: no]
- Arrow: Model runner ... <-> KV cache / plug point -> paged
- Arrow: Stop? -> finish_reason / stop / length
- Arrow: Streaming detokenizer / one per request / holds half-characters -> SSE out
- Note: new K,V in / all K,V out / (every layer)
- Note: text chunk
- Note: EOS -> stop / max_tokens -> length

---

## Panel 2: KV cache, contiguous then paged (S2)

### Before: contiguous cache

Three strips, one per request. Each strip is reserved for the maximum length; only the shaded start is actually used.

| Request | Shaded part | Rest of the strip |
|---|---|---|
| Req A | used (~25% of the strip) | reserved for max length, unused |
| Req B | used (~40% of the strip) | reserved for max length, unused |
| Req C | used (~16% of the strip) | reserved for max length, unused |

- Note: Problems: over-reserving, holes between strips (fragmentation), / copy on every growth. Older servers used only ~20-40% / of reserved KV memory for real tokens.

### After: paged cache

- Req A block table (40 tokens): cells `7`, `2`, `9` (purple) covering tokens 0-15, 16-31, 32-39
- Req B block table (20 tokens): cells `4`, `0` (teal) covering tokens 0-15, 16-19
- An arrow (->) points from the block tables to the pool: each table cell names a block in the pool, and the blocks are scattered, not side by side.
- Block pool (fixed at startup), drawn as a 4 x 3 grid (purple = request A, teal = request B, grey = free):

| | | | |
|---|---|---|---|
| Block 0 / B 4/16 | Block 1 / free | Block 2 / A full | Block 3 / free |
| Block 4 / B full | Block 5 / free | Block 6 / free | Block 7 / A full |
| Block 8 / free | Block 9 / A 8/16 | Block 10 / free | Block 11 / free |

- Box: Free list / 1, 3, 5, 6, 8, 10, 11
- Arrow: Free list -> Block pool  [label: allocate] (green)
- Arrow: Block pool -> Free list  [label: free on / finish] (red)
- Box: 1 block = 16 tokens / x 28 layers = ~1.8 MB / 4 GB pool = ~2,300 blocks / = ~36,000 tokens, / shared by all users / Waste: at most 15 slots per / request (its last block only)
- Note: Life cycle: prefill takes ceil(len / 16) blocks  ->  decode fills the last block, +1 block when full  ->  finish returns all blocks  ->  pool empty = preemption (S4)
- Note: Metrics -> Grafana / live terminal: KV utilization, memory waste %, max concurrent sequences, free blocks over time

### Component view (inside the engine thread)

- Box: Scheduler (S4) / always the one asking (dashed box)
- Box: Block manager / block pool, free list / block tables per request / ref counts (S3)
- Box: Model runner (S1) / doesn't know about / paging
- Box: Paged KV cache / (plug point) / writes new K,V into a block / gathers blocks into a / continuous strip
- Box: Fast attention / mx.fast.scaled_dot / _product_attention
- Box: Gather = extra copy / every step. Measured / and reported honestly. / Custom Metal paged- / attention kernel: out of scope (red)
- Arrow: Scheduler (S4) -> Block manager  [label: can fit? / allocate N / append 1 slot / free]
- Arrow: Model runner (S1) <-> Paged KV cache  [label: store / / give all]
- Arrow: Paged KV cache -> Fast attention  [label: strip]
- Arrow: Paged KV cache -> Block manager  [label: reads block tables]

---

## Panel 3: prefix caching (S3)

- Box: Prefix hash table / (inside block manager) / hash -> block number / h1 -> block 7 / h2 -> block 2 / h3 -> block 9 / one dictionary lookup / per block
- Box: Block states / Free: empty, on the free list / In use: ref count >= 1, / never evicted / Cached: ref count 0, K,V and / hash kept -> reused on a later / hit, or evicted LRU when the / pool runs out / Nothing left to evict: / preemption (S4)
- Arrow: h3 = hash(h2 + / tokens 32-47) -> Prefix hash table ...  [label: registered]

### The problem: repeated prompt beginnings

- Box: system prompt, 3,000 tokens
- Box: own question
- Box: system prompt, 3,000 tokens
- Box: own question
- Note: Req A
- Note: Req B
- Note: Without prefix caching, both requests prefill the / same 3,000 tokens and store two identical copies / of their K,V. Agents are worse: every turn / resends the whole conversation so far.

### Chained block hashes (one per FULL block)

- Box: Block 1 / tokens 0-15
- Box: Block 2 / tokens 16-31
- Box: Block 3 / tokens 32-47
- Box: Last block / partial: not hashed (dashed box)
- Box: h1 = hash( / tokens 0-15)
- Box: h2 = hash(h1 + / tokens 16-31)
- Box: h3 = hash(h2 + / tokens 32-47)
- Arrow: Block 1 / tokens 0-15 -> h1 = hash( / tokens 0-15)
- Arrow: Block 2 / tokens 16-31 -> h2 = hash(h1 + / tokens 16-31)
- Arrow: Block 3 / tokens 32-47 -> h3 = hash(h2 + / tokens 32-47)
- Arrow: h1 = hash( / tokens 0-15) -> h2 = hash(h1 + / tokens 16-31)
- Arrow: h2 = hash(h1 + / tokens 16-31) -> h3 = hash(h2 + / tokens 32-47)
- Note: Why chain? The same 16 tokens after a different history have / different K,V, so a block's hash must cover everything before it. / Same tokens, same order -> identical K,V -> safe to share.

### Admitting a new request (block manager, asked by the scheduler)

- Decision: Hash in / table?
- Box: Example: 3,000-token system prompt / + 50-token question -> 187 blocks hit, / prefill 58 tokens instead of 3,050. / If every block hits, still recompute the / last token to get the next-token scores.
- Box: Measure: prefix hit rate (hit tokens / / prompt tokens), TTFT cache on vs off / on workloads B and D, evictions. / CI: greedy output identical, on vs off. / Router (S8) reuses these hashes. (dashed box)
- Box: Prompt tokens / (after chat template)
- Box: Split into 16-token blocks, / chain-hash each full block
- Box: Hit: reuse that block / ref count +1 / add to block table
- Box: First miss: stop matching. / Prefill only the remaining / tokens, into new blocks
- Box: Register hashes of the / new full blocks / (other requests can hit)
- Box: Request finishes: / ref count -1; at 0 the / block becomes Cached
- Box: Gotcha: 1 changed / early token (a time- / stamp) = zero hits
- Arrow: Hit: reuse that block / ref count +1 / add to block table -> Hash in / table?  [label: check next block]
- Arrow: Prompt tokens / (after chat template) -> Split into 16-token blocks, / chain-hash each full block
- Arrow: Split into 16-token blocks, / chain-hash each full block -> Hash in / table?
- Arrow: Hash in / table? -> Hit: reuse that block / ref count +1 / add to block table  [label: yes]
- Arrow: Hash in / table? -> First miss: stop matching. / Prefill only the remaining / tokens, into new blocks  [label: no]
- Arrow: First miss: stop matching. / Prefill only the remaining / tokens, into new blocks -> Register hashes of the / new full blocks / (other requests can hit)
- Arrow: Register hashes of the / new full blocks / (other requests can hit) -> Request finishes: / ref count -1; at 0 the / block becomes Cached  [label: decode...]
- Note: Shared blocks are always full, so nobody writes / into them again. New tokens go into each / request's own last block. Sharing is read-only.

---

## Panel 4: scheduler (S4)

### Static vs continuous batching (time steps ->)

A timeline with steps 1 2 3 4 5 6 7 8 9 10 across the top. Each cell is one step: `run` = generating (orange), `idle` = slot held, already finished (grey dashed), `wait` = waiting to join (red), `pre` = prefill, `.` = not in the system.

Static

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Req A | run | run | run | idle | idle | idle | idle | idle | . | . |
| Req B | run | run | run | run | run | run | run | run | . | . |
| Req D | . | wait | wait | wait | wait | wait | wait | wait | pre | run |

Continuous

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Req A | run | run | run | . | . | . | . | . | . | . |
| Req B | run | run | run | run | run | run | run | run | . | . |
| Req D | . | wait | pre | run | run | run | run | run | run | run |

- Note: orange = generating   grey dashed = slot held, already finished   red = waiting to join / D arrives at step 2. Static: waits for the whole batch. Continuous: joins next step.

### Request states

- Box: Waiting / FCFS queue
- Box: Running / in the batch
- Box: Finished / blocks freed
- Box: Preempted / newest running
- Arrow: Waiting -> Running  [label: admit]
- Arrow: Running -> Finished  [label: done]
- Arrow: Running -> Preempted  [label: pool empty, / nothing to evict]
- Arrow: Preempted -> Waiting  [label: back to the FRONT / re-prefill later]
- Note: Preemption arrives together with the block manager.

### Admission limits

- Box: Admission limits (checked per waiting request) / max_num_seqs: batch-size cap (start at 16) / prefill token budget per step (start at 2,048) / enough free blocks for its prompt / (after prefix hits) + a small safety watermark / Starting values, not calculated optima: / tuned from the concurrency sweep (1, 2, 4, 8, 16). / Head of the queue doesn't fit -> stop (FCFS). / First: batch-size cap only. Then: blocks + preemption. / Policy is pluggable: FCFS now, fair share later.

### One engine step, explained with a running example

Two columns. Left column header: "What the scheduler does". Right column header: "Example   (before the step: Running = A, B, C    Waiting = D, E)". Each step box on the left has a dashed arrow to its example on the right. Steps run top to bottom, and an arrow loops from step 5 back to step 1 [label: back to step 1: one trip round = one engine step, repeated forever].

1. Box: 1. Remove finished requests / who: any request that hit EOS / or max_tokens / do: give its blocks back, send 'done'
   - Example: Running [A] [B] [C done] -> Running [A] [B]
   - Note: C produced EOS. Its blocks go back to the pool (blocks shared with others / stay Cached), and C's client gets finish_reason: stop.
2. Box: 2. Make room for running requests / each one will write 1 more token, / so its last block needs a free slot
   - Example: [A] has blocks full, full, last-partly-filled -> "last block still has room -> OK"
   - Example: [B] has blocks full, full, full -> gets a new block [+1] -> "last block full -> gets 1 new block"
   - Box (red): No free block in the pool? / 1. reuse a Cached block (S3) / 2. none left: preempt the / newest request (see right)
3. Box: 3. Let waiting requests in / oldest first, one by one, / while they fit; stop at the first / one that doesn't fit
   - Example: Waiting [D] [E] -> Running [A] [B] [D], Waiting [E]
   - Note: D's prompt fits in the free blocks -> D joins. E doesn't fit -> E stays at the / front, and nobody behind E may skip ahead (that is what FCFS means).
4. Box: 4. Choose the step type / did anyone new join in step 3?
   - Box (blue, highlighted): YES (this example) -> PREFILL step / run D's whole prompt: input [1, prompt length] / A and B wait for this one step
   - Box (dashed): NO -> DECODE step / every running request gets 1 token / input [B, 1], e.g. [3, 1] for A, B, D
5. Box: 5. Run the model / forward pass -> sampler -> stop check, / stream each new token to its user
   - Note: D gets its first token: that moment is D's TTFT. / Next loop: nobody new joins -> decode step for A, B, D, input [3, 1]. / Every decode loop gives each of them 1 more token: one TPOT.

#### Zoom-in: preemption (pool of 10 blocks)

The pool is drawn as a row of 10 squares, labelled by owner.

- 1. Pool is full
  - Pool: [A] [A] [A] [A] [B] [B] [B] [C] [C] [C]
  - Note: A's last block is full and A needs 1 more. / No free block, nothing Cached to reuse.
- 2. Pause the newest request: C
  - Pool: [A] [A] [A] [A] [B] [B] [B] [free] [free] [free]
  - Waiting: [C] [E]   <- C goes to the FRONT
  - Note: C's blocks are released. C keeps its prompt + every / token it already streamed (nothing is lost).
- 3. A continues
  - Pool: [A] [A] [A] [A] [A] [B] [B] [B] [free] [free]
  - Note: A gets one of the freed blocks. A and B keep generating.
- 4. Later: C comes back
  - Note: B finishes -> its blocks are freed -> C is let back in. / C re-prefills its prompt + its old tokens to rebuild its K,V / (often via prefix-cache hits), then keeps writing. / The user only sees a short pause. Nothing is repeated.
- Box: Why it's needed: admission only checks the prompt fits. / Nobody knows how long the answer will be, so memory can / run out mid-generation. Preemption is the safety valve. / Why the newest: least work to redo, oldest users keep moving. / Cost: C's re-prefill. Many preemptions = pool too small / or batch cap too high (tracked as a metric).

### Notes

- Box: Padding + mask (GPU math needs rectangles) / Prefill: token rows padded to the longest prompt; / waste hits every layer (attention + MLP). / Decode: input is [B, 1], no padding needed. / Only the gathered KV (different histories) is / padded, and only attention sees it. / Mask = padded positions get zero attention weight.
- Box: Prefill-first trade-off / While a prefill step runs, every running request / skips that step, so its next token waits. / Workload C (one 8k prompt): TPOT p99 spikes and / short requests queue behind it (head-of-line / blocking). Fix later: chunked prefill.
- Box: Preemption = recompute / Free the victim's blocks; its full blocks stay / Cached, so re-prefill often hits the prefix cache. / It goes to the FRONT of the queue and later / re-prefills prompt + tokens already sent. / No swap to CPU: unified memory is the same RAM.
- Box: Measure + test / Metrics: waiting queue length, batch size per / step, step time, preemptions, tokens per step. / Tests: every request finishes, FCFS order kept, / zero leaked blocks at the end, greedy output / same at batch 1 and batch 8 (check near-ties). (dashed box)

---

## Panel 5: engine loop and concurrency (S5)

- Note: Follow one request: user A asks 'What is 2+2?'   (time flows left to right; each band = who does the work)

### Swimlanes: one request crossing the threads

Four horizontal bands, top to bottom: "Client" | "API thread / (asyncio)" | "Bridge / (queues)" | "Engine thread / (owns MLX)". Each step box sits in the band of whoever does that step; time runs left to right.

| Step | Band | Box text |
|---|---|---|
| 1 | Client | 1. POST /v1/chat/ / completions / 'What is 2+2?' |
| 2 | API thread | 2. Handler starts / id = req-42, plus / its own output queue |
| 3 | API thread | 3. Chat template / + tokenize / (zoom below) |
| 4 | Bridge | 4. Ticket on the / inbound queue / req-42, IDs, settings |
| 5 | API thread | 5. Handler awaits / its output queue / (sleeps, costs ~0) |
| 6 | Engine thread | 6. Next step starts: / drain inbound, / req-42 -> Waiting |
| 7 | Engine thread | 7. Scheduler admits / req-42: this step / is its prefill |
| 8 | Engine thread | 8. Model + sampler / on the GPU / -> first token '2' |
| 9 | Bridge | 9. Chunk '2' into / req-42's output / queue |
| 10 | API thread | 10. Handler wakes, / sends SSE event / data: '2' |
| 11 | Client | 11. User sees '2' / (that moment = TTFT) |
| 12 | Engine thread | 12. Decode steps: / 8-11 repeat per token / ' +' ' 2' ' =' ' 4' |
| 13 | Engine thread | 13. EOS (<|im_end|>) / stop -> [DONE] sent, / blocks freed |

Arrows:
- 1 -> 2
- 2 -> 3
- 3 -> 4  [label: put]
- 4 -> 5  [label: then wait]
- 4 -> 6  [label: drained when the / engine's current / step finishes]
- 5 -> 10  (dashed)  [label: the handler sleeps here; the API thread serves other users meanwhile]
- 6 -> 7 -> 8
- 8 -> 9  [label: push]
- 9 -> 10
- 10 -> 11  [label: SSE]
- 8 -> 12, and 12 -> 8  [label: repeat per token]
- 12 -> 13

### Notes

- Box: Why two threads? / A step blocks for ~15 ms (decode) up to 100s of ms (prefill). / Inside the asyncio thread nothing else could run: no new / requests, no streaming, /health times out, router marks it dead. (red)
- Box: Why only one thread touches MLX / One GPU, so extra threads would just queue up on it, and / they would race on block tables and the KV pool. / One owner = no locks, no races.
- Box: Client disconnects mid-answer / API notices -> abort ticket -> engine frees the blocks next / step. Otherwise dead requests burn GPU until max_tokens. / Full inbound queue -> reply 503 (backpressure).
- Box: Measure + test / Metrics: queue depth, time in queue (part of TTFT), idle %. / Tests: 50 clients at once all finish, no mixed-up streams, / aborts free every block. GIL check: /health stays fast in C. (dashed box)

### Zoom: step 3, chat template + tokenize

- Box:

  ```
  [{"role": "user",
    "content": "What is 2+2?"}]
  ```

- Box:

  ```
  <|im_start|>user
  What is 2+2?<|im_end|>
  <|im_start|>assistant
  ```

- Box: token IDs: one number per text piece
- Arrow: [{"role": "user", / "content": "What is 2+2?"}] -> <|im_start|>user / What is 2+2?<|im_end|> / <|im_start|>assistant
- Arrow: <|im_start|>user / What is 2+2?<|im_end|> / <|im_start|>assistant -> token IDs: one number per text piece
- Note: JSON with roles: the model / can't read this directly.
- Note: chat template
- Note: One string in the exact format / the model was trained on. / Markers show who is speaking; / the open assistant line means / 'your turn to write'.
- Note: tokenizer
- Note: Only these numbers go on / the ticket.

### Zoom: steps 4 and 6, inbound queue vs waiting queue

- Box: API thread / puts tickets
- Box: Inbound queue / mailbox: BOTH threads / touch it (thread-safe)
- Box: Waiting queue / the scheduler's own list: / ONLY the engine touches it
- Box: ticket: req-42 / prompt IDs, max_tokens, / temperature, output queue (dashed box)
- Box: ticket: abort req-17 (dashed box)
- Arrow: API thread / puts tickets -> Inbound queue / mailbox: BOTH threads / touch it (thread-safe)
- Arrow: Inbound queue / mailbox: BOTH threads / touch it (thread-safe) -> Waiting queue / the scheduler's own list: / ONLY the engine touches it
- Note: drain at step start
- Note: Preempted requests go back / to the FRONT of this list. / The scheduler admits from / here, oldest first (FCFS).
- Note: Why two queues? If the API thread wrote into the waiting queue directly, / two threads would share scheduler state -> locks and races.

### Zoom: the engine loop (every step, forever)

- Box: 1. Drain the inbound queue (never waits)
- Box: 2. Nothing to do? Sleep until a ticket arrives
- Box: 3. Scheduler step (Panel 4)
- Box: 4. Model + sampler on the GPU (Panel 1)
- Box: 5. Push chunks out, publish telemetry
- Arrow: 5. Push chunks out, publish telemetry -> 1. Drain the inbound queue (never waits)
- Arrow: 1. Drain the inbound queue (never waits) -> 2. Nothing to do? Sleep until a ticket arrives
- Arrow: 2. Nothing to do? Sleep until a ticket arrives -> 3. Scheduler step (Panel 4)
- Arrow: 3. Scheduler step (Panel 4) -> 4. Model + sampler on the GPU (Panel 1)
- Arrow: 4. Model + sampler on the GPU (Panel 1) -> 5. Push chunks out, publish telemetry
- Note: One owner: the / scheduler, blocks, / KV pool and model / are touched only / here, so no locks.
- Note: Step 2 avoids busy-spinning: an idle loop would only heat a fanless Air.

---

## Panel 6: OpenAI-compatible API server (S6)

- Box: SSE in one line / The HTTP response stays open, and the server / writes 'data: ...' lines as tokens arrive. / One-way, plain HTTP, works through proxies: / it's what OpenAI uses, no WebSocket needed.
- Box: usage = exact token counts / Counted by the engine, never estimated. / Needed for benchmarks now and a billing / ledger later. Bonus: report prefix-cache hits / as prompt_tokens_details.cached_tokens.
- Box: Errors, in OpenAI's error format / 400: bad request (missing messages, bad / values, prompt longer than the context) / 404: unknown model   503: queue full, retry / Mid-stream failure: send an error event, close.
- Box: Prove compatibility / Contract test in CI: the official OpenAI SDK / calls Brigade, streaming and not. / The same load generator hits Brigade, mlx-lm, / Ollama and vllm-metal: a fair comparison. (dashed box)

### Who calls it, and on which endpoint

- Box: Callers / OpenAI Python SDK / OpenCode (agent) / curl, load generator / Go router (S8)
- Box: POST /v1/chat/completions / the main one
- Box: GET /v1/models / lists 'qwen3-0.6b'
- Box: GET /health / the router checks it (S8)
- Box: GET /metrics / Prometheus scrapes it (S9)
- Arrow: Callers ... -> POST /v1/chat/completions / the main one
- Arrow: Callers ... -> GET /v1/models / lists 'qwen3-0.6b'
- Arrow: Callers ... -> GET /health / the router checks it (S8)
- Arrow: Callers ... -> GET /metrics / Prometheus scrapes it (S9)
- Note: The only change a client / needs: base_url = / localhost:8000/v1

### The request: where each field goes

- Box:

  ```
  {
    "model": "qwen3-0.6b",
    "messages": [ ... ],
    "stream": true,
    "max_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.95,
    "stop": ["\n\n"],
    "tools": [ ... ],
    "stream_options":
        {"include_usage": true}
  }
  ```

- Box: messages -> chat template (S1)
- Box: max_tokens, stop -> stop check (S1)
- Box: temperature, top_p -> sampler (S1)
- Box: stream -> which reply mode (right)
- Box: tools -> tool calling (S7) (dashed box)
- Box: model -> must be the loaded one, / else error 404
- Arrow: { "model": "qwen3-0.6b", ... -> messages -> chat template (S1)
- Arrow: { "model": "qwen3-0.6b", ... -> max_tokens, stop -> stop check (S1)
- Arrow: { "model": "qwen3-0.6b", ... -> temperature, top_p -> sampler (S1)
- Arrow: { "model": "qwen3-0.6b", ... -> stream -> which reply mode (right)
- Arrow: { "model": "qwen3-0.6b", ... -> tools -> tool calling (S7)
- Arrow: { "model": "qwen3-0.6b", ... -> model -> must be the loaded one, / else error 404

### The reply: two modes

- Box:

  ```
  stream: false
  ONE JSON at the very end:
  
  {"choices": [{
     "message": {
       "content": "2 + 2 = 4."},
     "finish_reason": "stop"}],
   "usage": {
     "prompt_tokens": 14,
     "completion_tokens": 8,
     "total_tokens": 22}}
  ```

- Box: data: {"delta": {"role": "assistant"}}
- Box: data: {"delta": {"content": "2"}}
- Box: data: {"delta": {"content": " +"}}
- Box: ...   one event per token   ... (dashed box)
- Box: data: {"delta": {}, "finish_reason": "stop"}
- Box: data: {"choices": [], "usage": {...}}
- Box: data: [DONE]
- Note: stream: true -> SSE events, one by one
- Note: first token event = TTFT; usage only if include_usage

---

## Panel 7: tool calling (S7)

- Note: Follow one tool call: 'What's the weather in Pune?'   Round 1 (steps 1-6): the model asks for a tool.   Round 2 (steps 7-11): it uses the result.

### Swimlanes: one tool call, two rounds

Three horizontal bands, top to bottom: "Client / (runs tools)" | "API server / (template + / parser)" | "Engine / (only writes / text)". Time runs left to right; a dashed vertical line between steps 6 and 7 separates round 1 from round 2. Each step's arrow leads to the next step (1 -> 2 -> ... -> 11).

| Step | Band | Box text |
|---|---|---|
| 1 | Client | 1. POST + tools: / get_weather(city) / 'Weather in Pune?' |
| 2 | API server | 2. Chat template / writes the tool list / into the prompt |
| 3 | Engine | 3. Model writes / <tool_call> {name, / arguments} </tool_call> |
| 4 | API server | 4. Parser finds / <tool_call>, checks / JSON, name, args |
| 5 | API server | 5. Reply: tool_calls / finish_reason: / 'tool_calls' |
| 6 | Client | 6. Client runs / get_weather('Pune') / -> '31°C, sunny' |
| 7 | Client | 7. POST again: / history + tool call / + role 'tool' result |
| 8 | API server | 8. Template wraps / the result; round 1's / prefix hits the cache |
| 9 | Engine | 9. Model writes: / 'It's 31°C and / sunny in Pune.' |
| 10 | API server | 10. Normal reply / finish_reason: / 'stop' |
| 11 | Client | 11. User sees / the answer |

- Note (in the Engine band): The engine never runs tools: it only writes text.

### Notes

- Box: The engine never runs tools / It only writes text. The CLIENT (your app, / OpenCode) runs the tool and sends the result / back in a new request.
- Box: tool_choice / 'auto': the model decides (default). 'none': ignore tools. / 'required' or a named tool needs guaranteed output: / return 400 until JSON-schema output exists.
- Box: Agents = many rounds / Every round resends the whole history, so round N / shares its prefix with round N-1. Prefix caching (S3) / keeps TTFT low: that's workload D.
- Box: Plan + tests / Built after paged KV and the router. / Tests: parser on good and broken samples, an SDK / round trip with a fake tool, the tool-call eval. (dashed box)

### Zoom: steps 2-3, what the model sees and writes (simplified)

- Box:

  ```
  <|im_start|>system
  # Tools   (written in by the chat template)
  <tools>
  {"name": "get_weather", "parameters": {"city": "string"}}
  </tools>
  To call one, reply inside <tool_call></tool_call> tags.<|im_end|>
  <|im_start|>user
  What's the weather in Pune?<|im_end|>
  <|im_start|>assistant
  ```

- Box:

  ```
  <tool_call>
  {"name": "get_weather", "arguments": {"city": "Pune"}}
  </tool_call>
  ```

- Arrow: <|im_start|>system ... -> <tool_call> / {"name": "get_weather", "arguments": {"city": "Pune"}} / </tool_call>
- Note: model writes

### Zoom: step 4, the parser turns text into OpenAI's format

- Box:

  ```
  model text
  <tool_call>{"name": ...,
  "arguments": {"city":
  "Pune"}}</tool_call>
  ```

- Box:

  ```
  OpenAI format
  "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Pune\"}"}}]
  ```

- Arrow: model text ... -> OpenAI format ...  [label: parse / + check]
- Note: While streaming: normal text goes out as usual. From '<tool_call>' on, / hold the text back; at '</tool_call>' send one tool_calls event. / Never stream the raw tag to the client.
- Note: arguments is a JSON *string*. Several <tool_call> blocks = several calls.

### Zoom: when the model gets it wrong

- Box: Broken JSON (a missing brace)
- Box: Unknown tool name
- Box: A required argument is missing
- Box: Tag opened, never closed (max_tokens)
- Note: Never crash. / Return the text as / normal content and / count the failure. / The tool-call eval / (S11) = % of calls / that parse and match / the schema.
- Note: Smaller or 4-bit models fail more often: that's / why this eval runs for bf16, 8-bit and 4-bit.

---

## Panel 8: Go router (S8)

- Box: Why Go? / Built for network servers: goroutines handle / thousands of connections cheaply, and the standard / library has a reverse proxy. A separate process, / so it stays up when an engine crashes.
- Box: Honest framing: one Mac / The replicas share one GPU and 16 GB, so this / shows routing and failover, not more raw speed. / On a real cluster each replica is its own machine.
- Box: Correction to Section 3 / The router has no tokenizer, so it fingerprints / message text, not the engine's token blocks. / Same idea (chained hashes), coarser grain.
- Box: Plan, metrics, tests / First: proxy, round-robin, health, timeouts, retries. / Then: prefix-aware routing vs round-robin. / Metrics: load per replica, decisions, retries, breaker. / Test: kill a replica mid-run; no request is lost. (dashed box)

### Structure: one front door, three replicas

- Circle: Clients
- Box: Go router / :8000
- Box: Replica 1 / :8001 / cache: empty
- Box: Replica 2 / :8002 / cache: prompt S
- Box: Replica 3 / :8003 / DOWN (health) (dashed box)
- Arrow: Clients <-> Go router / :8000
- Arrow: Go router / :8000 -> Replica 1 / :8001 / cache: empty
- Arrow: Go router / :8000 -> Replica 2 / :8002 / cache: prompt S
- Arrow: Go router / :8000 -> Replica 3 / :8003 / DOWN (health)  (dashed)
- Note: Each replica is a full / engine process: API + / engine + its own KV / cache. Caches are NOT / shared between them.
- Note: One Mac = one GPU: the / replicas share it, so / throughput may not grow. / Measure, report honestly.
- Note: use 0.6B for these runs

### Routing decision (every request)

- Decision: Longest match / in the table?
- Box: Fingerprint the messages: / chained hash per message
- Box: Least-loaded healthy / replica (fewest in flight)
- Decision: Owner healthy / and not / overloaded?
- Box: Send to the owner / (its cache is warm)
- Box: Record its hashes, / forward + stream back
- Arrow: Fingerprint the messages: / chained hash per message -> Longest match / in the table?
- Arrow: Longest match / in the table? -> Least-loaded healthy / replica (fewest in flight)  [label: no]
- Arrow: Least-loaded healthy / replica (fewest in flight) -> Record its hashes, / forward + stream back
- Arrow: Owner healthy / and not / overloaded? -> Least-loaded healthy / replica (fewest in flight)  [label: no]
- Arrow: Longest match / in the table? -> Owner healthy / and not / overloaded?  [label: yes]
- Arrow: Owner healthy / and not / overloaded? -> Send to the owner / (its cache is warm)  [label: yes]
- Arrow: Send to the owner / (its cache is warm) -> Record its hashes, / forward + stream back
- Note: Turn N of a chat contains / turn N-1 exactly, so its / longest match is the last / turn's replica. / The table maps hash -> / replica, capped by LRU.
- Note: The router has no tokenizer: / it hashes message text, / not token blocks.

### Example: round-robin vs prefix-aware

- Box: 1. system prompt S / + question 1
- Box: R1: miss
- Box: R2: miss / (first time)
- Box: 2. S + question 2
- Box: R2: miss / (S recomputed)
- Box: R2: hit on S
- Box: 3. turn 2 of request 1 / (agent follow-up)
- Box: R3: miss / (turn 1 recomputed)
- Box: R2: hit on all / of turn 1
- Note: Request
- Note: Round-robin
- Note: Prefix-aware
- Note: Measure: prefix hit rate and TTFT, router vs / round-robin, on workloads B and D. / Catch: one popular system prompt would pile onto / one replica, hence the 'not overloaded' check.

### One forward, when things go wrong

- Box: Pick R2 / (decision above)
- Box: Connect fails, or / 503 'queue full'
- Box: Retry on R3 / (nothing sent / yet: safe)
- Box: Stream tokens, / flush each chunk / immediately
- Box: Client leaves: / cancel upstream / -> engine aborts (S5)
- Box: The retry rule: retry ONLY before the first byte reaches the client. / Once tokens are streamed, a retry would repeat text the user already saw, / so a mid-stream failure becomes an error event instead (S6). / Timeouts (starting values): connect ~1 s, first token ~30 s, gap between chunks ~30 s. / No total timeout: a long answer is not a failure.
- Arrow: Pick R2 / (decision above) -> Connect fails, or / 503 'queue full'
- Arrow: Connect fails, or / 503 'queue full' -> Retry on R3 / (nothing sent / yet: safe)
- Arrow: Retry on R3 / (nothing sent / yet: safe) -> Stream tokens, / flush each chunk / immediately
- Arrow: Stream tokens, / flush each chunk / immediately -> Client leaves: / cancel upstream / -> engine aborts (S5)

### Two ways to notice a broken replica

- Box: Health checks (in the background) / GET /health every ~2 s / 3 misses -> mark DOWN, stop routing / passes again -> back in rotation
- Box: Closed / (normal)
- Box: Open / (blocked)
- Box: Half-open / (1 trial)
- Arrow: Closed / (normal) -> Open / (blocked)  [label: 5 fails / in a row]
- Arrow: Open / (blocked) -> Half-open / (1 trial)  [label: cool-down / ~10 s]
- Arrow: Half-open / (1 trial) -> Closed / (normal)  (green, lower arrow: trial succeeds)
- Note: Half-open lets ONE trial request through. / It fails -> Open again (upper arrow). / It succeeds -> Closed (lower arrow). / Reacts to real traffic instantly.
- Arrow: Half-open / (1 trial) -> Open / (blocked)  (red, upper arrow: trial fails)
- Note: Health checks catch a dead or hung / process. The breaker catches a / replica that passes /health but / keeps failing real requests.

---

## Panel 9: metrics and tracing (S9)

### Native on the Mac (has the GPU)

- Box: Go router / :8000 / /metrics + OTel
- Box: Replica 1 / :8001 / /metrics + OTel
- Box: Replica 2 / :8002 / /metrics + OTel
- Box: Replica 3 / :8003 / /metrics + OTel
- Note: Every process exposes GET /metrics (numbers) and sends spans (timelines). / Updating a metric is a cheap counter bump; observability must never / block the engine thread (S5).

### Docker compose (a Linux VM: no Metal GPU)

- Box: Prometheus / pulls /metrics every 5 s, / stores time series
- Box: Jaeger / receives spans (OTLP), / shows request timelines
- Box: Grafana / dashboards that read / Prometheus (+ Jaeger)
- Box: Why the engines are NOT / in Docker: on a Mac, Docker / runs a Linux VM that can't / reach the Metal GPU, so MLX / would crawl on the CPU. / Prometheus reaches them at / host.docker.internal:8001-3. (red)
- Arrow: Prometheus -> [Native on the Mac]  [label: scrape / (pull)]
- Arrow: [Native on the Mac] -> Jaeger  [label: spans / (push)]
- Arrow: Prometheus -> Grafana
- Arrow: Jaeger -> Grafana

### Three kinds of metric

- Box: Counter: only goes up / requests_total, generation_tokens_total, preemptions_total / -> rate() turns it into 'per second'
- Box: Gauge: goes up and down / queue_depth, batch_size, kv_blocks_used
- Box: Histogram: counts per time bucket / ttft_seconds, tpot_seconds, step_seconds / -> histogram_quantile() gives p50 / p99 / (an average would hide the tail, S0)

### What Brigade measures

- Box: Engine (per replica) / TTFT / TPOT / end-to-end (histograms) / step time, labelled prefill | decode / batch size, queue depth, time in queue / prompt + generated tokens / prefix-cache hit tokens -> hit rate / KV blocks used / free, waste % / preemptions, aborts / finish_reason counts
- Box: Router / requests per replica / decision: prefix | least-load | rr / retries, upstream errors / replica healthy (0/1) / breaker state
- Box: Labels: replica, kind, reason. / NEVER a request ID as a label: / millions of series, Prometheus dies. (red)

### One trace = one request's timeline

Nested span bars on a time axis (an arrow labelled "time" runs left to right). Start and end are % of the request's total time:

| Span | Starts | Ends |
|---|---|---|
| router.request | 0% | 100% |
| router.forward (replica 2) | 2% | 99% |
| api.request | 4% | 98% |
| queue.wait | 6% | 15% |
| engine.prefill | 15% | 25% |
| engine.decode (212 tokens) | 25% | 96% |

- Note: Spans nest: parent -> child. The 'traceparent' header carries the trace ID / from Go into Python, and the ticket carries it across threads (S5).

### Notes

- Box: Spans per phase, not per token / A span per token = thousands per request: / slow and unreadable. Put token counts on / the decode span as attributes instead.
- Box: One Grafana screen / throughput, TTFT + TPOT p50/p99, batch size, / queue depth, KV usage, prefix hit rate, / preemptions, load per replica, breaker state.
- Box: Server vs client numbers / Server TTFT starts when the request arrives; / client TTFT (S12) adds network + router. / Cross-check: a big gap = a bug or a queue.
- Box: Plan + tests / First: /metrics + Grafana (docker compose). / Then OpenTelemetry spans, router + engine. / Test: counters match the load generator. (dashed box)

---

## Panel 10: live terminal (S10)

### Data path: engine -> browser

- Box: Engine thread / after every step: / emit events (S5)
- Box: Telemetry bus / bounded queue; if full, / drop the oldest event
- Box: SSE /telemetry/stream / (API thread) batches / events ~15 times/sec
- Box: Browser / React app / (canvas for drawing)
- Arrow: Engine thread -> Telemetry bus -> SSE /telemetry/stream -> Browser
- Note: -> request feed / -> batch view / -> KV block grid
- Note: Events: arrived, admitted, step {kind, rows, ms}, finished {reason}, preempted, KV snapshot (1 byte per block).

### Grafana vs live terminal

- Box: Grafana (S9) / trends over minutes / all replicas, aggregated / updates every ~5 s / 'how is it doing?'
- Box: Live terminal (S10) / individual requests + blocks / one replica, right now / updates ~15 times/sec / 'what is it doing?'

### View 1: request feed

A mockup table; the state column is a coloured pill.

| id | state | tokens | TTFT |
|---|---|---|---|
| req-41 | finished (green) | 300 | 210 ms |
| req-42 | running (orange) | 87 | 180 ms |
| req-43 | running (orange) | 12 | 950 ms |
| req-44 | preempted (brown) | 40 | 190 ms |
| req-45 | waiting (red) | - | - |
| req-46 | waiting (red) | - | - |

### View 2: batch view, one column per step

A mockup grid: rows = requests A to E, columns = 14 engine steps. `d` = one decode token (orange cell), `P` = prefill (blue cell), `.` = empty (faint).

```
     1 2 3 4 5 6 7 8 9 10 11 12 13 14
A    d d d d d d . . . .  .  .  .  .
B    d d d d d d . d d d  .  d  d  d
C    d d d d d d . d d d  .  d  d  d
D    . . . . . . P d d d  .  d  d  d
E    . . . . . . . . . .  P  d  d  d
```

- Note: orange = 1 decode token, blue P = prefill. Requests join and leave between steps. / Empty cells for B and C in steps 7 and 11 = the prefill-first stall (S4): a long / prompt (E) makes everyone else wait for that step.

### View 3: KV block grid, one square per block

A mockup grid of 72 squares (12 per row, 6 rows). `A`, `B`, `C` = owned by that request; `S` = shared prefix block (thick border); `K` = Cached block (dashed); `.` = free (grey).

```
S S S S S S S S A A B K
C C A . K B B C . A C K
. . C B A K K . C . B .
. K K A C . . B . C K .
A . . C . . . K . . B .
. . . A . . . . . K . .
```

- Note: A, B, C: / owned by / a request / S (thick): / shared prefix, / ref count > 1 / K (dashed): / Cached, ref 0, / evictable (S3) / grey: free
- Note: Watch: blocks scatter (paging, S2), S stays put while users share it, / K squares get evicted under pressure, a preempted request's squares vanish.

### Notes

- Box: Throttle, don't flood / The engine runs up to ~60 steps/s; people read a / few updates/s. Batch events ~15x/s; a KV snapshot / is 1 byte per block (~2,300 bytes).
- Box: Why canvas, not plain React / 2,300 squares re-rendered 15x/s as page elements / is slow. React for layout and controls, canvas / to paint the grid and the batch strip.
- Box: Never slow the engine, and prove it / Bounded bus, drops the oldest event; closed tab = 0 cost. / Benchmark with telemetry on vs off and report the / overhead (expect ~1%). SSE, not WebSocket: one-way.
- Box: Plan + demo / First: request feed + batch view. Then: KV grid. / README GIF: workload B (shared S squares) and / C (the wide prefill stall). Charts stay in Grafana. (dashed box)

---

## Panel 11: testing and evals (S11)

- Box: Rules for honest evals / Fixed datasets checked into the repo, greedy / decoding, the same template + thinking setting. / Never tune prompts on the eval set itself.
- Box: Never cut the quantization evals / They count for BOTH AI roles (evals: 22 of 30 / postings) and inference roles (quantization / with quality checks: ~9 of 29).
- Box: The near-tie caveat / Batching regroups bf16 math, so a nearly tied / top-2 pick can flip and the texts diverge. / Log the margin; explain it in DESIGN.md.
- Box: Plan / 1. CI, unit tests, greedy equivalence. / 2. Property tests with the block manager. / 3. Quantization + tool-call evals, CI gate. (dashed box)

### Tests vs evals, and the test layers

- Box: Tests: is the CODE correct? / pass / fail, same result every run
- Box: Evals: is the OUTPUT good enough? / a score, compared to a threshold
- Box: Integration: SDK, 50 clients, router
- Box: Output equivalence vs mlx-lm (greedy)
- Box: Property tests: block-manager invariants (hypothesis)
- Box: Unit tests: sampler, stop check, template, tool parser, SSE format
- The four test layers are drawn as a pyramid, top to bottom: Integration (top, narrowest), Output equivalence, Property tests, Unit tests (bottom, widest).
- Note: few + slow
- Note: many + fast / at the bottom

### Zoom: property tests on the block manager

- Box: hypothesis generates / random operation lists: / allocate(A, 3 blocks) / share(B, A's prefix) / append(A) / preempt(B) / free(A) / evict() / ... thousands of lists
- Box: After EVERY op, check: / no block owned twice / ref counts >= 0 / free + used + cached / = total (no leaks) / ref >= 1: never on the / free list or evicted / freed blocks reusable
- Box: Block / manager
- Arrow: hypothesis generates ... -> Block / manager
- Arrow: Block / manager -> After EVERY op, check: ...
- Note: On a failure, hypothesis SHRINKS the list to the shortest one that still / breaks, e.g. 'share, preempt, free': a ready-made bug report.

### Zoom: output equivalence (the correctness anchor)

- Box: mlx_lm.generate
- Decision: Same token / IDs?
- Box: same prompt, / greedy
- Box: Brigade: batched, / paged, prefix cache, / preempted + resumed
- Box: Differ: were the top-2 scores nearly tied / (bf16 rounding inside a batch)? / yes -> log the margin; no -> real bug, FAIL
- Box: pass
- Arrow: same prompt, / greedy -> mlx_lm.generate
- Arrow: mlx_lm.generate -> Same token / IDs?
- Arrow: same prompt, / greedy -> Brigade: batched, / paged, prefix cache, / preempted + resumed
- Arrow: Brigade: batched, / paged, prefix cache, / preempted + resumed -> Same token / IDs?
- Arrow: Same token / IDs? -> Differ: were the top-2 scores nearly tied ...  [label: no]
- Arrow: Same token / IDs? -> pass  [label: yes]
- Note: Runs for: 1 request, many at once, prefix cache on vs off, / paged vs contiguous, and a forced preemption.

### Evals: the quantization trade-off table (filled in by measuring)

`?` = to be measured. Weight sizes are for Qwen3-0.6B.

| | perplexity | exact match | tool calls valid | tokens / s | weights (0.6B) |
|---|---|---|---|---|---|
| bf16 | ? | ? | ? | ? | ~1.2 GB |
| 8-bit | ? | ? | ? | ? | ~0.6 GB |
| 4-bit | ? | ? | ? | ? | ~0.3 GB |

- Note: perplexity: how surprised the model is by a fixed real text (lower = better; no generation needed). / exact match: 100-200 questions with one right answer, greedy. Same template + thinking setting everywhere.

### CI gate: on every push (GitHub Actions, macOS arm64 runner)

- Box: 1. Lint + / type check
- Box: 2. Unit + / property tests
- Box: 3. Greedy equiv. / 0.6B, few prompts
- Box: 4. Eval subset / >= threshold
- Box: Green: merge / Red: blocked
- Arrow: 1. Lint + / type check -> 2. Unit + / property tests
- Arrow: 2. Unit + / property tests -> 3. Greedy equiv. / 0.6B, few prompts
- Arrow: 3. Greedy equiv. / 0.6B, few prompts -> 4. Eval subset / >= threshold
- Arrow: 4. Eval subset / >= threshold -> Green: merge / Red: blocked
- Note: MLX runs on the CPU there (the runner has no GPU), so CI stays small: / ~32 tokens per equivalence prompt, ~20 eval items. Full evals and / benchmarks run on your Mac, unattended. / Threshold = baseline score minus a small noise margin, so a real / quality drop turns CI red.

---

## Panel 12: benchmarking harness (S12)

### The harness: from workload to README

- Box: Workload file / A, B, C or D / prompts + arrival / times, fixed seed
- Box: Load generator / asyncio + httpx / fixed concurrency / or Poisson arrivals
- Box: Target (same OpenAI API) / Brigade (via the router) / Brigade v0 (sequential) / mlx_lm.server / Ollama, vllm-metal
- Box: Raw results (JSONL) / one line per request: / times, usage, errors
- Box: Analysis script / -> tables + charts / in the README
- Arrow: Workload file -> Load generator
- Arrow: Load generator <-> Target
- Arrow: Load generator -> Raw results (JSONL)  [label: the client times every token and writes it down]
- Arrow: Raw results (JSONL) -> Analysis script
- Note: Also saved per run: a /metrics snapshot (KV use, prefix hit rate, waste %), the powermetrics GPU log, / the exact config, git commit and model files. Anyone can re-run it and get the same table.

### Two load modes

- Note: Fixed concurrency (closed loop): N clients; each sends its next request / the moment its last one finishes. Throughput at N = 1, 2, 4, 8, 16.
  - Drawing: 3 client lanes, each a back-to-back row of request bars of varying length with no gaps (the next request starts as soon as the last ends).
- Note: Poisson arrivals (open loop): requests arrive at random times at an average / rate, whether or not the server keeps up. Shows queueing, like real traffic.
  - Drawing: tick marks at irregular points on a line, with an arrow labelled "time"; some ticks bunch together, some are far apart.
- Note: Rate above capacity -> the queue grows without limit -> TTFT explodes. (red)

### Workloads

| Workload | Description |
|---|---|
| A | Chat: prompts 100-1k tokens, outputs 50-300 / tests: throughput and batching |
| B | Shared prefix: 4-8 requests share a 2-4k system prompt / tests: prefix caching (S3) and routing (S8) |
| C | Mixed: one 8k-token prompt among short chats / tests: latency fairness, the prefill stall (S4) |
| D | Agent traces: recorded OpenCode sessions, replayed / tests: per-turn prefix hit rate and TTFT (S7) |

### What the client measures, per request

A timeline (left to right): a marker at "send", then a marker at "first token", then 5 more token markers at even gaps, then a marker at "done". Under it, three bars:

- Box: TTFT (spans send -> first token)
- Box: TPOT = (last - first) / (tokens - 1) (spans first token -> done)
- Box: end-to-end latency (spans send -> done)
- Note: Throughput = all output tokens / wall-clock time (tokens from usage). / Then p50 and p99 over every request in the run (S0).

### Charts to expect (shapes, not results)

- Chart 1: y axis "throughput", x axis "concurrency 1 -> 16". The line (blue) rises steeply, then flattens out.
- Chart 2: y axis "TTFT p99", x axis "request rate". The line (red) stays flat, then shoots up sharply; the bend is marked "knee = / capacity".
- Note: Plus: the Brigade vs baselines table, and router vs round-robin.

### Notes

- Box: Fair run conditions (a fanless Air throttles) / Plugged in, Low Power Mode off, apps closed, / 2-min cooldown, discard a warm-up run, / median of 3 runs, log GPU frequency / (powermetrics). Long matrices unattended.
- Box: Fair comparisons / Same model, prompts, max_tokens, thinking setting. / Ollama can't load MLX weights: use its closest / quantization and say so. Count real output / tokens, so an early stop doesn't look 'faster'.
- Box: Report honestly / Brigade will likely be slower than mlx-lm and / vllm-metal (gather copy, Python loop). The win / is explaining every gap with your own numbers.
- Box: Plan / 1. Load generator, v0 + mlx-lm baselines. / 2. Batching table (concurrency 1-16). / 3. Workload B. 4. Full matrix, router, D. (dashed box)

---

## Panel 13: later additions (S13, after the core build)

- Box: 1. API keys + rate limits / Redis token bucket, 429 / Postgres usage ledger (no / double-count on retries) / + fair share per key (S4) (dashed box)
- Box: 2. MCP server / expose metrics + controls / to agents like Claude Code / (S9, S10) (dashed box)
- Box: 3. JSON-schema output / constrain the sampler; / unlocks tool_choice / 'required' (S1, S7) (dashed box)
- Box: 4. Goodput / requests that meet TTFT / and TPOT targets / under load (S12) (dashed box)
- Box: 5. Chunked prefill / split long prompts and mix / them into decode steps: / fixes the stall (S4) (dashed box)
- Box: 6. Kubernetes blueprint / kind + Helm, autoscale / on queue depth (S8) (dashed box)
- Box: 7. Speculative decoding / a small draft model guesses, / the big model checks / (S1) (dashed box)
- Note: Also raised in this session: an Anthropic-compatible endpoint (so Claude Code can be a client), a small RAG agent over Brigade's own docs, / and a custom Metal paged-attention kernel (dropped from scope as too risky). Order above = the spec's priority order.
