# uLLM design notes

*Written by the author, in the author's own words, one section per milestone (roadmap P6). Agents fact-check this file when asked and never write it. The headings below are a skeleton.*

## M0 Foundations

### Contracts and why they come first

### Testing the engine without the model

### What the goldens taught us about greedy decoding

## M1 One request, end to end

### The request path, thread by thread

### The runner boundary

### Streaming, stop strings and the detokenizer

### Equivalence against mlx-lm

### First numbers, and what explains them

## M2 Continuous batching and the terminal

### The scheduler step

### Contiguous batch KV

### Telemetry without slowing the engine

### The batching sweep against the baselines

## M3 Blocks, prefix caching, preemption

### The block manager and its invariants

### The KV layout spike and the decision

### Prefix caching

### Preemption

### The timing side channel

## M4 Evals

### What is measured, and why these datasets

### The quantization trade-off

### The CI gate

## M5 Router

### Routing decisions

### Failure handling

### Tracing across Go and Python

## M6 Agent-facing API

### Tool calling

### Structured output

### Replaying agent traces

### The MCP server

## M7 v1.0

### The final numbers

### What I would do differently

## M8 Multi-tenant gateway

## M9 Deployment blueprint
