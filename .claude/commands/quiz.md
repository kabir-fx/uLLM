---
description: Ask the author ten interview questions on a milestone, then grade the answers against the code
argument-hint: <milestone, e.g. M1>
---

Quiz the author on milestone **$ARGUMENTS** of `docs/AGENT_ROADMAP.md`. The goal is that the author can defend every part of it in an interview without notes.

1. Read the milestone's tasks, their catalog entries, the decision records and the code and tests they produced (`git log` for the milestone's pull requests helps).
2. Write **ten questions**, as an interviewer at an inference, applied-AI or backend role would ask them. Mix:
   - how a mechanism works, step by step (at least three);
   - why it is shaped this way and what the alternative costs (at least two);
   - a failure mode, or what happens under a specific input or load (at least two);
   - a number from a committed result and what explains it (when results exist);
   - what the author would change next and why.
   Ground each one in this codebase, not in generic theory.
3. Ask all ten at once, numbered. Do not include answers, hints or file pointers. Then stop and wait.
4. When the author answers, grade each answer as **solid**, **partial** or **missing**, against the code itself, citing `file:line`. Correct anything wrong plainly.
5. Finish with the topics to revisit before the gate counts as passed, and suggest which `docs/DESIGN.md` sections need work. Do not write `DESIGN.md` for the author.
