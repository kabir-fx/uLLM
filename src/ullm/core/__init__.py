"""Engine core: requests, scheduler, policy, block manager, engine loop.

This package never imports ``mlx`` (rule 3 in docs/AGENT_ROADMAP.md). It talks to
the model only through the ``Runner`` protocol, so it runs and is tested anywhere.
"""
