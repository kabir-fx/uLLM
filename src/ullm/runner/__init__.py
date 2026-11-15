"""Runners: the only code that owns tensors. ``MLXRunner`` runs the real model;
``FakeRunner`` is deterministic and MLX-free, for tests and ``ullm sim``."""
