"""``ullm serve``'s flags become the engine's config (decisions 0020 and 0035)."""

from typing import Any

import pytest
import uvicorn

from ullm import cli
from ullm.server.app import ServeConfig


def _config(monkeypatch: pytest.MonkeyPatch, *flags: str) -> ServeConfig:
    built: list[ServeConfig] = []

    def create_app(runner: Any, tokenizer: Any, config: ServeConfig, emit: Any) -> None:
        built.append(config)

    monkeypatch.setattr(cli, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    assert cli.main(["serve", "--fake", *flags]) == 0
    (config,) = built
    return config


def test_kv_pool_gib_is_binary_gigabytes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _config(monkeypatch, "--kv-pool-gib", "0.5").engine.kv_pool_bytes == 2**29


def test_without_kv_pool_gib_the_pool_follows_the_slot_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _config(monkeypatch).engine.kv_pool_bytes is None
