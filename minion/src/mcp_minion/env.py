"""API-key lookup shared by the minion CLI and its consumers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import dotenv_values


def find_api_key(
    env_var: str = "OPENROUTER_API_KEY",
    extra_dotenv: Sequence[Path] = (),
    folder: Path | None = None,
) -> str | None:
    """Look up an API key without touching ``os.environ``.

    Order: the process environment, then ``folder/.env`` (if ``folder`` is
    given), then each file in ``extra_dotenv`` in order, then
    ``~/.mcp-minion``. Missing files are skipped; an empty value counts as
    unset. Returns None when no source has the key.
    """
    value = os.environ.get(env_var)
    if value:
        return value
    candidates: list[Path] = []
    if folder is not None:
        candidates.append(Path(folder) / ".env")
    candidates.extend(Path(p) for p in extra_dotenv)
    candidates.append(Path.home() / ".mcp-minion")
    for path in candidates:
        if path.is_file():
            value = dotenv_values(path).get(env_var)
            if value:
                return value
    return None
