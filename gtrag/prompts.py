"""Load prompt templates from prompts/*.txt.

Template conventions
  {{name}}  -> filled from keyword arguments passed to render()
  {name}    -> filled from config.prompts.delimiters (output-format delimiters), so prompt
               files never need escaped braces.
Files are read byte-for-byte (no stripping), so a prompt behaves exactly like the
string it was copied from.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gtrag.config import resolve_repo_path

_VAR = re.compile(r"\{\{(\w+)\}\}")


class PromptLibrary:
    def __init__(self, prompt_dir: str | Path, delimiters: dict[str, str] | None = None):
        self.dir = Path(prompt_dir)
        self.delimiters = dict(delimiters or {})
        self._cache: dict[str, str] = {}

    def raw(self, name: str) -> str:
        if name not in self._cache:
            with open(self.dir / f"{name}.txt", encoding="utf-8", newline="") as f:
                self._cache[name] = f.read()
        return self._cache[name]

    def render(self, name: str, **kwargs: Any) -> str:
        text = self.raw(name)
        for key, value in self.delimiters.items():
            text = text.replace("{" + key + "}", str(value))
        needed = set(_VAR.findall(text))
        missing, extra = needed - set(kwargs), set(kwargs) - needed
        if missing or extra:
            raise KeyError(f"Prompt '{name}': missing={sorted(missing)} unexpected={sorted(extra)}")
        return _VAR.sub(lambda m: str(kwargs[m.group(1)]), text)


def from_config(cfg) -> PromptLibrary:
    return PromptLibrary(resolve_repo_path(cfg.prompts.dir), cfg.prompts.delimiters)
