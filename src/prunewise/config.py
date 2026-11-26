"""Run configuration: data size, row-group sizes and the cost and small-file parameters."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    name: str
    n_rows: int
    row_group_sizes: tuple[int, ...]
    max_files: int  # a candidate expected to write more files than this is excluded
    min_file_rows: int  # a candidate whose median file is smaller gets the small-file penalty
    file_open_cost_bytes: int  # bytes-equivalent of opening one file (footer read, request)
    small_file_list_cost_bytes: int  # per file, per execution, for small-file candidates

    @property
    def baseline_row_group_size(self) -> int:
        return max(self.row_group_sizes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["row_group_sizes"] = list(self.row_group_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = dict(d)
        d["row_group_sizes"] = tuple(d["row_group_sizes"])
        return cls(**d)


PRESETS = {
    "tests": Config(
        name="tests",
        n_rows=40_000,
        row_group_sizes=(2_000, 8_000),
        max_files=1_000,
        min_file_rows=200,
        file_open_cost_bytes=1_024,
        small_file_list_cost_bytes=256,
    ),
    "demo": Config(
        name="demo",
        n_rows=250_000,
        row_group_sizes=(5_000, 20_000),
        max_files=2_000,
        min_file_rows=1_000,
        file_open_cost_bytes=4_096,
        small_file_list_cost_bytes=1_024,
    ),
}


def load_config(spec: str) -> Config:
    """Resolve a preset name (``tests``, ``demo``) or a TOML file that overrides a preset."""
    if spec in PRESETS:
        return PRESETS[spec]
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"unknown config {spec!r}: not a preset ({', '.join(PRESETS)}) or a file")
    raw = tomllib.loads(path.read_text())
    base = PRESETS[raw.pop("base", "demo")]
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "row_group_sizes" in raw:
        raw["row_group_sizes"] = tuple(raw["row_group_sizes"])
    return replace(base, name=path.stem, **raw)
