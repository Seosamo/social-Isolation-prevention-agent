from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path
    values: dict

    @classmethod
    def load(cls, path: str | Path = "config/default.json") -> "Config":
        path = Path(path).resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        if values["baseline_months"] != 2:
            raise ValueError("This version requires baseline_months=2.")
        if not 0 < values["minimum_coverage"] <= 1:
            raise ValueError("minimum_coverage must be in (0, 1].")
        if not 0 <= values["stable_change"] < values["suspect_decline"] <= 1:
            raise ValueError("Require 0 <= stable_change < suspect_decline <= 1.")
        return cls(path.parent.parent, values)

    def path(self, key: str) -> Path:
        path = Path(self.values[key])
        return path if path.is_absolute() else self.root / path

    @property
    def output(self) -> Path:
        return self.path("output_dir")

    @property
    def version(self) -> str:
        # Policy collection settings are independent of analytical features/models.
        payload = json.dumps({k: v for k, v in self.values.items() if k not in {"policies", "agent"}}, sort_keys=True).encode()
        for key in ("mapping_file", "industry_file"):
            path = self.path(key)
            payload += path.read_bytes() if path.exists() else b"missing"
        return hashlib.sha256(payload).hexdigest()[:12]


