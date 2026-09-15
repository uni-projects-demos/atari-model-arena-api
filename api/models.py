from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class RuntimePolicy(Protocol):
    def reset(self) -> None: ...
    def predict(self, rgb: np.ndarray) -> int: ...


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    type: str
    src: str
    repo_id: str | None = None
    filename: str | None = None
    profile: str | None = None
    obs_shape: tuple[int, ...] = (4, 84, 84)
    n_actions: int = 6
    fb_type: str | None = None
    is_default: bool = False
    is_default_model_ii: bool = False
    sb3_overrides: tuple[tuple[str, float | int], ...] = ()


@dataclass
class ModelEntry:
    spec: ModelSpec
    game_key: str = ""
    base: Any | None = None
    err: str | None = None

    @classmethod
    def from_bundled(
        cls,
        spec: ModelSpec,
        game_key: str,
    ) -> "ModelEntry":
        return cls(spec=spec, game_key=game_key)

    def public(self) -> dict[str, Any]:
        public: dict[str, Any] = {
            "id": self.spec.id,
            "name": self.spec.name,
            "uploaded": self.spec.src == "uploaded",
            "default": self.spec.is_default,
        }
        if self.spec.is_default_model_ii:
            public["default_model_ii"] = True
        return public
