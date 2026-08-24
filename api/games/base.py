from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..models import ModelSpec, RuntimePolicy

USER_MODE_KEY = "user"
GAME_MODE_KEY = "game"
DEFAULT_MODE_KEY = GAME_MODE_KEY
STANDARD_MODE_KEYS = frozenset({USER_MODE_KEY, GAME_MODE_KEY})


@dataclass(frozen=True)
class CompatProfile:
    key: str
    label: str
    frame_skip: int | tuple[int, int]
    repeat_action_prob: float | int


@dataclass(frozen=True)
class UserControls:
    move: str | None = None
    fire: str | None = None


@dataclass(frozen=True)
class MatchMode:
    key: str
    left_label: str
    right_label: str
    is_user: bool = False
    user_controls: UserControls | None = None


@dataclass(frozen=True)
class MatchSnapshot:
    frame: np.ndarray
    left_score: float | int
    right_score: float | int
    model_action: int
    is_done: bool = False


class MatchController(ABC):
    def __init__(self, policy: RuntimePolicy) -> None:
        self._policy: RuntimePolicy = policy
        self._left_score: float | int = 0.0
        self._right_score: float | int = 0.0
        self._model_action: int = 0

    def _frame(self) -> np.ndarray:
        raise NotImplementedError

    def _reset_scores(self) -> None:
        self._left_score = self._right_score = 0.0
        self._model_action = 0

    def _snapshot(self, is_done: bool = False) -> MatchSnapshot:
        return MatchSnapshot(
            self._frame(),
            self._left_score,
            self._right_score,
            self._model_action,
            is_done,
        )

    @abstractmethod
    def reset(self) -> MatchSnapshot: ...

    def set_user(self, direction: str) -> None:
        pass

    @abstractmethod
    def step(self) -> MatchSnapshot: ...

    @abstractmethod
    def close(self) -> None: ...


class GamePlugin(ABC):
    key: str
    label: str
    description: str = ""
    rom_name: str
    default_profile_key: str
    action_names: Mapping[int, str]
    profiles: Mapping[str, CompatProfile]
    match_modes: Mapping[str, MatchMode]
    models: tuple[ModelSpec, ...] = ()

    @abstractmethod
    def profile_key(self, val: str | None) -> str | None: ...

    @abstractmethod
    def rom_candidates(self, base: Path) -> tuple[Path, ...]: ...

    @abstractmethod
    def mode_status(self, mode_key: str) -> tuple[bool, str | None]: ...

    @abstractmethod
    def create_controller(
        self,
        mode_key: str,
        profile_key: str,
        policy: RuntimePolicy,
    ) -> MatchController: ...

    def get_profile(self, key: str) -> CompatProfile:
        try:
            return self.profiles[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {self.label} compatibility profile: {key}."
            ) from exc

    def get_match_mode(self, key: str) -> MatchMode:
        try:
            return self.match_modes[key]
        except KeyError as exc:
            raise ValueError(f"Unknown {self.label} match mode: {key}.") from exc

    def find_rom(self, base: Path) -> Path | None:
        return next(
            (path for path in self.rom_candidates(base) if path.is_file()), None
        )

    def is_model_mirror(self, mode_key: str) -> bool:
        return False

    def is_model_public(self, spec: ModelSpec) -> bool:
        return True

    def create_runtime(
        self,
        runtime_type: str,
        *,
        is_mirror: bool = False,
    ) -> RuntimePolicy:
        raise ValueError(f"Unknown {self.label} runtime type: {runtime_type}.")

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "modes": [asdict(mode) for mode in self.match_modes.values()],
        }


def get_profile_key(
    val: str | None,
    hints: tuple[tuple[str, str], ...],
) -> str | None:
    if val:
        normalized: str = "".join(ch for ch in val.lower() if ch.isalnum())
        for token, profile_key in hints:
            if token in normalized:
                return profile_key
    return None
