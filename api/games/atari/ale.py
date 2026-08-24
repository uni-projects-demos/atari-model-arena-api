from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ale_py import ALEInterface

from ..base import CompatProfile


class SinglePlayerALE:
    def __init__(
        self,
        *,
        label: str,
        profile: CompatProfile,
        rom_path: Path,
        expected_actions: int,
        reset_actions: Sequence[int] = (1, 2),
        is_episodic_life: bool = False,
        seed: int | None = None,
    ) -> None:
        self._label = label
        self._profile = profile
        self._reset_actions = tuple(int(action) for action in reset_actions)
        self._is_episodic_life = is_episodic_life
        self._ale: Any = ALEInterface()
        self._rng = config_ale(
            ale=self._ale,
            profile=profile,
            rom_path=rom_path,
            seed=seed,
            is_bytes=False,
            frame_skip=1,
        )
        self._actions = list(self._ale.getMinimalActionSet())
        validate_action_cnt(
            actions=self._actions, label=label, expected_actions=expected_actions
        )
        self._lives = self._ale_lives()
        self._final_frame = self._screen()

    def _screen(self) -> np.ndarray:
        return np.asarray(a=self._ale.getScreenRGB(), dtype=np.uint8).copy()

    def _ale_lives(self) -> int | None:
        try:
            return int(self._ale.lives())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def _action_code(self, action_idx: int) -> Any:
        if action_idx < 0 or action_idx >= len(self._actions):
            action_idx = 0
        return self._actions[action_idx]

    def _repeat_action(self, action: int) -> tuple[np.ndarray, float | int, bool]:
        ttl_reward: float | int = 0.0
        recent_frames: list[np.ndarray] = []
        for _ in range(max(1, repeat_cnt(profile=self._profile, rng=self._rng))):
            ttl_reward += float(self._ale.act(self._action_code(action)))
            recent_frames.append(self._screen())
            if len(recent_frames) > 2:
                recent_frames.pop(0)
            if self._ale.game_over():
                break
        return (
            pooled_frame(frames=recent_frames, fallback=self._screen),
            ttl_reward,
            self._ale.game_over(),
        )

    def _resume_after_life_loss(self) -> np.ndarray:
        for action in (0, *self._reset_actions):
            frame, _, done = self._repeat_action(action=action)
            if done:
                break
        self._lives = self._ale_lives()
        return frame

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._ale.reset_game()
        for _ in range(int(self._rng.integers(low=1, high=31))):
            self._ale.act(self._action_code(0))
            if self._ale.game_over():
                self._ale.reset_game()

        frame: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._screen()
        for action in self._reset_actions:
            frame, _reward, done = self._repeat_action(action=action)
            if done:
                break

        self._lives = self._ale_lives()
        self._final_frame = frame
        return self._final_frame, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float | int, bool, bool, dict[str, Any]]:
        prev_lives: int | None = self._lives
        frame, ttl_reward, is_game_over = self._repeat_action(action=action)
        cur_lives: int | None = self._ale_lives()
        is_life_lost = (
            self._is_episodic_life
            and not is_game_over
            and prev_lives is not None
            and cur_lives is not None
            and 0 < cur_lives < prev_lives
        )

        if is_life_lost:
            frame = self._resume_after_life_loss()
        else:
            self._lives = cur_lives

        self._final_frame = np.asarray(a=frame, dtype=np.uint8)
        return (
            self._final_frame,
            ttl_reward,
            is_game_over,
            False,
            {"life_lost": is_life_lost},
        )

    def close(self) -> None:
        pass


def validate_action_cnt(actions, label: str, expected_actions: int) -> None:
    if len(actions) != expected_actions:
        raise RuntimeError(
            f"Expected {label}'s {expected_actions}-action, got {len(actions)} actions"
        )


def repeat_cnt(profile: CompatProfile, rng: np.random.Generator) -> int:
    frame_skip: int | tuple[int, int] = profile.frame_skip
    if isinstance(frame_skip, tuple):
        return int(rng.integers(*frame_skip))
    return int(frame_skip)


def pooled_frame(
    frames: list[np.ndarray], fallback: Callable[[], np.ndarray]
) -> np.ndarray:
    if len(frames) >= 2:
        return np.maximum(frames[-2], frames[-1]).astype(dtype=np.uint8)
    elif frames:
        return np.asarray(a=frames[-1], dtype=np.uint8)
    return np.asarray(a=fallback(), dtype=np.uint8)


def set_ale_val(
    ale,
    method: str,
    name: str,
    val: Any,
    *,
    is_bytes: bool,
) -> None:
    setter: Any = getattr(ale, method)
    keys: tuple[bytes, str] | tuple[str, bytes] = (
        (name.encode(), name) if is_bytes else (name, name.encode())
    )
    try:
        setter(keys[0], val)
    except TypeError:
        setter(keys[1], val)


def config_ale(
    ale,
    profile: CompatProfile,
    rom_path: Path,
    seed: int | None,
    *,
    is_bytes: bool,
    frame_skip: int | None = None,
) -> np.random.Generator:
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    random_seed: int = int(
        seed if seed is not None else rng.integers(low=0, high=2**31 - 1)
    )
    set_ale_val(
        ale=ale, method="setInt", name="random_seed", val=random_seed, is_bytes=is_bytes
    )
    if frame_skip is not None:
        set_ale_val(
            ale=ale,
            method="setInt",
            name="frame_skip",
            val=frame_skip,
            is_bytes=is_bytes,
        )
    set_ale_val(
        ale=ale,
        method="setFloat",
        name="repeat_action_probability",
        val=float(profile.repeat_action_prob),
        is_bytes=is_bytes,
    )
    ale.loadROM(str(rom_path))
    return rng
