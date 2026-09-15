from pathlib import Path
from typing import Any

import numpy as np
from multi_agent_ale_py import ALEInterface

from ..atari import (
    SinglePlayerALE,
    config_ale,
    find_rom,
    missing_rom_msg,
    pooled_frame,
    repeat_cnt,
    rom_candidates,
    status,
    validate_action_cnt,
)
from ..base import CompatProfile
from .config import PONG_PROFILES


class UserVsModelPong:
    def __init__(
        self,
        profile_key: str,
        seed: int | None = None,
        *,
        starting_side: str | None = None,
    ) -> None:
        if starting_side not in {None, "left", "right"}:
            raise ValueError(f"Invalid Pong starting side: {starting_side}.")
        self._starting_side: str | None = starting_side
        self._is_open_pending: bool = starting_side == "right"
        self._ale: Any = ALEInterface()
        ALEInterface.setLoggerMode(mode="error")
        self._profile, self.rng = _configure_ale(
            ale=self._ale,
            profile_key=profile_key,
            seed=seed,
            is_bytes=True,
            frame_skip=1,
        )

        self._ale.setMode(mode=4)
        if int(self._ale.numPlayersActive()) != 2:
            raise RuntimeError("Multi-Agent ALE failed to two players.")

        self._action_mapping: np.ndarray = np.asarray(
            a=self._ale.getMinimalActionSet(), dtype=np.int32
        )
        _validate_six_actions(actions=self._action_mapping)

        self._agents: list[str] = ["first_0", "second_0"]
        self._possible_agents: list[str] = list[str](self._agents)
        self._frame: int = 0
        self._max_cycles: int = 100_000
        self._last_frame: np.ndarray = self._screen()

    def _screen(self) -> np.ndarray:
        return np.asarray(a=self._ale.getScreenRGB(), dtype=np.uint8)

    def _set_opening_direction(self) -> None:
        if not self._is_open_pending:
            return
        velocity: int = int(self._ale.getRAM()[58])
        if velocity == 0:
            return
        if velocity < 128:
            self._ale.setRAM(58, (-velocity) & 0xFF)
        self._is_open_pending = False

    @property
    def agents(self) -> list[str]:
        return self._agents

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float | int],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        minimal: np.ndarray[tuple, np.dtype] = np.asarray(
            a=[
                int(actions.get("first_0", 0)),
                int(actions.get("second_0", 0)),
            ],
            dtype=np.int32,
        )
        minimal = np.clip(a=minimal, a_min=0, a_max=len(self._action_mapping) - 1)
        ale_actions: np.ndarray = self._action_mapping[minimal]

        ttl_rewards: np.ndarray = np.zeros(shape=2, dtype=np.float64)
        recent_frames: list[np.ndarray] = []
        for _ in range(max(1, repeat_cnt(self._profile, self.rng))):
            step_rewards: np.ndarray = np.asarray(
                a=self._ale.act(ale_actions),
                dtype=np.float64,
            ).reshape(-1)

            self._set_opening_direction()

            if step_rewards.size >= 2:
                ttl_rewards += step_rewards[:2]

            recent_frames.append(self._screen())
            if len(recent_frames) > 2:
                recent_frames.pop(0)

            if self._ale.game_over():
                break

        self._last_frame = pooled_frame(frames=recent_frames, fallback=self._screen)
        self._frame += 1

        is_game_over: bool = self._ale.game_over()
        is_truncated: bool = self._frame >= self._max_cycles
        return (
            {agent: self._last_frame.copy() for agent in self._agents},
            {
                "first_0": float(ttl_rewards[0]),
                "second_0": float(ttl_rewards[1]),
            },
            {agent: is_game_over for agent in self._agents},
            {agent: is_truncated for agent in self._agents},
            {agent: {} for agent in self._agents},
        )

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, dict[Any, Any]]]:
        self._ale.reset_game()
        self._is_open_pending = self._starting_side == "right"
        self._agents = list[str](self._possible_agents)
        self._frame = 0
        self._last_frame = self._screen()
        return {agent: self._last_frame.copy() for agent in self._agents}, {
            agent: {} for agent in self._agents
        }

    def close(self) -> None:
        pass


class GameVsModelPong(SinglePlayerALE):
    def __init__(self, profile_key: str, seed: int | None = None) -> None:
        rom_path: Path | None = find_pong_rom()
        if rom_path is None:
            raise RuntimeError(_missing_rom_message())

        super().__init__(
            label="Pong",
            profile=PONG_PROFILES[profile_key],
            rom_path=rom_path,
            expected_actions=6,
            reset_actions=(1, 2),
            seed=seed,
        )


def pong_rom_candidates(base: Path) -> tuple[Path, ...]:
    return rom_candidates(rom_name="pong", base=base)


def find_pong_rom(base: Path | None = None) -> Path | None:
    return find_rom(rom_name="pong", base=base)


def _missing_rom_message() -> str:
    return missing_rom_msg(rom_name="pong", label="Pong")


def multiplayer_status() -> tuple[bool, str | None]:
    return status(rom_name="pong", label="Pong")


def single_player_status() -> tuple[bool, str | None]:
    return status(rom_name="pong", label="Pong")


def _validate_six_actions(actions: Any) -> None:
    validate_action_cnt(actions=actions, label="Pong", expected_actions=6)


def _configure_ale(
    ale,
    profile_key: str,
    seed: int | None,
    *,
    is_bytes: bool,
    frame_skip: int | None = None,
) -> tuple[CompatProfile, np.random.Generator]:
    rom_path: Path | None = find_pong_rom()
    if rom_path is None:
        raise RuntimeError(_missing_rom_message())

    profile: CompatProfile = PONG_PROFILES[profile_key]
    return (
        profile,
        config_ale(
            ale=ale,
            profile=profile,
            rom_path=rom_path,
            seed=seed,
            is_bytes=is_bytes,
            frame_skip=frame_skip,
        ),
    )


def user_vs_model_pong(
    profile_key: str, *, starting_side: str | None = None
) -> UserVsModelPong:
    return UserVsModelPong(profile_key=profile_key, starting_side=starting_side)


def game_vs_model_pong(profile_key: str) -> GameVsModelPong:
    return GameVsModelPong(profile_key=profile_key)
