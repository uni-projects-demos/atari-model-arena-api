from dataclasses import dataclass
from typing import Any

import numpy as np

from .games import DEFAULT_MODE_KEY, GamePlugin, get_game
from .games.base import MatchController, MatchMode, MatchSnapshot
from .models import RuntimePolicy
from .policies import PolicyManager


@dataclass(frozen=True)
class SessionConfig:
    game: str
    profile: str = "auto"
    model: str = "default"
    mode: str | None = None


@dataclass(frozen=True)
class FramePacket:
    metadata: dict[str, Any]
    pixels: bytes


class GameSession:
    def __init__(self, config: SessionConfig, policies: PolicyManager) -> None:
        self._config: SessionConfig = config
        self._controller: MatchController | None = None
        self._policies: PolicyManager = policies
        self._profile_key: str = config.profile
        self._snapshot_frame: np.ndarray | None = None
        self._steps: int = 0

        self._game: GamePlugin = get_game(key=config.game)
        self._mode_key: str = config.mode or DEFAULT_MODE_KEY
        self._game.get_match_mode(key=self._mode_key)

        self.reset()

    def _resolve_profile(self) -> str:
        return self._policies.resolve_profile(
            model_id=self._config.model,
            req_profile=self._config.profile,
            game_key=self._game.key,
        )

    def _create_policy(self) -> RuntimePolicy:
        return self._policies.runtime(
            model_id=self._config.model,
            is_mirror=self._game.is_model_mirror(mode_key=self._mode_key),
            game_key=self._game.key,
        )

    def _rgba_frame(self) -> tuple[np.ndarray, int, int]:
        pixels: np.ndarray = np.asarray(a=self._snapshot_frame, dtype=np.uint8)
        if pixels.ndim == 2:
            pixels = np.repeat(a=pixels[..., None], repeats=3, axis=2)
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 frame, got {pixels.shape}")

        height, width = pixels.shape[:2]
        if pixels.shape[2] == 4:
            return np.ascontiguousarray(a=pixels), height, width

        rgba: np.ndarray = np.empty(shape=(height, width, 4), dtype=np.uint8)
        rgba[..., :3] = pixels
        rgba[..., 3] = 255
        return rgba, height, width

    def _packet(self, snapshot: MatchSnapshot) -> FramePacket:
        self._snapshot_frame = snapshot.frame
        frame, height, width = self._rgba_frame()
        mode: MatchMode = self.match_mode
        return FramePacket(
            metadata={
                "type": "frame",
                "width": width,
                "height": height,
                "left_score": snapshot.left_score,
                "right_score": snapshot.right_score,
                "left_label": mode.left_label,
                "right_label": mode.right_label,
                "steps": self._steps,
                "model_action": snapshot.model_action,
                "model_action_name": self._game.action_names.get(
                    snapshot.model_action,
                    str(snapshot.model_action),
                ),
            },
            pixels=frame.tobytes(),
        )

    @property
    def match_mode(self) -> MatchMode:
        return self._game.get_match_mode(key=self._mode_key)

    def set_user(self, direction: str) -> None:
        if self._controller:
            self._controller.set_user(direction=direction)

    def step(self) -> FramePacket:
        if self._controller is None:
            self.reset()
        assert self._controller is not None
        snapshot: MatchSnapshot = self._controller.step()
        self._steps += 1
        packet: FramePacket = self._packet(snapshot=snapshot)
        if snapshot.is_done:
            self.reset()
        return packet

    def reset(self) -> None:
        self.close()
        self._steps = 0
        self._profile_key = self._resolve_profile()
        policy: RuntimePolicy = self._create_policy()
        policy.reset()
        self._controller = self._game.create_controller(
            mode_key=self._mode_key,
            profile_key=self._profile_key,
            policy=policy,
        )
        if self._controller:
            self._controller.reset()

    def close(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None
