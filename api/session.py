from dataclasses import dataclass
from secrets import choice
from typing import Any

import numpy as np

from .games import DEFAULT_MODE_KEY, GamePlugin, get_game
from .games.atari.user import UserPolicy
from .games.base import MODEL_MODE_KEY, MatchController, MatchMode, MatchSnapshot
from .models import RuntimePolicy
from .policies import PolicyManager


@dataclass(frozen=True)
class SessionPlayer:
    slot: str
    type: str
    model: str | None = None


def resolve_players(
    game: GamePlugin,
    mode_key: str,
    requested: Any,
    model_id: str,
) -> tuple[SessionPlayer, ...]:
    slots = game.get_match_mode(mode_key).players
    if requested is None:
        return tuple[SessionPlayer](
            SessionPlayer(
                slot.key,
                slot.player_type,
                model_id if slot.player_type == "model" else None,
            )
            for slot in slots
        )
    if not isinstance(requested, list):
        raise TypeError("'players' must be a list.")
    by_slot: dict[str, Any] = {}
    for player in requested:
        if not isinstance(player, dict) or not isinstance(player.get("slot"), str):
            raise TypeError("Each player must include a slot.")
        if player["slot"] in by_slot:
            raise ValueError(f"Duplicate player slot: {player['slot']}.")
        by_slot[player["slot"]] = player
    if set[str](by_slot) != {slot.key for slot in slots}:
        raise ValueError("Players must match the selected mode's slots.")
    resolved = []
    for slot in slots:
        player = by_slot[slot.key]
        if player.get("type") != slot.player_type:
            raise ValueError(f"Invalid player type for slot {slot.key}.")
        selected = player.get("model") or model_id
        if slot.player_type == "model" and not isinstance(selected, str):
            raise ValueError(f"Invalid model for slot {slot.key}.")
        resolved.append(
            SessionPlayer(
                slot.key,
                slot.player_type,
                selected if slot.player_type == "model" else None,
            )
        )
    return tuple(resolved)


@dataclass(frozen=True)
class SessionConfig:
    game: str
    profile: str = "auto"
    model: str = "default"
    mode: str | None = None
    players: tuple[SessionPlayer, ...] | None = None


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
        self._starting_side: str | None = None
        self._match_number: int = 0

        self._game: GamePlugin = get_game(key=config.game)
        self._mode_key: str = config.mode or DEFAULT_MODE_KEY
        self._game.get_match_mode(key=self._mode_key)
        self._players = (
            config.players
            if config.players is not None
            else resolve_players(self._game, self._mode_key, None, config.model)
        )

        self.reset()

    def _resolve_profile(self) -> str:
        model = next((p.model for p in self._players if p.model is not None), None)
        if model is None:
            profile = self._config.profile
            if profile == "auto":
                return self._game.default_profile_key
            return self._game.get_profile(profile).key
        return self._policies.resolve_profile(
            model_id=model,
            req_profile=self._config.profile,
            game_key=self._game.key,
        )

    def _create_policies(self) -> list[RuntimePolicy]:
        slots = {slot.key: slot for slot in self.match_mode.players}
        return [
            self._policies.runtime(
                model_id=player.model,
                is_mirror=slots[player.slot].is_mirror,
                game_key=self._game.key,
            )
            for player in self._players
            if player.model is not None
        ]

    @property
    def started_metadata(self) -> dict[str, Any]:
        slots = {slot.key: slot for slot in self.match_mode.players}
        return {
            "type": "started",
            **self._opening_metadata(),
            "game": self._game.key,
            "mode": self._mode_key,
            "profile": self._profile_key,
            "players": [
                {
                    "slot": p.slot,
                    "type": p.type,
                    "model": p.model,
                    "side": slots[p.slot].side,
                }
                for p in self._players
            ],
        }

    def _opening_metadata(self) -> dict[str, Any]:
        if self._mode_key != MODEL_MODE_KEY:
            return {}
        return {
            "starting_side": self._starting_side,
            "match_number": self._match_number,
        }

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
                **self._opening_metadata(),
                "width": width,
                "height": height,
                "left_score": snapshot.left_score,
                "right_score": snapshot.right_score,
                "left_label": mode.left_label,
                "right_label": mode.right_label,
                "steps": self._steps,
                "player_actions": snapshot.player_actions or {},
                "player_action_names": {
                    slot: self._game.action_names.get(action, str(action))
                    for slot, action in (snapshot.player_actions or {}).items()
                },
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
        policies = self._create_policies()
        if not policies:
            policies = [UserPolicy(game_key=self._game.key)]
        for policy in policies:
            policy.reset()
        controller_args: dict[str, Any] = {
            "mode_key": self._mode_key,
            "profile_key": self._profile_key,
            "policy": policies[0],
        }
        if len(policies) > 1:
            controller_args["opponent_policy"] = policies[1]
        starting_side = self._starting_side
        if self._mode_key == MODEL_MODE_KEY:
            starting_side = (
                choice(("left", "right"))
                if starting_side is None
                else "right" if starting_side == "left" else "left"
            )
            controller_args["starting_side"] = starting_side
        self._controller = self._game.create_controller(**controller_args)
        if self._controller:
            self._controller.reset()
        self._starting_side = starting_side
        self._match_number += 1

    def close(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None
