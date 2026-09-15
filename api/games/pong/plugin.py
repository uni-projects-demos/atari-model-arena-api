from pathlib import Path

from ...models import ModelSpec, RuntimePolicy
from .. import CompatProfile
from ..base import (
    GAME_MODE_KEY,
    MODEL_MODE_KEY,
    USER_GAME_MODE_KEY,
    USER_MODE_KEY,
    USER_MODEL_MODE_KEY,
    GamePlugin,
    MatchController,
    MatchMode,
)
from ..register import register_game
from .config import (
    PONG_ACTIONS,
    PONG_MODES,
    PONG_PROFILES,
    get_pong_profile_key,
)
from .controllers import GameVsModelPongController, UserVsModelPongController
from .environment import (
    multiplayer_status,
    pong_rom_candidates,
    single_player_status,
)
from .runtime import TrackingRuntime

PONG_MODE_STATUS = {
    USER_MODE_KEY: multiplayer_status,
    MODEL_MODE_KEY: multiplayer_status,
    USER_GAME_MODE_KEY: single_player_status,
    USER_MODEL_MODE_KEY: multiplayer_status,
    GAME_MODE_KEY: single_player_status,
}
PONG_CONTROLLERS = {
    USER_MODE_KEY: UserVsModelPongController,
    MODEL_MODE_KEY: UserVsModelPongController,
    USER_GAME_MODE_KEY: GameVsModelPongController,
    USER_MODEL_MODE_KEY: UserVsModelPongController,
    GAME_MODE_KEY: GameVsModelPongController,
}


@register_game
class PongPlugin(GamePlugin):
    key: str = "pong"
    label: str = "Atari Pong"
    description: str = (
        "Classic Pong supporting model-vs-game, model-vs-model, model-vs-user, "
        "user-vs-game, and user-vs-model modes."
    )
    rom_name: str = "pong"
    default_profile_key: str = "pong-noframeskip-v4"
    profiles: dict[str, CompatProfile] = PONG_PROFILES
    match_modes: dict[str, MatchMode] = PONG_MODES
    action_names: dict[int, str] = PONG_ACTIONS
    models: tuple[ModelSpec, ...] = (
        ModelSpec(
            id="default",
            name="PongNoFrameskip-v4 DQN",
            type="sb3-dqn",
            src="Hugging Face sb3/RL-Zoo",
            repo_id="sb3/dqn-PongNoFrameskip-v4",
            filename="dqn-PongNoFrameskip-v4.zip",
            profile="pong-noframeskip-v4",
            obs_shape=(4, 84, 84),
            n_actions=6,
            fb_type="tracking",
            is_default=True,
            sb3_overrides=(("learning_rate", 1e-4),),
        ),
        ModelSpec(
            id="ppo",
            name="PongNoFrameskip-v4 PPO",
            type="sb3-ppo",
            src="Hugging Face sb3/RL-Zoo",
            repo_id="sb3/ppo-PongNoFrameskip-v4",
            filename="ppo-PongNoFrameskip-v4.zip",
            profile="pong-noframeskip-v4",
            obs_shape=(4, 84, 84),
            n_actions=6,
            fb_type="tracking",
            is_default_model_ii=True,
            sb3_overrides=(
                ("learning_rate", 2.5e-4),
                ("clip_range", 0.1),
            ),
        ),
    )

    def profile_key(self, val: str | None) -> str | None:
        return get_pong_profile_key(val=val)

    def rom_candidates(self, base: Path) -> tuple[Path, ...]:
        return pong_rom_candidates(base=base)

    def mode_status(self, mode_key: str) -> tuple[bool, str | None]:
        status = PONG_MODE_STATUS.get(mode_key)
        return status() if status else (False, f"Unknown Pong match mode: {mode_key}.")

    def is_model_mirror(self, mode_key: str) -> bool:
        return mode_key in {USER_MODE_KEY, MODEL_MODE_KEY}

    def is_model_public(self, spec: ModelSpec) -> bool:
        return spec.id != "tracking"

    def create_controller(
        self,
        mode_key: str,
        profile_key: str,
        policy: RuntimePolicy,
        opponent_policy: RuntimePolicy | None = None,
        starting_side: str = "left",
    ) -> MatchController:
        self.get_profile(key=profile_key)
        try:
            controller = PONG_CONTROLLERS[mode_key]
        except KeyError as exc:
            raise ValueError(f"Unknown Pong match mode: {mode_key}.") from exc
        if mode_key in {MODEL_MODE_KEY, USER_MODE_KEY, USER_MODEL_MODE_KEY}:
            is_model_left: bool = mode_key != USER_MODEL_MODE_KEY
            if mode_key == MODEL_MODE_KEY and opponent_policy is None:
                raise ValueError("Model vs Model requires two policies.")
            return UserVsModelPongController(
                profile_key=profile_key,
                policy=policy,
                opponent_policy=opponent_policy,
                starting_side=starting_side,
                is_model_left=is_model_left,
            )
        return controller(profile_key=profile_key, policy=policy)

    def create_runtime(
        self,
        runtime_type: str,
        *,
        is_mirror: bool = False,
    ) -> RuntimePolicy:
        if runtime_type not in {"heuristic", "tracking"}:
            return super().create_runtime(
                runtime_type=runtime_type,
                is_mirror=is_mirror,
            )
        return TrackingRuntime(
            is_mirror=is_mirror,
        )
