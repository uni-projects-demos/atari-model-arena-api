from pathlib import Path

from ...models import ModelSpec, RuntimePolicy
from .. import CompatProfile
from ..atari import SinglePlayerModelController
from ..base import GAME_MODE_KEY, GamePlugin, MatchController, MatchMode
from ..register import register_game
from .config import (
    BREAKOUT_ACTIONS,
    BREAKOUT_MODES,
    BREAKOUT_PROFILES,
    get_breakout_profile_key,
)
from .environment import (
    breakout_rom_candidates,
    breakout_status,
    game_vs_model_breakout,
)


@register_game
class BreakoutPlugin(GamePlugin):
    key: str = "breakout"
    label: str = "Atari Breakout"
    description: str = "Classic Breakout supporting model-vs-game mode."
    rom_name: str = "breakout"
    default_profile_key: str = "breakout-noframeskip-v4"
    profiles: dict[str, CompatProfile] = BREAKOUT_PROFILES
    match_modes: dict[str, MatchMode] = BREAKOUT_MODES
    action_names: dict[int, str] = BREAKOUT_ACTIONS
    models: tuple[ModelSpec, ModelSpec] = (
        ModelSpec(
            id="default",
            name="BreakoutNoFrameskip-v4",
            type="sb3-ppo",
            src="Hugging Face sb3/RL-Zoo",
            repo_id="sb3/ppo-BreakoutNoFrameskip-v4",
            filename="ppo-BreakoutNoFrameskip-v4.zip",
            profile="breakout-noframeskip-v4",
            obs_shape=(4, 84, 84),
            n_actions=4,
            is_default=True,
            sb3_overrides=(
                ("learning_rate", 2.5e-4),
                ("clip_range", 0.1),
            ),
        ),
        ModelSpec(
            id="dqn",
            name="BreakoutNoFrameskip-v4 DQN",
            type="sb3-dqn",
            src="Hugging Face sb3/RL-Zoo",
            repo_id="sb3/dqn-BreakoutNoFrameskip-v4",
            filename="dqn-BreakoutNoFrameskip-v4.zip",
            profile="breakout-noframeskip-v4",
            obs_shape=(4, 84, 84),
            n_actions=4,
            sb3_overrides=(("learning_rate", 1e-4),),
        ),
    )

    def profile_key(self, val: str | None) -> str | None:
        return get_breakout_profile_key(val=val)

    def rom_candidates(self, base: Path) -> tuple[Path, ...]:
        return breakout_rom_candidates(base=base)

    def mode_status(self, mode_key: str) -> tuple[bool, str | None]:
        if mode_key != GAME_MODE_KEY:
            return False, "Breakout supports Model vs Game only."
        return breakout_status()

    def create_controller(
        self,
        mode_key: str,
        profile_key: str,
        policy: RuntimePolicy,
    ) -> MatchController:
        if mode_key != GAME_MODE_KEY:
            raise ValueError(f"Unknown Breakout match mode: {mode_key}")

        self.get_profile(key=profile_key)
        return SinglePlayerModelController(
            profile_key=profile_key,
            policy=policy,
            env_factory=game_vs_model_breakout,
        )
