from pathlib import Path

from ..atari import (
    SinglePlayerALE,
    find_rom,
    missing_rom_msg,
    rom_candidates,
    status,
)
from .config import BREAKOUT_PROFILES


class GameVsModelBreakout(SinglePlayerALE):
    def __init__(self, profile_key: str, seed: int | None = None) -> None:
        rom_path: Path | None = find_breakout_rom()
        if rom_path is None:
            raise RuntimeError(_missing_rom_msg())

        super().__init__(
            label="Breakout",
            profile=BREAKOUT_PROFILES[profile_key],
            rom_path=rom_path,
            expected_actions=4,
            reset_actions=(1, 2),
            is_episodic_life=True,
            seed=seed,
        )


def breakout_rom_candidates(base: Path) -> tuple[Path, ...]:
    return rom_candidates(rom_name="breakout", base=base)


def find_breakout_rom(base: Path | None = None) -> Path | None:
    return find_rom(rom_name="breakout", base=base)


def _missing_rom_msg() -> str:
    return missing_rom_msg(rom_name="breakout", label="Breakout")


def breakout_status() -> tuple[bool, str | None]:
    return status(rom_name="breakout", label="Breakout")


def game_vs_model_breakout(profile_key: str) -> GameVsModelBreakout:
    return GameVsModelBreakout(profile_key=profile_key)
