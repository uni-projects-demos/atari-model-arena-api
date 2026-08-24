from typing import Any

from fastapi import APIRouter

from ..games import GAME_REGISTER, GamePlugin, get_game
from ..services import get_policies

router: APIRouter = APIRouter()


@router.get("/")
def service_info() -> dict[str, str]:
    return {"service": "Atari Model Arena API"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/info")
def info() -> dict[str, Any]:
    return {"games": [_game_info(game_key=key) for key in GAME_REGISTER]}


def _game_info(game_key: str) -> dict[str, Any]:
    game: GamePlugin = get_game(key=game_key)
    game_info: dict[str, Any] = game.describe()
    for mode in game_info["modes"]:
        mode["available"], mode["error"] = game.mode_status(mode_key=mode["key"])
    game_info["models"] = get_policies().public_entries(game_key=game_key)
    return game_info
