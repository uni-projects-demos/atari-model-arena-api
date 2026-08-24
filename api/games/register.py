from collections.abc import Iterable, Iterator
from importlib import import_module
from importlib.metadata import entry_points
from inspect import isclass
from pkgutil import iter_modules
from typing import Any, TypeVar, overload

from .base import STANDARD_MODE_KEYS, GamePlugin

TGame = TypeVar("TGame", bound=type[GamePlugin])


class GameRegister:
    def __init__(self) -> None:
        self._games: dict[str, GamePlugin] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._games))

    def register(self, game: GamePlugin) -> GamePlugin:
        if not game.key:
            raise ValueError("Game must have a key to register.")
        elif game.key in self._games:
            raise ValueError(f"Game already registered: {game.key}.")

        unknown_modes: set[str] = set(game.match_modes) - STANDARD_MODE_KEYS
        if unknown_modes:
            raise ValueError(
                f"Game {game.key} uses invalid modes: {sorted(unknown_modes)}."
            )
        elif game.default_profile_key not in game.profiles:
            raise ValueError(
                f"Default profile {game.default_profile_key!r} is not registered for {game.key}."
            )

        self._games[game.key] = game
        return game

    def get(self, key: str) -> GamePlugin:
        try:
            return self._games[key]
        except KeyError as exc:
            raise ValueError(f"Unknown game: {key}.") from exc

    def values(self) -> Iterable[GamePlugin]:
        return (self._games[key] for key in sorted(self._games))

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._games))


GAME_REGISTER = GameRegister()


@overload
def register_game(game: TGame) -> TGame: ...


@overload
def register_game(game: GamePlugin) -> GamePlugin: ...


def register_game(game: Any):
    if isclass(object=game) and issubclass(game, GamePlugin):
        GAME_REGISTER.register(game=game())
        return game
    return GAME_REGISTER.register(game=game)


def get_game(key: str) -> GamePlugin:
    return GAME_REGISTER.get(key=key)


def discover_games(pkg_name: str, paths: Any) -> None:
    ignored: set[str] = {"atari", "base", "register"}
    for module in iter_modules(path=paths):
        if module.name.startswith("_") or module.name in ignored:
            continue
        import_module(
            name=(
                f"{pkg_name}.{module.name}.plugin"
                if module.ispkg
                else f"{pkg_name}.{module.name}"
            )
        )

    try:
        discovered_entry_points = entry_points(group="atari_model_arena.games")
    except TypeError:
        discovered_entry_points = entry_points().select(group="atari_model_arena.games")
    for entry_point in discovered_entry_points:
        entry_point.load()
