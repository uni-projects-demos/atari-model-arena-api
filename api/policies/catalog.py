from typing import Any

from ..games import GAME_REGISTER, GameRegister
from ..models import ModelEntry


class ModelCatalog:
    def __init__(self, games: GameRegister = GAME_REGISTER) -> None:
        self._games: GameRegister = games
        self._entries: dict[tuple[str, str], ModelEntry] = {}
        for game in self._games.values():
            for spec in game.models:
                self.register(
                    entry=ModelEntry.from_bundled(spec=spec, game_key=game.key)
                )

    def entries(self) -> tuple[ModelEntry, ...]:
        return tuple(self._entries.values())

    def register(self, entry: ModelEntry) -> ModelEntry:
        if not entry.game_key:
            raise ValueError("Model entry must include 'game_key'.")
        self._games.get(key=entry.game_key)
        key: tuple[str, str] = (entry.game_key, entry.spec.id)
        if key in self._entries:
            raise ValueError(
                f"Model already registered for {entry.game_key}: {entry.spec.id}."
            )
        self._entries[key] = entry
        return entry

    def get_entry(self, model_id: str, game_key: str | None = None) -> ModelEntry:
        if game_key is not None:
            try:
                return self._entries[(game_key, model_id)]
            except KeyError as exc:
                raise KeyError(f"Unknown model id for {game_key}: {model_id}.") from exc

        matches: list[ModelEntry] = [
            entry for entry in self._entries.values() if entry.spec.id == model_id
        ]
        if not matches:
            raise KeyError(f"Unknown model ID: {model_id}.")
        if len(matches) > 1:
            raise KeyError(
                f"Model id {model_id!r} exists for multiple games; pass game_key."
            )
        return matches[0]

    def default_model_id(self, game_key: str) -> str:
        entry: ModelEntry | None = next(
            (
                entry
                for entry in self._entries.values()
                if entry.game_key == game_key and entry.spec.is_default
            ),
            None,
        )
        if entry is None:
            raise KeyError(f"No default model registered for game: {game_key}.")
        return entry.spec.id

    def public_entries(self, game_key: str | None = None) -> list[dict[str, Any]]:
        return [
            entry.public()
            for entry in self._entries.values()
            if self._games.get(key=entry.game_key).is_model_public(spec=entry.spec)
            and (game_key is None or entry.game_key == game_key)
        ]

    def resolve_profile(
        self,
        model_id: str,
        req_profile: str = "auto",
        game_key: str | None = None,
    ) -> str:
        entry: ModelEntry = self.get_entry(model_id=model_id, game_key=game_key)
        game = self._games.get(key=entry.game_key)
        if req_profile != "auto":
            game.get_profile(key=req_profile)
            return req_profile

        inferred: str | None = entry.spec.profile or next(
            (
                profile
                for value in (entry.spec.name, entry.spec.src, entry.spec.filename)
                if (profile := game.profile_key(value)) is not None
            ),
            None,
        )
        if inferred is not None:
            game.get_profile(key=inferred)
            return inferred
        return game.default_profile_key
