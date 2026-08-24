from asyncio import AbstractEventLoop, get_running_loop
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from ..config import settings
from ..games import GAME_REGISTER, GamePlugin, GameRegister
from ..models import ModelEntry, ModelSpec, RuntimePolicy
from .catalog import ModelCatalog
from .loaders import (
    SB3_ALGOS,
    ModelLoader,
    _load_native_upload,
    _load_sb3_upload,
    _uploaded_model_name,
)
from .runtime import DEVICE, ModelRuntime

T = TypeVar("T")


class PolicyManager:
    def __init__(self, games: GameRegister = GAME_REGISTER) -> None:
        self._games: GameRegister = games
        self._catalog: ModelCatalog = ModelCatalog(games=games)
        self._loader: ModelLoader = ModelLoader(entries=self._catalog.entries())
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=_executor_workers(),
            thread_name_prefix="atari-policy",
        )
        self._loading_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="atari-model-load",
        )

    def _get_entry(self, model_id: str, game_key: str | None = None) -> ModelEntry:
        return self._catalog.get_entry(model_id=model_id, game_key=game_key)

    def _load_bundled(self, model_id: str, game_key: str | None = None) -> Any:
        entry: ModelEntry = self._get_entry(model_id=model_id, game_key=game_key)
        return self._loader.load_bundled(entry=entry)

    def register(self, entry: ModelEntry) -> ModelEntry:
        registered: ModelEntry = self._catalog.register(entry=entry)
        self._loader.register(entry=registered)
        return registered

    def default_model_id(self, game_key: str) -> str:
        return self._catalog.default_model_id(game_key=game_key)

    def public_entries(self, game_key: str | None = None) -> list[dict[str, Any]]:
        return self._catalog.public_entries(game_key=game_key)

    def resolve_profile(
        self,
        model_id: str,
        req_profile: str = "auto",
        game_key: str | None = None,
    ) -> str:
        return self._catalog.resolve_profile(
            model_id=model_id, req_profile=req_profile, game_key=game_key
        )

    def load(self, model_id: str, *, game_key: str | None = None) -> ModelEntry:
        entry: ModelEntry = self._get_entry(model_id=model_id, game_key=game_key)
        if entry.spec.type in SB3_ALGOS:
            try:
                self._load_bundled(model_id=model_id, game_key=entry.game_key)
            except Exception as exc:
                reason: str = entry.err or f"{type(exc).__name__}: {exc}."
                raise RuntimeError(
                    f"Model {entry.spec.name} failed to load: {reason}."
                ) from exc
        elif entry.spec.type in {"sb3-upload", "native-dqn"}:
            if entry.base is None:
                raise RuntimeError(f"Model {entry.spec.name} is not loaded.")
        elif entry.spec.type != "heuristic":
            raise RuntimeError(f"Unsupported model type: {entry.spec.type}.")
        return entry

    def runtime(
        self,
        model_id: str,
        *,
        is_mirror: bool = False,
        game_key: str | None = None,
    ) -> RuntimePolicy:
        entry: ModelEntry = self._get_entry(model_id=model_id, game_key=game_key)
        game: GamePlugin = self._games.get(key=entry.game_key)

        if entry.spec.type == "heuristic":
            return game.create_runtime(
                runtime_type=entry.spec.type,
                is_mirror=is_mirror,
            )
        elif entry.spec.type in SB3_ALGOS:
            if entry.base is not None:
                return ModelRuntime(
                    model=entry.base, backend="sb3", is_mirror=is_mirror
                )
            elif entry.err is not None and entry.spec.fb_type:
                return game.create_runtime(
                    runtime_type=entry.spec.fb_type,
                    is_mirror=is_mirror,
                )
            try:
                model: Any = self._load_bundled(
                    model_id=model_id, game_key=entry.game_key
                )
                return ModelRuntime(model=model, backend="sb3", is_mirror=is_mirror)
            except Exception as exc:
                reason: str = entry.err or f"{type(exc).__name__}: {exc}."
                if entry.spec.fb_type:
                    return game.create_runtime(
                        runtime_type=entry.spec.fb_type,
                        is_mirror=is_mirror,
                    )
                raise RuntimeError(
                    f"Model {entry.spec.name} failed to load: {reason}."
                ) from exc
        elif entry.spec.type == "sb3-upload":
            return ModelRuntime(model=entry.base, backend="sb3", is_mirror=is_mirror)
        elif entry.spec.type == "native-dqn":
            return ModelRuntime(
                model=entry.base, backend="native-dqn", is_mirror=is_mirror
            )
        raise RuntimeError(f"Unsupported model kind: {entry.spec.type}.")

    async def load_async(
        self,
        model_id: str,
        *,
        game_key: str | None = None,
    ) -> ModelEntry:
        return await self._run_loading(self.load, model_id, game_key=game_key)

    async def prepare_runtime(
        self,
        model_id: str,
        *,
        game_key: str | None = None,
    ) -> None:
        entry: ModelEntry = self._get_entry(model_id=model_id, game_key=game_key)
        if entry.spec.type not in SB3_ALGOS or entry.base is not None:
            return
        try:
            await self.load_async(model_id=model_id, game_key=entry.game_key)
        except RuntimeError:
            if not entry.spec.fb_type:
                raise

    async def run(self, func: Callable[..., T], /, *args, **kwargs) -> T:
        loop: AbstractEventLoop = get_running_loop()
        return await loop.run_in_executor(
            executor=self._executor, func=partial[T](func, *args, **kwargs)
        )

    async def _run_loading(self, func: Callable[..., T], /, *args, **kwargs) -> T:
        loop: AbstractEventLoop = get_running_loop()
        return await loop.run_in_executor(
            executor=self._loading_executor, func=partial[T](func, *args, **kwargs)
        )

    async def upload(
        self,
        upload_file,
        game_key: str,
    ) -> ModelEntry:
        game: GamePlugin = self._games.get(key=game_key)
        data: Any = await upload_file.read(n=settings.max_model_size + 1)
        if len(data) > settings.max_model_size:
            raise ValueError("Model file exceeds upload size limit.")

        model_id: str = f"upload-{uuid4().hex[:10]}."
        src_name: str = upload_file.filename or model_id

        suffix: str = Path(upload_file.filename or "model").suffix.lower()
        if suffix == ".zip":
            display_name: str = _uploaded_model_name(filename=src_name)
            entry: ModelEntry = ModelEntry(
                spec=ModelSpec(
                    id=model_id,
                    name=display_name,
                    type="sb3-upload",
                    src="uploaded",
                    profile=game.profile_key(val=src_name),
                ),
                base=await self._run_loading(_load_sb3_upload, data),
                game_key=game_key,
            )
        elif suffix not in {".pt", ".pth"}:
            raise ValueError("File types must be .zip, .pt or .pth.")
        else:
            model, n_actions, embedded_name = await self._run_loading(
                _load_native_upload,
                data,
            )
            display_name = _uploaded_model_name(
                filename=src_name,
                embedded_name=embedded_name,
            )
            entry = ModelEntry(
                spec=ModelSpec(
                    id=model_id,
                    name=display_name,
                    type="native-dqn",
                    src="uploaded",
                    profile=game.profile_key(val=src_name),
                    n_actions=n_actions,
                ),
                base=model,
                game_key=game_key,
            )
        return self.register(entry=entry)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._loading_executor.shutdown(wait=False, cancel_futures=True)


def _executor_workers() -> int:
    if settings.inference_workers:
        return settings.inference_workers
    elif DEVICE.type == "cuda":
        return 1
    return max(1, min(4, (cpu_count() or 2) // 2))
