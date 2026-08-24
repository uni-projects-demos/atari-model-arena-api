from collections.abc import Iterable
from io import BytesIO
from os import unlink
from pathlib import Path
from re import IGNORECASE, sub
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import numpy as np
import stable_baselines3
from gymnasium import spaces
from huggingface_hub import hf_hub_download
from stable_baselines3 import A2C, DQN, PPO
from torch import load

from ..models import ModelEntry
from .runtime import DEVICE, AtariDQN

SB3_ALGOS: dict[str, str] = {
    "sb3-ppo": "PPO",
    "sb3-dqn": "DQN",
    "sb3-a2c": "A2C",
}


class ModelLoader:

    def __init__(self, entries: Iterable[ModelEntry] = ()) -> None:
        self._load_locks: dict[tuple[str, str], Lock] = {}
        for entry in entries:
            self.register(entry=entry)

    def register(self, entry: ModelEntry) -> None:
        key: tuple[str, str] = (entry.game_key, entry.spec.id)
        self._load_locks[key] = Lock()

    def load_bundled(self, entry: ModelEntry) -> Any:
        key: tuple[str, str] = (entry.game_key, entry.spec.id)
        with self._load_locks[key]:
            return _load_bundled(entry=entry)


def _linear_schedule(initial: float):
    def schedule(progress_remaining: float) -> float | int:
        return float(progress_remaining) * float(initial)

    return schedule


def _sb3_custom_obs(entry: ModelEntry) -> dict[str, Any]:
    custom_obs: dict[str, Any] = {
        "observation_space": spaces.Box(
            low=0,
            high=255,
            shape=entry.spec.obs_shape,
            dtype=np.uint8,
        ),
        "action_space": spaces.Discrete[np.integer[Any]](n=entry.spec.n_actions),
    }
    for object_name, initial_value in entry.spec.sb3_overrides:
        schedule = _linear_schedule(initial=initial_value)
        custom_obs[object_name] = schedule
        if object_name == "learning_rate":
            custom_obs["lr_schedule"] = schedule

    if entry.spec.type == "sb3-dqn":
        custom_obs.update(
            {
                "buffer_size": 1,
                "optimize_memory_usage": False,
                "replay_buffer_kwargs": {},
            }
        )
    return custom_obs


def _load_bundled(entry: ModelEntry) -> Any:
    if entry.base is not None:
        return entry.base
    if not entry.spec.repo_id or not entry.spec.filename:
        raise ValueError(f"Model {entry.spec.id} download source is not configured.")

    try:
        algorithm: Any = getattr(stable_baselines3, SB3_ALGOS[entry.spec.type])
    except KeyError as exc:
        raise ValueError(f"Unsupported bundled SB3 kind: {entry.spec.type}") from exc

    try:
        path: str = hf_hub_download(
            repo_id=entry.spec.repo_id,
            filename=entry.spec.filename,
        )
        entry.base = algorithm.load(
            path=path,
            device=str(DEVICE),
            custom_objects=_sb3_custom_obs(entry),
        )
        entry.err = None
        return entry.base
    except Exception as exc:
        entry.err = f"{type(exc).__name__}: {exc}"
        raise


def _load_native_upload(data: bytes) -> tuple[AtariDQN, int, str | None]:
    payload = load(f=BytesIO(initial_bytes=data), map_location="cpu", weights_only=True)
    model_name: str | None = None

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict: Any = payload["state_dict"]
        n_actions: int = int(payload.get("n_actions", 6))
        for key in ("model_name", "name"):
            value: Any | None = payload.get(key)
            if isinstance(value, str) and value.strip():
                model_name = value.strip()
                break
    elif isinstance(payload, dict):
        state_dict, n_actions = payload, 6
    else:
        raise ValueError(
            "Expected a state_dict or {'state_dict': ..., 'n_actions': N} checkpoint."
        )

    model: AtariDQN = AtariDQN(n_actions=n_actions)
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.to(DEVICE).eval()
    return model, n_actions, model_name


def _uploaded_model_name(filename: str | None, embedded_name: str | None = None) -> str:
    if embedded_name and embedded_name.strip():
        return embedded_name.strip()
    stem: str = Path(filename or "model").stem
    stem = sub(
        pattern=r"^(?:ppo|dqn|a2c|qrdqn|sac|td3)[-_]",
        repl="",
        string=stem,
        flags=IGNORECASE,
    )
    return stem or "Uploaded model"


def _load_sb3_upload(data: bytes):
    with NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(data)
        path: str = tmp.name
    try:
        for algorithm in (PPO, DQN, A2C):
            try:
                return algorithm.load(path=path, device=str(DEVICE))
            except Exception as exc:  # noqa: BLE001
                err = exc
        if err:
            raise ValueError(f"Failed to load SB3 PPO/DQN/A2C archive: {err}.")
    finally:
        unlink(path=path)
