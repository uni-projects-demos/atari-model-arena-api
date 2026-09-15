from collections.abc import Callable

import numpy as np

from ...models import RuntimePolicy
from ..base import MatchController, MatchSnapshot
from .ale import SinglePlayerALE
from .user import UserPolicy


class SinglePlayerModelController(MatchController):
    def __init__(
        self,
        profile_key: str,
        policy: RuntimePolicy,
        env_factory: Callable[[str], SinglePlayerALE],
    ):
        super().__init__(policy=policy)
        self._env: SinglePlayerALE = env_factory(profile_key)
        self._obs: np.ndarray | None = None

    def _frame(self) -> np.ndarray:
        if self._obs is not None:
            return np.asarray(a=self._obs, dtype=np.uint8)
        return np.zeros(shape=(210, 160, 3), dtype=np.uint8)

    def set_user(self, direction: str) -> None:
        if isinstance(self._policy, UserPolicy):
            self._policy.set_user(direction)

    def _snapshot(self, is_done: bool = False) -> MatchSnapshot:
        return MatchSnapshot(
            self._frame(),
            self._left_score,
            self._right_score,
            self._model_action,
            is_done,
            {"first_0": self._model_action},
        )

    def step(self) -> MatchSnapshot:
        self._model_action: int = int(self._policy.predict(rgb=self._frame()))
        self._obs, reward, is_terminated, is_truncated, info = self._env.step(
            action=self._model_action
        )
        if info.get("life_lost"):
            self._policy.reset()

        reward = float(reward)
        if reward > 0:
            self._right_score += reward
        elif reward < 0:
            self._left_score -= reward
        return self._snapshot(is_done=is_terminated or is_truncated)

    def reset(self) -> MatchSnapshot:
        self._reset_scores()
        self._obs, _ = self._env.reset()
        return self._snapshot()

    def close(self) -> None:
        self._env.close()
