from collections import deque
from typing import ClassVar

import numpy as np

from ...models import RuntimePolicy
from ..atari import SinglePlayerModelController
from ..base import MatchController, MatchSnapshot
from .environment import (
    UserVsModelPong,
    game_vs_model_pong,
    user_vs_model_pong,
)


class UserVsModelPongController(MatchController):
    USER_ACTIONS: ClassVar[dict[str, int]] = {
        "none": 0,
        "fire": 1,
        "up": 2,
        "down": 3,
        "up_fire": 4,
        "down_fire": 5,
    }

    def __init__(self, profile_key: str, policy: RuntimePolicy) -> None:
        super().__init__(policy=policy)
        self._env: UserVsModelPong = user_vs_model_pong(profile_key=profile_key)
        self._obs: dict[str, np.ndarray] = {}
        self._user_action: int = 0
        self._serve_actions: deque[int] = deque()

    def _queue_serve(self) -> None:
        self._serve_actions.clear()
        self._serve_actions.extend((1, 2))

    def _frame(self) -> np.ndarray:
        if self._obs:
            return np.asarray(
                a=self._obs[
                    "first_0" if "first_0" in self._obs else next(iter(self._obs))
                ],
                dtype=np.uint8,
            )
        return np.zeros(shape=(210, 160, 3), dtype=np.uint8)

    def set_human(self, direction: str) -> None:
        self._user_action = self.USER_ACTIONS.get(direction, 0)

    def set_user(self, direction: str) -> None:
        self.set_human(direction=direction)

    def step(self) -> MatchSnapshot:
        if not getattr(self._env, "agents", []):
            return self._snapshot(is_done=True)

        actions: dict[str, int] = {}
        for agent in self._env.agents:
            if agent == "first_0":
                actions[agent] = int(self._user_action)
            elif agent == "second_0":
                action: int = (
                    self._serve_actions.popleft()
                    if self._serve_actions
                    else int(
                        self._policy.predict(
                            rgb=np.asarray(a=self._obs[agent], dtype=np.uint8)
                        )
                    )
                )
                self._model_action = action
                actions[agent] = self._model_action
            else:
                actions[agent] = 0

        self._obs, rewards, is_terminated_dict, is_truncated_dict, _ = self._env.step(
            actions=actions
        )
        user_reward = float(rewards.get("first_0", 0.0))
        model_reward = float(rewards.get("second_0", 0.0))
        if model_reward > 0:
            self._left_score += model_reward
        if user_reward > 0:
            self._right_score += user_reward
        if user_reward != 0.0 or model_reward != 0.0:
            self._queue_serve()

        is_done_agents_set: set = set(is_terminated_dict) | set(is_truncated_dict)
        is_done: bool = bool(is_done_agents_set) and all(
            is_terminated_dict.get(agent, False) or is_truncated_dict.get(agent, False)
            for agent in is_done_agents_set
        )
        return self._snapshot(is_done=is_done)

    def reset(self) -> MatchSnapshot:
        self._reset_scores()
        self._user_action = 0
        self._serve_actions.clear()
        self._obs, _ = self._env.reset()
        self._queue_serve()
        return self._snapshot()

    def close(self) -> None:
        self._env.close()


class GameVsModelPongController(SinglePlayerModelController):
    def __init__(self, profile_key: str, policy: RuntimePolicy) -> None:
        super().__init__(
            profile_key=profile_key, policy=policy, env_factory=game_vs_model_pong
        )
