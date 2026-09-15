import numpy as np


class UserPolicy:
    def __init__(self, game_key: str) -> None:
        self._actions = {"none": 0, "fire": 1, "right": 2, "left": 3}
        if game_key == "pong":
            self._actions.update(up=2, down=3, up_fire=4, down_fire=5)
        self._action = 0

    def set_user(self, direction: str) -> None:
        self._action = self._actions.get(direction, 0)

    def predict(self, rgb: np.ndarray) -> int:
        return self._action

    def reset(self) -> None:
        self._action = 0
