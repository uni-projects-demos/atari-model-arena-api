import numpy as np

from ...models import RuntimePolicy


class TrackingRuntime(RuntimePolicy):
    def __init__(self, is_mirror: bool = False) -> None:
        self._is_mirror: bool = is_mirror

    def predict(self, rgb: np.ndarray) -> int:
        frame: np.ndarray = np.asarray(a=rgb)
        if self._is_mirror:
            frame = np.flip(m=frame, axis=1)
        region: np.ndarray = frame.mean(axis=2)[35:195, 15:145]

        ys, xs = np.where(region > 180)
        if len(xs) == 0:
            return 1

        ball_mask: np.ndarray | bool = xs < 110
        if not np.any(a=ball_mask):
            return 0
        ball_y: float | int = float(np.median(a=ys[ball_mask]))

        paddle_mask: np.ndarray | bool = xs > 115
        paddle_y: float | int = (
            float(np.median(a=ys[paddle_mask])) if np.any(a=paddle_mask) else 80.0
        )

        if ball_y < paddle_y - 3:
            return 2
        if ball_y > paddle_y + 3:
            return 3
        return 0

    def reset(self) -> None:
        pass
