import numpy as np

from ...models import RuntimePolicy


class TrackingRuntime(RuntimePolicy):
    def __init__(self, is_mirror: bool = False) -> None:
        self._is_mirror: bool = is_mirror

    def predict(self, rgb: np.ndarray) -> int:
        frame: np.ndarray = np.asarray(a=rgb)
        if self._is_mirror:
            frame = np.flip(m=frame, axis=1)
        court: np.ndarray[tuple, np.dtype] = frame[34:194]
        ball_ys, _ = np.where(np.all(court[:, 20:140] > 180, axis=2))
        if not len(ball_ys):
            return 1
        ball_y: float = float(np.median(ball_ys))

        lane: np.ndarray[tuple, np.dtype] = court[:, 140:144]
        background: np.ndarray[tuple, np.dtype] = np.median(
            court[:, 78:82], axis=(0, 1)
        )
        paddle_mask: np.ndarray[tuple, np.dtype[np.bool[bool]]] = np.any(
            np.abs(lane.astype(float) - background) > 20, axis=2
        )
        paddle_ys, _ = np.where(paddle_mask)
        if not len(paddle_ys):
            return 0
        paddle_y = float(np.median(paddle_ys))

        if ball_y < paddle_y - 3:
            return 2
        if ball_y > paddle_y + 3:
            return 3
        return 0

    def reset(self) -> None:
        pass
