from collections import deque
from typing import Any

import cv2 as cv
import numpy as np
from torch import Tensor, cuda, device, float32, from_numpy, inference_mode, nn

from ..models import RuntimePolicy

DEVICE: device = device(device="cuda" if cuda.is_available() else "cpu")


class AtariDQN(nn.Module):
    def __init__(self, n_actions: int = 6) -> None:
        super().__init__()
        self._features: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )
        self._head: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._head(self._features(x))


class FrameStack84:
    def __init__(self, is_mirror: bool = False) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=4)
        self._is_mirror: bool = is_mirror

    def reset(self) -> None:
        self._frames.clear()

    @staticmethod
    def _gray84(frame: np.ndarray) -> np.ndarray:
        return np.asarray(
            a=cv.resize(
                src=cv.cvtColor(src=frame, code=cv.COLOR_RGB2GRAY),
                dsize=(84, 84),
                interpolation=cv.INTER_AREA,
            ),
            dtype=np.uint8,
        )

    def push(self, rgb: np.ndarray) -> np.ndarray:
        frame: np.ndarray = np.asarray(a=rgb, dtype=np.uint8)
        if self._is_mirror:
            frame = np.flip(m=frame, axis=1)
        gray: np.ndarray = self._gray84(frame=frame)
        if self._frames:
            self._frames.append(gray)
        else:
            zero: np.ndarray = np.zeros_like(a=gray)
            self._frames.extend((zero.copy(), zero.copy(), zero.copy(), gray))
        return np.stack(arrays=self._frames, axis=-1)


class ModelRuntime(RuntimePolicy):
    def __init__(self, model: Any, backend: str, is_mirror: bool = False) -> None:
        self._model: Any = model
        self._backend: str = backend
        self._stack: FrameStack84 = FrameStack84(is_mirror=is_mirror)

    def reset(self) -> None:
        self._stack.reset()

    @inference_mode()
    def predict(self, rgb: np.ndarray) -> int:
        obs: np.ndarray = self._stack.push(rgb=rgb)
        if self._backend == "native-dqn":
            tensor: Tensor = (
                from_numpy(ndarray=obs.copy())
                .permute(dims=(2, 0, 1))
                .unsqueeze(dim=0)
                .to(DEVICE, dtype=float32)
                / 255.0
            )
            return int(self._model(tensor).argmax(dim=1).item())
        elif self._backend == "sb3":
            if tuple[int](
                getattr(self._model.observation_space, "shape", (84, 84, 4))
            ) == (
                4,
                84,
                84,
            ):
                obs = np.transpose(a=obs, axes=(2, 0, 1))
            action, _ = self._model.predict(observation=obs, deterministic=True)
            return int(np.asarray(a=action).reshape(-1)[0])

        raise RuntimeError(f"Unsupported runtime backend: {self._backend}.")
