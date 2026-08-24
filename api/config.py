from dataclasses import dataclass
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    inference_workers: int  # number of concurrent processes
    max_model_size: int  # model upload size limit
    allow_origins: tuple[str, ...]  # URLs granted API access
    rom_dir: Path  # where Atari ROMs are installed

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            inference_workers=max(0, int(getenv("INFERENCE_WORKERS", "0"))),
            max_model_size=100 * 1024 * 1024,
            allow_origins=("http://localhost:5173",),
            rom_dir=(Path(__file__).resolve().parent.parent / ".autorom").resolve(),
        )


settings: Settings = Settings.from_env()
