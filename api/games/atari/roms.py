from pathlib import Path

from ale_py import roms

from ...config import settings


def rom_candidates(rom_name: str, base: Path | None) -> tuple[Path, ...]:
    base = base or settings.rom_dir
    filename: str = f"{rom_name}.bin"
    try:
        path_name: Path | None = roms.get_rom_path(name=rom_name)
        packaged_rom: Path | None = (
            Path(path_name).resolve() if path_name is not None else None
        )
    except (FileNotFoundError, KeyError, ValueError):
        packaged_rom = None

    candidates: tuple[Path, ...] = (
        base / filename,
        base / "roms" / filename,
        base / "ROM" / rom_name / filename,
    )
    return ((packaged_rom,) if packaged_rom is not None else ()) + candidates


def find_rom(rom_name: str, base: Path | None = None) -> Path | None:
    return next(
        (
            path
            for path in rom_candidates(rom_name=rom_name, base=base)
            if path.is_file()
        ),
        None,
    )


def missing_rom_msg(rom_name: str, label: str) -> str:
    return f"{label} ROM ({rom_name}.bin) not found."


def status(
    rom_name: str,
    label: str,
) -> tuple[bool, str | None]:
    is_find_rom = find_rom(rom_name=rom_name) is not None
    return is_find_rom, (
        None if is_find_rom else missing_rom_msg(rom_name=rom_name, label=label)
    )
