from subprocess import run

from api.config import settings


def main() -> None:
    if not settings.rom_dir.exists():
        settings.rom_dir.mkdir(parents=True, exist_ok=True)

        print("Installing Atari ROMs...")
        run(
            args=[
                "AutoROM",
                "--accept-license",
                "--install-dir",
                str(settings.rom_dir.resolve()),
            ],
            check=True,
        )
