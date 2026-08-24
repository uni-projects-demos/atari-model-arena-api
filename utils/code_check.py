from subprocess import run
from sys import executable


def execute(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}")
    run(command, check=True)


def main() -> None:
    # format
    execute(
        [
            executable,
            "-m",
            "black",
            "api/.",
            "utils/.",
        ]
    )

    # lint
    execute(
        [
            executable,
            "-m",
            "ruff",
            "check",
            "api/.",
            "utils/.",
        ]
    )

    # type check
    execute(
        [
            executable,
            "-m",
            "mypy",
            "api",
            "utils",
        ]
    )

    print("\nAll checks pass.")
