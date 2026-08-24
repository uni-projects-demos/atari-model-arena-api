from . import games
from .games import GAME_REGISTER, discover_games
from .policies import PolicyManager

policies: PolicyManager | None = None


def start_services() -> PolicyManager:
    global policies
    if policies is None:
        discover_games(pkg_name=games.__name__, paths=games.__path__)
        policies = PolicyManager(games=GAME_REGISTER)
    return policies


def get_policies() -> PolicyManager:
    return start_services()


def close_services() -> None:
    global policies
    if policies is not None:
        policies.close()
        policies = None
