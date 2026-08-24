from ..base import (
    GAME_MODE_KEY,
    USER_MODE_KEY,
    CompatProfile,
    MatchMode,
    UserControls,
    get_profile_key,
)

PONG_PROFILES: dict[str, CompatProfile] = {
    "pong-v0": CompatProfile(
        key="pong-v0",
        label="Pong-v0",
        frame_skip=(2, 5),
        repeat_action_prob=0.25,
    ),
    "pong-noframeskip-v0": CompatProfile(
        key="pong-noframeskip-v0",
        label="PongNoFrameskip-v0",
        frame_skip=1,
        repeat_action_prob=0.25,
    ),
    "pong-deterministic-v0": CompatProfile(
        key="pong-deterministic-v0",
        label="PongDeterministic-v0",
        frame_skip=4,
        repeat_action_prob=0.25,
    ),
    "pong-v4": CompatProfile(
        key="pong-v4",
        label="Pong-v4",
        frame_skip=(2, 5),
        repeat_action_prob=0.0,
    ),
    "pong-noframeskip-v4": CompatProfile(
        key="pong-noframeskip-v4",
        label="PongNoFrameskip-v4 · RL-Zoo",
        frame_skip=4,
        repeat_action_prob=0.0,
    ),
    "pong-deterministic-v4": CompatProfile(
        key="pong-deterministic-v4",
        label="PongDeterministic-v4",
        frame_skip=4,
        repeat_action_prob=0.0,
    ),
    "ale-pong-v5": CompatProfile(
        key="ale-pong-v5",
        label="ALE/Pong-v5",
        frame_skip=4,
        repeat_action_prob=0.25,
    ),
}

PONG_MODES: dict[str, MatchMode] = {
    USER_MODE_KEY: MatchMode(
        key=USER_MODE_KEY,
        left_label="MODEL",
        right_label="YOU",
        is_user=True,
        user_controls=UserControls(
            move="↑/↓ or W/S",
            fire="SPACE",
        ),
    ),
    GAME_MODE_KEY: MatchMode(key=GAME_MODE_KEY, left_label="GAME", right_label="MODEL"),
}


PONG_ACTIONS: dict[int, str] = {
    0: "NOOP",
    1: "FIRE",
    2: "MOVE-RIGHT",
    3: "MOVE-LEFT",
    4: "MOVE-RIGHT + FIRE",
    5: "MOVE-LEFT + FIRE",
}


def get_pong_profile_key(val: str | None) -> str | None:
    return get_profile_key(
        val=val,
        hints=(
            ("pongnoframeskipv0", "pong-noframeskip-v0"),
            ("pongdeterministicv0", "pong-deterministic-v0"),
            ("pongnoframeskipv4", "pong-noframeskip-v4"),
            ("pongdeterministicv4", "pong-deterministic-v4"),
            ("alepongv5", "ale-pong-v5"),
            ("pongv0", "pong-v0"),
            ("pongv4", "pong-v4"),
        ),
    )
