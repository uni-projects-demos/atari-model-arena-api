from ..base import (
    GAME_MODE_KEY,
    USER_GAME_MODE_KEY,
    CompatProfile,
    MatchMode,
    PlayerSlot,
    UserControls,
    get_profile_key,
)

BREAKOUT_PROFILES: dict[str, CompatProfile] = {
    "breakout-noframeskip-v4": CompatProfile(
        key="breakout-noframeskip-v4",
        label="BreakoutNoFrameskip-v4 · RL-Zoo",
        frame_skip=4,
        repeat_action_prob=0.0,
    ),
    "ale-breakout-v5": CompatProfile(
        key="ale-breakout-v5",
        label="ALE/Breakout-v5",
        frame_skip=4,
        repeat_action_prob=0.25,
    ),
}

BREAKOUT_MODES: dict[str, MatchMode] = {
    GAME_MODE_KEY: MatchMode(
        key=GAME_MODE_KEY,
        left_label="GAME",
        right_label="MODEL",
        players=(PlayerSlot("first_0", "model", side="right"),),
    ),
    USER_GAME_MODE_KEY: MatchMode(
        key=USER_GAME_MODE_KEY,
        left_label="GAME",
        right_label="YOU",
        is_user=True,
        user_controls=UserControls(move="←/→ or A/D", fire="SPACE"),
        players=(PlayerSlot("first_0", "human", side="right"),),
    ),
}

BREAKOUT_ACTIONS: dict[int, str] = {
    0: "NOOP",
    1: "FIRE",
    2: "MOVE-RIGHT",
    3: "MOVE-LEFT",
}


def get_breakout_profile_key(val: str | None) -> str | None:
    return get_profile_key(
        val=val,
        hints=(
            ("breakoutnoframeskipv4", "breakout-noframeskip-v4"),
            ("alebreakoutv5", "ale-breakout-v5"),
        ),
    )
