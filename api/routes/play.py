from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Task,
    create_task,
    get_running_loop,
    wait,
)
from logging import Logger, getLogger
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..games import DEFAULT_MODE_KEY, GamePlugin, get_game
from ..policies import PolicyManager
from ..services import get_policies
from ..session import GameSession, SessionConfig

logger: Logger = getLogger(name=__name__)
router: APIRouter = APIRouter()


@router.websocket("/ws/play")
async def play(ws: WebSocket):
    ws_origin: str | None = ws.headers.get("origin")
    if (
        ws_origin
        and "*" not in settings.allow_origins
        and ws_origin not in settings.allow_origins
    ):
        await ws.close(code=1008, reason="Websocket origin is invalid.")
        return
    await ws.accept()

    game_session: GameSession | None = None
    policies: PolicyManager = get_policies()

    try:
        req_msg: Any = await ws.receive_json()
        if req_msg.get("type") != "start":
            await ws.send_json(
                data={
                    "type": "error",
                    "message": "First request message 'type' must be 'start'.",
                }
            )
            return

        game_session, fps = await _create_game_session(
            req_msg=req_msg, policies=policies
        )
        if game_session:
            await ws.send_json(data={"type": "started"})
            await _stream_game_session(
                ws=ws, game_session=game_session, policies=policies, fps=fps
            )

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json(
                data={
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        except (WebSocketDisconnect, RuntimeError, OSError) as send_exc:
            logger.debug("WebSocket error message failed: %s.", send_exc)
    finally:
        if game_session is not None:
            await policies.run(game_session.close)


async def _create_game_session(
    req_msg: Any,
    policies: PolicyManager,
) -> tuple[GameSession, int]:
    game_key: Any = req_msg.get("game")
    if not game_key:
        raise RuntimeError("Start request message must include 'game' key.")

    game: GamePlugin = get_game(key=game_key)
    mode: Any = req_msg.get("mode") or DEFAULT_MODE_KEY

    is_available, err = game.mode_status(mode_key=mode)
    if not is_available:
        raise RuntimeError(err or f"Mode '{mode}' is unavailable.")

    model_id: str = req_msg.get("model") or policies.default_model_id(game_key=game_key)
    await policies.prepare_runtime(model_id=model_id, game_key=game_key)

    game_session: GameSession = await policies.run(
        GameSession,
        SessionConfig(
            game=game_key,
            mode=mode,
            profile=req_msg.get("profile", "auto"),
            model=model_id,
        ),
        policies,
    )
    return game_session, max(5, min(int(req_msg.get("fps", 20)), 60))


async def _stream_game_session(
    ws: WebSocket,
    game_session: GameSession,
    policies: PolicyManager,
    fps: int,
) -> None:
    loop: AbstractEventLoop = get_running_loop()
    frame_interval: float | int = 1 / fps
    next_frame: float | int = loop.time()
    receive_task: Task[Any] = create_task(coro=ws.receive_json())

    try:
        while True:
            done, _ = await wait(
                {receive_task}, timeout=max(0.0, next_frame - loop.time())
            )
            if receive_task in done:
                req_msg: Any = receive_task.result()
                match req_msg.get("type"):
                    case "input":
                        game_session.set_user(
                            direction=req_msg.get("direction", "none")
                        )
                    case "reset":
                        await policies.run(game_session.reset)
                    case "stop":
                        return
                receive_task = create_task(coro=ws.receive_json())

                if loop.time() < next_frame:
                    continue

            packet = await policies.run(game_session.step)
            await ws.send_json(data=packet.metadata)
            await ws.send_bytes(data=packet.pixels)

            next_frame += frame_interval
            next_frame = max(next_frame, loop.time())
    finally:
        if not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except CancelledError:
                pass
