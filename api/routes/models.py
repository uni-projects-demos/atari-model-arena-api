from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..games import get_game
from ..policies import PolicyManager
from ..services import get_policies

router: APIRouter = APIRouter()


@router.post("/api/models/load")
async def load_model(
    game: Annotated[str, Form()],
    model: Annotated[str, Form()],
):
    policies: PolicyManager = get_policies()
    return await _model_action(
        game,
        lambda: policies.load_async(model_id=model, game_key=game),
    )


@router.post("/api/models/upload")
async def upload_model(
    file: Annotated[UploadFile, File()],
    game: Annotated[str, Form()],
):
    policies: PolicyManager = get_policies()
    return await _model_action(
        game,
        lambda: policies.upload(upload_file=file, game_key=game),
    )


async def _model_action(
    game_key: str,
    action: Callable[[], Awaitable[Any]],
) -> dict[str, str]:
    try:
        get_game(key=game_key)
        entry: Any = await action()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": entry.spec.id, "name": entry.spec.name}
