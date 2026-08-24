from fastapi import APIRouter

from .info import router as info_router
from .models import router as models_router
from .play import router as play_router

router = APIRouter()
router.include_router(info_router)
router.include_router(models_router)
router.include_router(play_router)

__all__ = ["router"]
