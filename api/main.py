from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import router
from .services import close_services, start_services


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_services()
    try:
        yield
    finally:
        close_services()


app: FastAPI = FastAPI(
    title="Atari Model Arena API",
    lifespan=lifespan,
)
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=list[str](settings.allow_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=router)
