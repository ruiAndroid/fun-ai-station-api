from fastapi import APIRouter

from src.api.routes import agents, auth, chat, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(agents.router)
api_router.include_router(chat.router)

