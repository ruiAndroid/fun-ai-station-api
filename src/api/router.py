from fastapi import APIRouter

from src.api.routes import agent_service, agents, auth, chat, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(agents.router)
api_router.include_router(chat.router)
api_router.include_router(agent_service.router)

