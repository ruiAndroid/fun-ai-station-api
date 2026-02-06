from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.router import api_router
from src.core.config import get_settings
from src.core.db import SessionLocal
from src.models.agent import Agent
from src.seed.default_agents import DEFAULT_AGENTS

settings = get_settings()

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
def seed_default_data():
    # Local/dev convenience: ensure default agents exist
    if settings.ENV not in ("local", "dev"):
        return

    db: Session = SessionLocal()
    try:
        existing_any = db.execute(select(Agent.id).limit(1)).scalar_one_or_none()
        if existing_any:
            return
        for a in DEFAULT_AGENTS:
            db.add(Agent(**a))
        db.commit()
    finally:
        db.close()

