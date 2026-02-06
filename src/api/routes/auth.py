import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import get_current_user
from src.core.config import get_settings
from src.core.db import get_db
from src.core.security import create_access_token, hash_password, verify_password
from src.models.user import User
from src.schemas.auth import Token, UserCreate, UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=UserPublic)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        user = User(email=payload.email, hashed_password=hash_password(payload.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserPublic(id=user.id, email=user.email)
    except Exception as e:
        db.rollback()
        logger.exception("Register failed")
        # In local/dev return error details for faster iteration
        if settings.ENV in ("local", "dev"):
            raise HTTPException(status_code=500, detail=f"Register failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.email == form_data.username)).scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password"
        )
    token = create_access_token(subject=user.id)
    return Token(access_token=token)


@router.get("/me", response_model=UserPublic)
def me(current_user: User = Depends(get_current_user)):
    return UserPublic(id=current_user.id, email=current_user.email)

