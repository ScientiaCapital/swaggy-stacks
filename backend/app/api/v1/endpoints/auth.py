"""
Authentication API endpoints
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.auth import authenticate_user, create_access_token, get_password_hash
from app.core.config import settings
from app.core.database import get_db

router = APIRouter()

# OAuth2 password bearer for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    username: str = None


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True


@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]},
        expires_delta=access_token_expires,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@router.post("/register", response_model=UserResponse, status_code=201)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user
    Note: In production, implement proper user creation in database
    """
    # For demo purposes, return success
    # In production, check if user exists, create in database, etc.

    return {
        "id": 1,
        "username": user_data.username,
        "email": user_data.email,
        "is_active": True,
    }


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user=Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """
    Get current user information
    """
    # This would use get_current_user dependency in production
    return {
        "id": 1,
        "username": "demo_user",
        "email": "demo@example.com",
        "is_active": True,
    }
