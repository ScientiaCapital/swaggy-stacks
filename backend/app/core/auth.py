"""
JWT Authentication utilities
"""

from datetime import datetime, timedelta
from typing import Optional, Union

from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT token
    Returns the payload if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return {"username": username, "user_id": payload.get("user_id")}
    except JWTError:
        return None


def authenticate_user(username: str, password: str, user_db) -> Union[dict, bool]:
    """
    Authenticate a user with username and password
    Returns user data if authentic, False otherwise
    """
    # This would typically query your user database
    # For now, implementing basic demo authentication

    # Demo user credentials (in production, query from database)
    demo_users = {
        "demo_user": {
            "username": "demo_user",
            "hashed_password": get_password_hash("demo_password"),
            "id": 1,
            "email": "demo@example.com",
            "is_active": True,
        }
    }

    user = demo_users.get(username)
    if not user:
        return False

    if not verify_password(password, user["hashed_password"]):
        return False

    return user


def create_jwt_exception() -> HTTPException:
    """Create a standardized JWT authentication exception"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
