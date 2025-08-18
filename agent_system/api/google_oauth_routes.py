from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
import requests
from typing import Optional
import json

from .auth_models import UserResponse, AuthResponse
from ..models.database import get_db, User
from ..config.settings import settings

router = APIRouter(prefix="/auth/google", tags=["Google OAuth"])

def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token for user"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt

@router.get("/login")
async def google_login():
    """Initiate Google OAuth login"""
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={settings.google_client_id}&"
        f"redirect_uri={settings.google_redirect_uri}&"
        "response_type=code&"
        "scope=openid email profile&"
        "access_type=offline"
    )
    return RedirectResponse(url=google_auth_url)

@router.get("/callback")
async def google_callback(code: str, db: Session = Depends(get_db)):
    """Handle Google OAuth callback"""
    try:
        # Exchange authorization code for access token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.google_redirect_uri,
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        token_info = token_response.json()
        
        access_token = token_info["access_token"]
        
        # Get user info from Google
        user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = requests.get(user_info_url, headers=headers)
        user_response.raise_for_status()
        google_user = user_response.json()
        
        # Extract user information
        google_id = google_user["id"]
        email = google_user["email"]
        name = google_user.get("name", "")
        picture = google_user.get("picture", "")
        
        # Check if user exists
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            # Create new user
            user = User(
                email=email,
                username=name or email.split("@")[0],
                password_hash="google_oauth",  # Special marker for OAuth users
                created_at=datetime.utcnow()
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Create JWT token
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = create_jwt_token(token_data)
        
        # Return user info and token
        user_response = UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            created_at=user.created_at
        )
        
        return AuthResponse(
            success=True,
            user=user_response,
            access_token=access_token,
            token_type="bearer"
        )
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to authenticate with Google: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )

@router.get("/profile")
async def get_google_profile(request: Request, db: Session = Depends(get_db)):
    """Get Google user profile (for testing)"""
    # This endpoint can be used to test the OAuth flow
    return {"message": "Google OAuth is configured and working"}
