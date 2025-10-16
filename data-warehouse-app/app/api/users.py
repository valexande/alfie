from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from ..models.user import UserCreate, UserUpdate, UserResponse, User
from ..services.metadata_service import metadata_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=UserResponse)
async def create_user(user_data: UserCreate):
    """Create a new user"""
    try:
        user = await metadata_service.create_user(user_data)
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user by user_id"""
    user = await metadata_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.get("/username/{username}", response_model=UserResponse)
async def get_user_by_username(username: str):
    """Get user by username"""
    user = await metadata_service.get_user_by_username(username)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, update_data: UserUpdate):
    """Update user information"""
    updated_user = await metadata_service.update_user(user_id, update_data)
    
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=updated_user.user_id,
        username=updated_user.username,
        email=updated_user.email,
        full_name=updated_user.full_name,
        is_active=updated_user.is_active,
        is_admin=updated_user.is_admin,
        created_at=updated_user.created_at,
        updated_at=updated_user.updated_at
    )
