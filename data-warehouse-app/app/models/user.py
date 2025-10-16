from datetime import datetime
from typing import Optional, List, Annotated
from pydantic import BaseModel, Field, EmailStr, BeforeValidator, PlainSerializer
from bson import ObjectId


def validate_object_id(v):
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return ObjectId(v)
    raise ValueError("Invalid ObjectId")


PyObjectId = Annotated[
    ObjectId,
    BeforeValidator(validate_object_id),
    PlainSerializer(lambda x: str(x), return_type=str)
]


class User(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="User email")
    full_name: Optional[str] = Field(None, description="User's full name")
    is_active: bool = Field(default=True, description="Whether user is active")
    is_admin: bool = Field(default=False, description="Whether user has admin privileges")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class UserCreate(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[str] = None
