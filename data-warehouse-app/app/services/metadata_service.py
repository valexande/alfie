from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from fastapi import HTTPException
from ..core.database import get_database
from ..models.dataset import DatasetMetadata, DatasetCreate, DatasetUpdate, DatasetResponse
from ..models.user import User, UserCreate, UserUpdate, UserResponse
import logging

logger = logging.getLogger(__name__)


class MetadataService:
    def __init__(self):
        self.db = None

    async def initialize(self):
        """Initialize the metadata service"""
        self.db = get_database()

    # Dataset operations
    async def create_dataset_metadata(
        self, 
        dataset_data: DatasetCreate, 
        file_metadata: Dict[str, Any]
    ) -> DatasetMetadata:
        """Create new dataset metadata"""
        try:
            # Combine dataset data with file metadata
            metadata_dict = {
                **dataset_data.dict(),
                **file_metadata,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            # Create DatasetMetadata object
            dataset_metadata = DatasetMetadata(**metadata_dict)
            
            # Insert into database
            result = await self.db.datasets.insert_one(dataset_metadata.dict(by_alias=True, exclude={'id'}))
            
            # Retrieve the created document
            created_dataset = await self.db.datasets.find_one({"_id": result.inserted_id})
            
            logger.info(f"Created dataset metadata: {dataset_data.dataset_id}")
            return DatasetMetadata(**created_dataset)
            
        except DuplicateKeyError:
            raise HTTPException(
                status_code=400,
                detail="A dataset with this user_id, dataset_id, and version already exists"
            )
        except Exception as e:
            logger.error(f"Error creating dataset metadata: {e}")
            raise HTTPException(status_code=500, detail="Failed to create dataset metadata")

    async def get_dataset_by_id(self, dataset_id: str, user_id: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by dataset_id and user_id"""
        try:
            dataset = await self.db.datasets.find_one({
                "dataset_id": dataset_id,
                "user_id": user_id
            })
            
            if dataset:
                return DatasetMetadata(**dataset)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving dataset {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve dataset")

    async def get_latest_dataset(self, dataset_id: str, user_id: str) -> Optional[DatasetMetadata]:
        """Get the latest version of a dataset for a user by created_at desc"""
        try:
            cursor = self.db.datasets.find({
                "dataset_id": dataset_id,
                "user_id": user_id
            }).sort("created_at", -1).limit(1)
            results = await cursor.to_list(length=1)
            if results:
                return DatasetMetadata(**results[0])
            return None
        except Exception as e:
            logger.error(f"Error retrieving latest dataset {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve dataset")

    async def get_dataset_by_version(self, dataset_id: str, user_id: str, version: str) -> Optional[DatasetMetadata]:
        """Get a specific version of a dataset for a user"""
        try:
            dataset = await self.db.datasets.find_one({
                "dataset_id": dataset_id,
                "user_id": user_id,
                "version": version
            })
            if dataset:
                return DatasetMetadata(**dataset)
            return None
        except Exception as e:
            logger.error(f"Error retrieving dataset {dataset_id} version {version}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve dataset")

    async def get_datasets_by_user(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[DatasetMetadata]:
        """Get all datasets for a user with pagination"""
        try:
            cursor = self.db.datasets.find({"user_id": user_id}).skip(skip).limit(limit)
            datasets = await cursor.to_list(length=limit)
            
            return [DatasetMetadata(**dataset) for dataset in datasets]
            
        except Exception as e:
            logger.error(f"Error retrieving datasets for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve datasets")

    async def update_dataset_metadata(
        self, 
        dataset_id: str, 
        user_id: str, 
        update_data: DatasetUpdate
    ) -> Optional[DatasetMetadata]:
        """Update dataset metadata"""
        try:
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            update_dict['updated_at'] = datetime.utcnow()
            
            result = await self.db.datasets.update_one(
                {"dataset_id": dataset_id, "user_id": user_id},
                {"$set": update_dict}
            )
            
            if result.matched_count == 0:
                return None
                
            # Return updated dataset
            updated_dataset = await self.db.datasets.find_one({
                "dataset_id": dataset_id,
                "user_id": user_id
            })
            
            return DatasetMetadata(**updated_dataset)
            
        except Exception as e:
            logger.error(f"Error updating dataset {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update dataset")

    async def delete_dataset_metadata(self, dataset_id: str, user_id: str) -> bool:
        """Delete dataset metadata"""
        try:
            result = await self.db.datasets.delete_one({
                "dataset_id": dataset_id,
                "user_id": user_id
            })
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete dataset")

    async def delete_latest_dataset_metadata(self, dataset_id: str, user_id: str) -> Optional[DatasetMetadata]:
        """Delete the latest version and return it if deleted"""
        try:
            latest = await self.get_latest_dataset(dataset_id, user_id)
            if not latest:
                return None
            result = await self.db.datasets.delete_one({"_id": latest.id})
            return latest if result.deleted_count == 1 else None
        except Exception as e:
            logger.error(f"Error deleting latest dataset {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete dataset")

    async def delete_dataset_by_version(self, dataset_id: str, user_id: str, version: str) -> bool:
        """Delete a specific version of a dataset"""
        try:
            result = await self.db.datasets.delete_one({
                "dataset_id": dataset_id,
                "user_id": user_id,
                "version": version
            })
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id} version {version}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete dataset")

    async def delete_all_dataset_versions(self, dataset_id: str, user_id: str) -> int:
        """Delete all versions of a dataset and return the count of deleted documents"""
        try:
            result = await self.db.datasets.delete_many({
                "dataset_id": dataset_id,
                "user_id": user_id
            })
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting all versions of dataset {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete all dataset versions")

    async def get_dataset_versions(self, dataset_id: str, user_id: str) -> List[DatasetMetadata]:
        """Get all versions of a dataset"""
        try:
            # Extract base dataset name (remove version suffix if present)
            base_dataset_id = dataset_id.split('_v')[0] if '_v' in dataset_id else dataset_id
            
            cursor = self.db.datasets.find({
                "user_id": user_id,
                "dataset_id": {"$regex": f"^{base_dataset_id}"}
            }).sort("created_at", -1)
            
            datasets = await cursor.to_list(length=None)
            return [DatasetMetadata(**dataset) for dataset in datasets]
            
        except Exception as e:
            logger.error(f"Error retrieving dataset versions: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve dataset versions")

    # User operations
    async def create_user(self, user_data: UserCreate) -> User:
        """Create new user"""
        try:
            user_dict = user_data.dict()
            # Remove password from stored data (in real app, hash the password)
            user_dict.pop('password', None)
            user_dict['created_at'] = datetime.utcnow()
            user_dict['updated_at'] = datetime.utcnow()
            
            user = User(**user_dict)
            result = await self.db.users.insert_one(user.dict(by_alias=True, exclude={'id'}))
            
            created_user = await self.db.users.find_one({"_id": result.inserted_id})
            logger.info(f"Created user: {user_data.user_id}")
            
            return User(**created_user)
            
        except DuplicateKeyError:
            raise HTTPException(status_code=400, detail="User ID, username, or email already exists")
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Failed to create user")

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by user_id"""
        try:
            user = await self.db.users.find_one({"user_id": user_id})
            if user:
                return User(**user)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user")

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            user = await self.db.users.find_one({"username": username})
            if user:
                return User(**user)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving user by username {username}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user")

    async def update_user(self, user_id: str, update_data: UserUpdate) -> Optional[User]:
        """Update user information"""
        try:
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            update_dict['updated_at'] = datetime.utcnow()
            
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": update_dict}
            )
            
            if result.matched_count == 0:
                return None
                
            updated_user = await self.db.users.find_one({"user_id": user_id})
            return User(**updated_user)
            
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update user")

    async def search_datasets(
        self, 
        user_id: str,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        file_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DatasetMetadata]:
        """Search datasets with various filters, returning only the latest version of each dataset"""
        try:
            # Build match stage for filtering
            match_stage = {"user_id": user_id}
            
            if query:
                match_stage["$or"] = [
                    {"name": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}},
                    {"dataset_id": {"$regex": query, "$options": "i"}}
                ]
            
            if tags:
                match_stage["tags"] = {"$in": tags}
                
            if file_type:
                match_stage["file_type"] = file_type
            
            # Use aggregation pipeline to get only the latest version of each dataset
            pipeline = [
                {"$match": match_stage},
                {"$sort": {"created_at": -1}},
                {
                    "$group": {
                        "_id": "$dataset_id",
                        "doc": {"$first": "$$ROOT"}
                    }
                },
                {"$replaceRoot": {"newRoot": "$doc"}},
                {"$sort": {"created_at": -1}},
                {"$skip": skip},
                {"$limit": limit}
            ]
            
            cursor = self.db.datasets.aggregate(pipeline)
            datasets = await cursor.to_list(length=limit)
            
            return [DatasetMetadata(**dataset) for dataset in datasets]
            
        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            raise HTTPException(status_code=500, detail="Failed to search datasets")


# Global metadata service instance
metadata_service = MetadataService()
