from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings
import logging

logger = logging.getLogger(__name__)


class MongoDB:
    client: AsyncIOMotorClient = None
    database = None


mongodb = MongoDB()


async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb.client = AsyncIOMotorClient(settings.mongodb_url)
        mongodb.database = mongodb.client[settings.mongodb_database]
        
        # Test the connection
        await mongodb.client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
        
        # Create indexes for better performance
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        logger.info("Disconnected from MongoDB")


async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Dataset collection indexes
        datasets_collection = mongodb.database.datasets
        # Ensure old unique index on dataset_id alone is removed to support versions
        try:
            index_info = await datasets_collection.index_information()
            if "dataset_id_1" in index_info and index_info["dataset_id_1"].get("unique"):
                await datasets_collection.drop_index("dataset_id_1")
        except Exception as drop_error:
            logger.warning(f"Could not drop old dataset_id unique index (may not exist): {drop_error}")

        await datasets_collection.create_index("user_id")
        await datasets_collection.create_index([("user_id", 1), ("dataset_id", 1)])
        # Unique per user, dataset, and version
        await datasets_collection.create_index(
            [("user_id", 1), ("dataset_id", 1), ("version", 1)],
            name="user_dataset_version_unique",
            unique=True
        )
        await datasets_collection.create_index("created_at")
        
        # Users collection indexes
        users_collection = mongodb.database.users
        await users_collection.create_index("user_id", unique=True)
        await users_collection.create_index("username", unique=True)
        await users_collection.create_index("email", unique=True)
        
        # Bias reports collection indexes
        bias_collection = mongodb.database.bias_reports
        await bias_collection.create_index([("user_id", 1), ("dataset_id", 1)], unique=True)
        await bias_collection.create_index("created_at")

        # Transformation reports collection indexes
        transform_collection = mongodb.database.transformation_reports
        await transform_collection.create_index([("user_id", 1), ("dataset_id", 1), ("version", 1)], unique=True)
        await transform_collection.create_index("created_at")

        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")


def get_database():
    """Get database instance"""
    return mongodb.database
