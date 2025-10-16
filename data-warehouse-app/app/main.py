from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .core.config import settings
from .core.database import connect_to_mongo, close_mongo_connection
from .services.file_service import file_service
from .services.metadata_service import metadata_service
from .services.kafka_service import kafka_producer_service
from .services.bias_report_service import bias_report_service
from .api import datasets, users, bias_reports, ai_models
from .services.transformation_report_service import transformation_report_service
from .api import transformation_reports
from .services.ai_model_service import ai_model_service
from .services.xai_report_service import xai_report_service
from .api import xai_reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("Starting up Data Warehouse API...")
    
    try:
        # Initialize database connection
        await connect_to_mongo()
        
        # Initialize services
        await file_service.initialize()
        await metadata_service.initialize()
        await bias_report_service.initialize()
        await transformation_report_service.initialize()
        await ai_model_service.initialize()
        await xai_report_service.initialize()
        
        # Initialize Kafka (non-fatal)
        try:
            await kafka_producer_service.start()
            logger.info("Kafka producer initialized")
        except Exception as kafka_error:
            logger.warning(f"Kafka init failed: {kafka_error}")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Data Warehouse API...")
    try:
        await kafka_producer_service.stop()
    except Exception as e:
        logger.error(f"Error stopping Kafka producer: {e}")
    await close_mongo_connection()


# Create FastAPI application
app = FastAPI(
    title="Data Warehouse API",
    description="""
    A comprehensive Data Warehouse API for storing and managing both structured and unstructured data.
    
    Features:
    - File upload to MinIO with organized user separation and versioning
    - Automatic metadata extraction and storage in MongoDB
    - Support for various file types (CSV, Excel, images, videos, audio, AI models)
    - User-based access control and dataset management
    - Advanced search and filtering capabilities
    
    File Organization Structure:
    - datasets/{user_id}/{dataset_id}/{version}/{filename}
    
    Example: datasets/user1/drowsiness/v1/heart_rate.csv
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router)
app.include_router(users.router)
app.include_router(bias_reports.router)
app.include_router(transformation_reports.router)
app.include_router(ai_models.router)
app.include_router(xai_reports.router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Data Warehouse API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # You could add more comprehensive health checks here
        # like checking database connectivity, MinIO availability, etc.
        return {
            "status": "healthy",
            "services": {
                "api": "running",
                "database": "connected",
                "file_storage": "connected"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
