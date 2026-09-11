#!/usr/bin/env python3
"""
Entry point for the XAI Explainability API.

Run with:
    python run.py

Or with uvicorn directly:
    uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
"""

import uvicorn
from api.config import settings


def main():
    """Run the FastAPI application."""
    print("=" * 60)
    print("Starting XAI Explainability API")
    print("=" * 60)
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Debug: {settings.debug}")
    print("=" * 60)
    print("API Documentation available at:")
    print(f"  - Swagger UI: http://{settings.host}:{settings.port}/docs")
    print(f"  - ReDoc: http://{settings.host}:{settings.port}/redoc")
    print("=" * 60)
    
    uvicorn.run(
        "api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1
    )


if __name__ == "__main__":
    main()
