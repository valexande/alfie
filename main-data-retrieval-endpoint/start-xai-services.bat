@echo off
REM Script to start XAI services with Docker Compose on Windows

echo Starting XAI Services...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Please start Docker first.
    exit /b 1
)

REM Build and start services
echo Building Docker images...
docker-compose -f docker-compose.xai.yml build

echo Starting services...
docker-compose -f docker-compose.xai.yml up -d

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo Service Status:
docker-compose -f docker-compose.xai.yml ps

echo.
echo Services started!
echo.
echo Service URLs:
echo   - Universal Model Explainability: http://localhost:5010/health
echo   - Flexible Data Interpretability: http://localhost:5001/health
echo.
echo To view logs:
echo   docker-compose -f docker-compose.xai.yml logs -f
echo.
echo To stop services:
echo   docker-compose -f docker-compose.xai.yml down


