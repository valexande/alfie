@echo off
REM UC-2 Flask Application - Build and Run Script for Windows

echo 🐳 Building UC-2 Flask Application Docker Image...

REM Build the Docker image
docker build -t uc2-flask-app .

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully!
    echo.
    echo 🚀 Available commands:
    echo   build-and-run.bat run       - Run the Flask application
    echo   build-and-run.bat dev       - Run in development mode
    echo   build-and-run.bat compose   - Run with docker-compose
    echo   build-and-run.bat stop      - Stop running containers
    echo.
    
    REM Check if a specific command was provided
    if "%1"=="run" (
        echo 🚀 Running UC-2 Flask Application...
        docker run --rm -p 5000:5000 -v %cd%/csv-pkl-json:/app/csv-pkl-json:ro -v %cd%/output:/app/output uc2-flask-app
    ) else if "%1"=="dev" (
        echo 🔧 Running in development mode...
        docker-compose --profile dev up uc2-flask-dev
    ) else if "%1"=="compose" (
        echo 🐳 Running with docker-compose...
        docker-compose up uc2-flask-app
    ) else if "%1"=="stop" (
        echo 🛑 Stopping containers...
        docker-compose down
    ) else if "%1" neq "" (
        echo ❌ Unknown command: %1
        echo Available commands: run, dev, compose, stop
        exit /b 1
    ) else (
        echo 💡 Usage examples:
        echo   build-and-run.bat run     - Start the application
        echo   build-and-run.bat dev     - Start in development mode
        echo   build-and-run.bat compose - Use docker-compose
        echo.
        echo 🌐 Once running, access the API at:
        echo   http://localhost:5000/health
        echo   http://localhost:5000/explain-uc2-model
        echo   http://localhost:5000/explain-uc2-data
    )
) else (
    echo ❌ Docker build failed!
    exit /b 1
)
