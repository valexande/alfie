#!/bin/bash

# Script to start XAI services with Docker Compose

echo "🚀 Starting XAI Services..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

# Build and start services
echo "📦 Building Docker images..."
docker-compose -f docker-compose.xai.yml build

echo "🚀 Starting services..."
docker-compose -f docker-compose.xai.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.xai.yml ps

echo ""
echo "✅ Services started!"
echo ""
echo "Service URLs:"
echo "  - Universal Model Explainability: http://localhost:5010/health"
echo "  - Flexible Data Interpretability: http://localhost:5001/health"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.xai.yml logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f docker-compose.xai.yml down"


