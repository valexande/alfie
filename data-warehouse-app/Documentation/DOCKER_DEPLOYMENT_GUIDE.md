# Docker Deployment Guide for Third-Party Integration

## Overview

This guide explains how external services (like the Agentic Core) can communicate with the Data Warehouse when running in separate Docker containers.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Network: data-warehouse-network         │
│                                                  │
│  ┌──────────────┐    ┌──────────────┐          │
│  │   Kafka      │    │   MongoDB    │          │
│  │   :9092      │    │   :27017     │          │
│  └──────────────┘    └──────────────┘          │
│                                                  │
│  ┌──────────────┐    ┌──────────────┐          │
│  │   MinIO      │    │   DW API     │          │
│  │   :9000      │    │   :8000      │          │
│  └──────────────┘    └──────────────┘          │
│                                                  │
└─────────────────────────────────────────────────┘
           ↑                    ↑
           │                    │
           │ Kafka Events       │ HTTP API
           │                    │
┌──────────┴────────────────────┴─────────┐
│  External Container (Partner Service)    │
│  - Agentic Core                          │
│  - Bias Detector                         │
│  - AutoML Service                        │
│  - XAI Service                           │
└──────────────────────────────────────────┘
```

---

## 🚨 Key Networking Issues

### Issue 1: `localhost` Won't Work in Docker

**Why your tests work outside Docker:**
```python
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"  # ✅ Works outside Docker
API_BASE = "http://localhost:8000"          # ✅ Works outside Docker
```

**Why this FAILS inside Docker:**
- `localhost` in a container refers to that container itself
- Services in other containers are NOT reachable via `localhost`

**Solution:**
Use Docker service names or host networking.

---

## ✅ Solution 1: Docker Compose Network (Recommended)

### Update `docker-compose.yml`

Add a custom network and expose services:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - data-warehouse-network

  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "29092:29092"  # Internal Docker port
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    networks:
      - data-warehouse-network

  mongodb:
    image: mongo:latest
    container_name: mongodb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
      MONGO_INITDB_DATABASE: data_warehouse
    volumes:
      - ./init-mongo.js:/docker-entrypoint-initdb.d/init-mongo.js:ro
      - mongodb_data:/data/db
    networks:
      - data-warehouse-network

  minio:
    image: minio/minio:latest
    container_name: minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    networks:
      - data-warehouse-network

  dw-api:
    build: .
    container_name: dw-api
    ports:
      - "8000:8000"
    environment:
      MONGODB_URL: mongodb://mongodb:27017
      MINIO_ENDPOINT: minio:9000
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
    depends_on:
      - kafka
      - mongodb
      - minio
    networks:
      - data-warehouse-network

networks:
  data-warehouse-network:
    driver: bridge

volumes:
  mongodb_data:
  minio_data:
```

### Partner's Docker Compose (Agentic Core)

```yaml
version: '3.8'

services:
  agentic-core:
    image: partner/agentic-core:latest
    container_name: agentic-core
    environment:
      # Use service names, not localhost!
      KAFKA_BOOTSTRAP_SERVERS: kafka:29092
      API_BASE: http://dw-api:8000
      MONGODB_URL: mongodb://mongodb:27017  # If direct access needed (not recommended)
    networks:
      - data-warehouse-network  # Join the DW network

networks:
  data-warehouse-network:
    external: true  # Use existing network
```

---

## ✅ Solution 2: Host Networking (Simple but Less Isolated)

If partner service runs with `--network="host"`:

```yaml
services:
  agentic-core:
    image: partner/agentic-core:latest
    network_mode: "host"  # Use host network
    environment:
      # Now localhost works!
      KAFKA_BOOTSTRAP_SERVERS: localhost:9092
      API_BASE: http://localhost:8000
```

**Pros:**
- Simple - no network configuration needed
- `localhost` works as expected

**Cons:**
- Less isolation
- Port conflicts possible
- Not recommended for production

---

## 🔐 Best Practices for External Services

### 1. Communication via API Only (Recommended)

**✅ Good:**
```python
# External service communicates via DW API
import requests

# Get dataset
response = requests.get(f"{API_BASE}/datasets/{user_id}/{dataset_id}")
dataset_metadata = response.json()

# Download file
response = requests.get(f"{API_BASE}/datasets/{user_id}/{dataset_id}/download")
dataset_bytes = response.content
```

**❌ Bad:**
```python
# External service directly accessing MongoDB
from pymongo import MongoClient
client = MongoClient("mongodb://mongodb:27017")
db = client.data_warehouse
# Don't do this!
```

### 2. Kafka Access

External services MUST access Kafka for event-driven architecture:

```python
# This is OK - Kafka is designed for this
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

consumer = AIOKafkaConsumer(
    "dataset-events",
    bootstrap_servers="kafka:29092",  # Use Docker service name
    # ...
)
```

### 3. Environment Variables

Create a `.env` file for the partner service:

```bash
# .env for external service
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
KAFKA_DATASET_TOPIC=dataset-events
KAFKA_BIAS_TOPIC=bias-events
# ... other topics

API_BASE=http://dw-api:8000

# NOT recommended - use API instead
# MONGODB_URL=mongodb://mongodb:27017
# MINIO_ENDPOINT=minio:9000
```

---

## 📝 Configuration Guide for Partners

### For Running Outside Docker (Development)

```bash
# .env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
API_BASE=http://localhost:8000
```

### For Running Inside Docker (Production)

```bash
# .env
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
API_BASE=http://dw-api:8000
```

### Making It Flexible

```python
import os

# Auto-detect environment
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"  # Default for development
)

API_BASE = os.getenv(
    "API_BASE",
    "http://localhost:8000"  # Default for development
)
```

---

## 🧪 Testing Connectivity

### From Partner Container

```bash
# Enter partner container
docker exec -it agentic-core bash

# Test Kafka connectivity
nc -zv kafka 29092

# Test DW API connectivity
curl http://dw-api:8000/health

# Test MongoDB connectivity (if direct access)
nc -zv mongodb 27017

# Test MinIO connectivity (if direct access)
nc -zv minio 9000
```

### Health Check Endpoints

Add to DW API for partners to test:

```python
# app/main.py

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "kafka": await check_kafka(),
            "mongodb": await check_mongodb(),
            "minio": await check_minio()
        }
    }

@app.get("/health/kafka")
async def kafka_health():
    """Check if Kafka is reachable"""
    try:
        # Attempt to connect
        from aiokafka import AIOKafkaProducer
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers
        )
        await producer.start()
        await producer.stop()
        return {"status": "healthy", "kafka": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "kafka": str(e)}
```

---

## 🔒 Security Considerations

### 1. Expose Only What's Needed

**Current setup exposes:**
- ✅ Kafka: 9092 (needed for events)
- ✅ DW API: 8000 (needed for data access)
- ⚠️ MongoDB: 27017 (should NOT be exposed)
- ⚠️ MinIO: 9000 (should NOT be exposed)

**Recommended:**
```yaml
# docker-compose.yml

mongodb:
  # Remove ports - only accessible internally
  # ports:
  #   - "27017:27017"  # Don't expose!
  networks:
    - data-warehouse-network

minio:
  # Remove ports - only accessible internally
  # ports:
  #   - "9000:9000"  # Don't expose!
  networks:
    - data-warehouse-network
```

### 2. API Authentication

Add authentication to the DW API:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@router.post("/upload/{user_id}", dependencies=[Depends(verify_api_key)])
async def upload_dataset(...):
    # Endpoint protected by API key
    ...
```

### 3. Kafka Security (Production)

For production, enable Kafka SASL/SSL:

```yaml
kafka:
  environment:
    KAFKA_SECURITY_PROTOCOL: SASL_SSL
    KAFKA_SASL_MECHANISM: PLAIN
    KAFKA_SSL_ENDPOINT_IDENTIFICATION_ALGORITHM: ""
```

---

## 📋 Checklist for Partners

Before integrating, partners should:

- [ ] Understand Docker networking (service names vs localhost)
- [ ] Configure environment variables correctly
- [ ] Test connectivity to Kafka
- [ ] Test connectivity to DW API
- [ ] NOT directly access MongoDB/MinIO
- [ ] Use provided Kafka consumer examples as templates
- [ ] Test health check endpoints
- [ ] Handle connection errors gracefully
- [ ] Use retry logic for transient failures

---

## 🚀 Quick Start for Partners

### 1. Join the Network

```bash
# If DW network already exists
docker network connect data-warehouse-network <partner-container>
```

### 2. Update Environment

```bash
export KAFKA_BOOTSTRAP_SERVERS=kafka:29092
export API_BASE=http://dw-api:8000
```

### 3. Test Connection

```python
# test_connection.py
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Test API
response = requests.get(f"{API_BASE}/health")
print(f"API Health: {response.json()}")

# Test Kafka (from consumer script)
python kafka_agentic_core_consumer_example.py
```

---

## 💡 Summary

**Your Current Setup (Working):**
- ✅ Services in Docker
- ✅ Consumers running on host machine
- ✅ Using `localhost` to connect
- ✅ Works because ports are exposed

**Partner Setup (When in Docker):**
- ✅ Services in Docker
- ✅ Partner service in Docker
- ❌ Can't use `localhost`
- ✅ Must use Docker service names OR host networking
- ✅ Must join the same Docker network

**Recommended for Partners:**
1. Use Docker Compose networks
2. Configure service names (kafka:29092, not localhost:9092)
3. Access ONLY via API and Kafka
4. Use environment variables
5. Test connectivity before full integration

**Files to Share with Partners:**
- ✅ `kafka_*_consumer_example.py` scripts (as templates)
- ✅ `docker-compose.yml` (network configuration)
- ✅ `.env.example` (environment variables template)
- ✅ API documentation
- ✅ This deployment guide!

Ready to share! 🎉

