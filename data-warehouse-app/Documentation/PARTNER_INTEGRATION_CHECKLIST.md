# Partner Integration Checklist

## 📋 Pre-Integration Checklist

Before starting integration, ensure you have:

- [ ] **Data Warehouse running** - All services (Kafka, MongoDB, MinIO, DW API) are up
- [ ] **Network access** - Your service can reach DW services (same Docker network or host network)
- [ ] **API documentation** - Understand the endpoints and data formats
- [ ] **Kafka topics list** - Know which topics to consume/produce
- [ ] **Example consumer scripts** - Have the template scripts for reference
- [ ] **Environment variables** - Created `.env` file from `.env.example`

---

## 🔌 Connectivity Tests

### Test 1: DW API Connectivity

```bash
# From your container or host
curl http://dw-api:8000/health
# OR (if outside Docker)
curl http://localhost:8000/health

# Expected: {"status":"healthy", ...}
```

### Test 2: Kafka Connectivity

```bash
# From your container
nc -zv kafka 29092
# OR (if outside Docker)
nc -zv localhost 9092

# Expected: Connection succeeded
```

### Test 3: MongoDB Connectivity (Optional)

```bash
# Only if you need direct access (NOT RECOMMENDED)
nc -zv mongodb 27017
# OR
nc -zv localhost 27017
```

### Test 4: End-to-End Test

```bash
# Upload a test dataset
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@test.csv" \
  -F "dataset_id=test" \
  -F "name=Test"

# Verify your consumer receives the event
# Check your consumer logs for "dataset-events"
```

---

## 🏗️ Integration Steps

### Step 1: Set Up Docker Network

**Option A: Join Existing Network**
```bash
docker network connect data-warehouse-network <your-container>
```

**Option B: Use Host Networking**
```yaml
# docker-compose.yml
services:
  your-service:
    network_mode: "host"
```

**Option C: Shared Network**
```yaml
# docker-compose.yml
services:
  your-service:
    networks:
      - data-warehouse-network

networks:
  data-warehouse-network:
    external: true
```

### Step 2: Configure Environment Variables

Create `.env` file:
```bash
# For Docker
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
API_BASE=http://dw-api:8000

# For local development
# KAFKA_BOOTSTRAP_SERVERS=localhost:9092
# API_BASE=http://localhost:8000
```

### Step 3: Implement Consumer Logic

Use the provided templates:
- `kafka_agentic_core_consumer_example.py` - Orchestrator template
- `kafka_bias_detector_consumer_example.py` - Bias detection template
- `kafka_automl_consumer_example.py` - AutoML template
- `kafka_xai_consumer_example.py` - XAI template

### Step 4: Test Event Flow

1. Upload dataset to DW
2. Verify your consumer receives `dataset-events`
3. Process event and produce `bias-trigger-events`
4. Verify bias detector receives trigger
5. Continue through the pipeline

---

## ✅ Implementation Checklist

### Agentic Core Implementation

- [ ] Consumes `dataset-events`
- [ ] Interacts with user (get target column, task type)
- [ ] Produces `bias-trigger-events`
- [ ] Consumes `bias-events`
- [ ] Produces `automl-trigger-events`
- [ ] Consumes `automl-events`
- [ ] Produces `xai-trigger-events`
- [ ] Consumes `xai-events`
- [ ] Reports completion to user

### Bias Detector Implementation

- [ ] Consumes `bias-trigger-events`
- [ ] Downloads dataset from DW API
- [ ] Analyzes bias
- [ ] Posts bias report to DW API
- [ ] Verifies `bias-events` is produced

### AutoML Implementation

- [ ] Consumes `automl-trigger-events`
- [ ] Downloads dataset from DW API
- [ ] Trains model
- [ ] Uploads model to DW API
- [ ] Verifies `automl-events` is produced

### XAI Implementation

- [ ] Consumes `xai-trigger-events`
- [ ] Downloads dataset and model from DW API
- [ ] Generates explanations
- [ ] Uploads HTML reports to DW API
- [ ] Verifies `xai-events` is produced

---

## 🔒 Security Checklist

- [ ] **Use API for data access** - Don't directly access MongoDB/MinIO
- [ ] **Secure Kafka connections** - Use SASL/SSL in production
- [ ] **API authentication** - Implement if DW has API keys
- [ ] **Network isolation** - Only expose necessary ports
- [ ] **Error handling** - Handle connection failures gracefully
- [ ] **Logging** - Log events without sensitive data
- [ ] **Rate limiting** - Respect API rate limits

---

## 📊 Monitoring Checklist

- [ ] **Consumer lag** - Monitor Kafka consumer lag
- [ ] **Error rates** - Track processing errors
- [ ] **API latency** - Monitor DW API response times
- [ ] **Success rates** - Track end-to-end pipeline success
- [ ] **Resource usage** - Monitor CPU, memory, disk
- [ ] **Health checks** - Implement liveness/readiness probes

---

## 🐛 Troubleshooting Guide

### Issue: Can't connect to Kafka

**Check:**
```bash
# Is Kafka running?
docker ps | grep kafka

# Can you reach it?
nc -zv kafka 29092

# Check environment variable
echo $KAFKA_BOOTSTRAP_SERVERS
```

**Solutions:**
- Use `kafka:29092` inside Docker, not `localhost:9092`
- Ensure you're on the same Docker network
- Check firewall rules

### Issue: Can't connect to DW API

**Check:**
```bash
# Is DW API running?
docker ps | grep dw-api

# Can you reach it?
curl http://dw-api:8000/health

# Check environment variable
echo $API_BASE
```

**Solutions:**
- Use `http://dw-api:8000` inside Docker
- Ensure services are on same network
- Check API logs for errors

### Issue: Consumer not receiving messages

**Check:**
```bash
# Check consumer group
docker exec -it kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group <your-group-id>

# Check topic exists
docker exec -it kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --list
```

**Solutions:**
- Verify topic names match exactly
- Check consumer is subscribed to correct topic
- Check consumer group ID is unique
- Verify `auto_offset_reset` setting

### Issue: Uploaded files not visible

**Check:**
```bash
# Check file was uploaded
curl http://localhost:8000/datasets/<user_id>

# Check MinIO
# Login to MinIO console at http://localhost:9001
```

**Solutions:**
- Verify upload response was 200 OK
- Check DW API logs for errors
- Verify file format is supported

---

## 📝 Documentation to Review

Before integration:

- [ ] `DOCKER_DEPLOYMENT_GUIDE.md` - Networking and deployment
- [ ] `API_CHANGES_USER_ID_AND_VERSIONING.md` - API changes
- [ ] `KAFKA_ORCHESTRATION_COMPLETE.md` - Complete Kafka flow
- [ ] `KAFKA_QUICK_REFERENCE.md` - Topic and payload reference
- [ ] Consumer example scripts - Implementation templates

---

## 🚀 Go-Live Checklist

Before production deployment:

- [ ] All tests passing
- [ ] Error handling implemented
- [ ] Retry logic for transient failures
- [ ] Monitoring and alerting configured
- [ ] Documentation completed
- [ ] Performance testing done
- [ ] Security review completed
- [ ] Backup and recovery plan
- [ ] Rollback plan prepared
- [ ] Team trained

---

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f <service-name>`
2. Test connectivity using commands above
3. Review documentation
4. Check example consumer scripts
5. Contact DW team with:
   - Error messages
   - Service logs
   - Environment configuration
   - Steps to reproduce

---

## ✅ Sign-Off

**Partner Information:**
- Company: ___________________
- Service: ___________________
- Contact: ___________________
- Date: ___________________

**Integration Status:**
- [ ] Connectivity tested
- [ ] Consumer implemented
- [ ] Producer implemented
- [ ] End-to-end test passed
- [ ] Documentation reviewed
- [ ] Ready for production

**Signatures:**
- Partner Lead: ___________________
- DW Team Lead: ___________________

Ready to integrate! 🎉

