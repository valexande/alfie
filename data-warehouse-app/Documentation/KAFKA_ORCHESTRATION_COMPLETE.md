# Complete Kafka Orchestration Flow

## Overview
This document describes the complete Kafka-based ML pipeline orchestration in the Data Warehouse application. The Agentic Core orchestrates the entire flow from dataset upload to XAI report generation.

---

## Architecture Diagram

```
┌──────────────┐
│   Dataset    │
│   Upload     │
└──────┬───────┘
       │ dataset-events
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENTIC CORE                              │
│  (Orchestrator - kafka_agentic_core_consumer_example.py)    │
└──────────────────────────────────────────────────────────────┘
       │
       │ 1. Listens to: dataset-events
       │    Produces: bias-trigger-events
       │
       ▼
┌──────────────────────┐
│  Bias Detector       │  ← Listens to: bias-trigger-events
│  Consumer            │
└──────┬───────────────┘
       │ Saves bias report to DW
       │ DW produces: bias-events
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENTIC CORE                              │
└──────────────────────────────────────────────────────────────┘
       │
       │ 2. Listens to: bias-events
       │    Produces: automl-trigger-events
       │
       ▼
┌──────────────────────┐
│  AutoML              │  ← Listens to: automl-trigger-events
│  Consumer            │
└──────┬───────────────┘
       │ Trains & saves model to DW
       │ DW produces: automl-events
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENTIC CORE                              │
└──────────────────────────────────────────────────────────────┘
       │
       │ 3. Listens to: automl-events
       │    Produces: xai-trigger-events
       │
       ▼
┌──────────────────────┐
│  XAI                 │  ← Listens to: xai-trigger-events
│  Consumer            │
└──────┬───────────────┘
       │ Generates & saves XAI report to DW
       │ DW produces: xai-events
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENTIC CORE                              │
│                  (Reports to user)                           │
└──────────────────────────────────────────────────────────────┘
```

---

## Complete Flow

### Step 1: Dataset Upload → Bias Detection

**Trigger**: User uploads dataset via API

**Flow**:
1. User uploads dataset → DW saves it → DW produces `dataset-events` message
2. **Agentic Core** listens to `dataset-events`
3. Agentic Core interacts with user to get:
   - Target column name
   - Task type (classification, regression, etc.)
4. Agentic Core produces `bias-trigger-events` message
5. **Bias Detector Consumer** (`kafka_bias_detector_consumer_example.py`) listens to `bias-trigger-events`
6. Bias Detector:
   - Downloads dataset
   - Analyzes bias
   - Saves bias report to DW
7. DW automatically produces `bias-events` message

**Topics**:
- Input: `dataset-events`
- Output: `bias-trigger-events` → `bias-events`

**Payload Structures**:

`bias-trigger-events`:
```json
{
  "event_type": "bias-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "target_column_name": "target",
  "task_type": "classification",
  "timestamp": "2025-10-10T12:00:00.000000",
  "metadata": {...},
  "record_count": 1000
}
```

`bias-events`:
```json
{
  "event_type": "bias.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "bias_report_id": "report_id",
  "has_transformation_report": false,
  "target_column_name": "target",
  "task_type": "classification",
  "timestamp": "2025-10-10T12:00:00.000000"
}
```

---

### Step 2: Bias Detection → AutoML Training

**Trigger**: Bias report saved to DW (produces `bias-events`)

**Flow**:
1. **Agentic Core** listens to `bias-events`
2. Agentic Core reports bias findings to user
3. Agentic Core produces `automl-trigger-events` message
4. **AutoML Consumer** (`kafka_automl_consumer_example.py`) listens to `automl-trigger-events`
5. AutoML Consumer:
   - Downloads dataset
   - Identifies ML problem
   - Trains model using AutoML
   - Saves model to DW
6. DW automatically produces `automl-events` message

**Topics**:
- Input: `bias-events`
- Output: `automl-trigger-events` → `automl-events`

**Payload Structures**:

`automl-trigger-events`:
```json
{
  "event_type": "automl-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "target_column_name": "target",
  "task_type": "classification",
  "time_budget": "10",
  "timestamp": "2025-10-10T12:00:00.000000"
}
```

`automl-events`:
```json
{
  "event_type": "model.uploaded",
  "event_id": "model_uploaded_model123_1728561234",
  "timestamp": "2025-10-10T12:00:00.000000",
  "user_id": "user123",
  "model_id": "model123",
  "dataset_id": "dataset123",
  "version": "v1",
  "framework": "sklearn",
  "model_type": "classification",
  "algorithm": "RandomForest",
  "model_size_mb": 25.5,
  "training_accuracy": 0.95,
  "validation_accuracy": 0.93,
  "test_accuracy": 0.92
}
```

---

### Step 3: AutoML Training → XAI Generation

**Trigger**: Model saved to DW (produces `automl-events`)

**Flow**:
1. **Agentic Core** listens to `automl-events`
2. Agentic Core reports model training results to user
3. Agentic Core asks user if they want XAI explanations
4. Agentic Core produces `xai-trigger-events` message
5. **XAI Consumer** (`kafka_xai_consumer_example.py`) listens to `xai-trigger-events`
6. XAI Consumer:
   - Downloads dataset and model
   - Generates XAI explanations (SHAP, LIME, etc.)
   - Creates HTML reports for different expertise levels
   - Saves reports to DW
7. DW automatically produces `xai-events` message

**Topics**:
- Input: `automl-events`
- Output: `xai-trigger-events` → `xai-events`

**Payload Structures**:

`xai-trigger-events`:
```json
{
  "event_type": "xai-trigger.reported",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "version": "v1",
  "level": "beginner",
  "timestamp": "2025-10-10T12:00:00.000000"
}
```

`xai-events`:
```json
{
  "event_type": "xai.reported",
  "event_id": "xai_reported_model123_1728561234",
  "timestamp": "2025-10-10T12:00:00.000000",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "report_type": "model_explanation",
  "level": "beginner",
  "xai_report_id": "report_id"
}
```

---

### Step 4: XAI Generation → User Notification

**Trigger**: XAI report saved to DW (produces `xai-events`)

**Flow**:
1. **Agentic Core** listens to `xai-events`
2. Agentic Core notifies user that XAI report is ready
3. User can view report at: `/xai-reports/{user_id}/{dataset_id}/{model_id}/{report_type}/{level}/view`

**Topics**:
- Input: `xai-events`
- Output: User notification

---

## Kafka Topics Summary

| Topic | Producer | Consumer | Purpose |
|-------|----------|----------|---------|
| `dataset-events` | Data Warehouse API | Agentic Core | Dataset uploaded |
| `bias-trigger-events` | Agentic Core | Bias Detector | Trigger bias detection |
| `bias-events` | Data Warehouse API | Agentic Core | Bias report saved |
| `automl-trigger-events` | Agentic Core | AutoML Consumer | Trigger model training |
| `automl-events` | Data Warehouse API | Agentic Core | Model uploaded |
| `xai-trigger-events` | Agentic Core | XAI Consumer | Trigger XAI generation |
| `xai-events` | Data Warehouse API | Agentic Core | XAI report saved |

---

## Configuration

All topics are configured in `app/core/config.py`:

```python
# Kafka Configuration
kafka_bootstrap_servers: str = "localhost:9092"
kafka_dataset_topic: str = "dataset-events"
kafka_bias_topic: str = "bias-events"
kafka_automl_topic: str = "automl-events"
kafka_xai_topic: str = "xai-events"
kafka_bias_trigger_topic: str = "bias-trigger-events"
kafka_automl_trigger_topic: str = "automl-trigger-events"
kafka_xai_trigger_topic: str = "xai-trigger-events"
```

---

## Files Modified/Created

### Data Warehouse API (Producers)

1. **`app/api/bias_reports.py`** ✅
   - Sends `bias-events` when bias report is saved
   - Includes `target_column_name` and `task_type` in payload

2. **`app/api/ai_models.py`** ✅
   - Sends `automl-events` when model is uploaded
   - Includes all model metadata in payload

3. **`app/api/xai_reports.py`** ✅
   - Sends `xai-events` when XAI report is uploaded
   - Includes report type and expertise level

4. **`app/services/kafka_service.py`** ✅
   - Added `send_bias_event()` (fixed to accept target_column_name, task_type)
   - Added `send_automl_event()`
   - Added `send_xai_event()`

### Consumers

5. **`kafka_agentic_core_consumer_example.py`** ✅
   - Orchestrates entire pipeline
   - Listens to: `dataset-events`, `bias-events`, `automl-events`, `xai-events`
   - Produces: `bias-trigger-events`, `automl-trigger-events`, `xai-trigger-events`

6. **`kafka_bias_detector_consumer_example.py`** ✅
   - Listens to: `bias-trigger-events`
   - Generates and saves bias reports to DW

7. **`kafka_automl_consumer_example.py`** ✅
   - Listens to: `automl-trigger-events`
   - Trains models and saves to DW

8. **`kafka_xai_consumer_example.py`** ✅
   - Listens to: `xai-trigger-events`
   - Generates XAI reports and saves to DW

---

## Testing the Complete Flow

### 1. Start Kafka Services

```bash
docker-compose up -d kafka zookeeper
```

### 2. Start Data Warehouse API

```bash
python run.py
```

### 3. Start All Consumers (in separate terminals)

**Terminal 1: Agentic Core**
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
python kafka_agentic_core_consumer_example.py
```

**Terminal 2: Bias Detector**
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_BIAS_TRIGGER_TOPIC=bias-trigger-events \
KAFKA_BIAS_TRIGGER_CONSUMER_GROUP=bias-trigger-consumer \
python kafka_bias_detector_consumer_example.py
```

**Terminal 3: AutoML Consumer**
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_AUTOML_TRIGGER_TOPIC=automl-trigger-events \
KAFKA_CONSUMER_GROUP=automl-consumer \
python kafka_automl_consumer_example.py
```

**Terminal 4: XAI Consumer**
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_XAI_TRIGGER_TOPIC=xai-trigger-events \
KAFKA_CONSUMER_GROUP=xai-consumer \
python kafka_xai_consumer_example.py
```

### 4. Upload a Dataset

```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@dataset.csv" \
  -F "user_id=user123" \
  -F "dataset_id=test-dataset-1" \
  -F "name=Test Dataset"
```

### 5. Watch the Flow

You should see messages flowing through all terminals:

1. **Agentic Core** receives dataset-events → produces bias-trigger-events
2. **Bias Detector** receives bias-trigger-events → saves report → DW produces bias-events
3. **Agentic Core** receives bias-events → produces automl-trigger-events
4. **AutoML Consumer** receives automl-trigger-events → saves model → DW produces automl-events
5. **Agentic Core** receives automl-events → produces xai-trigger-events
6. **XAI Consumer** receives xai-trigger-events → saves report → DW produces xai-events
7. **Agentic Core** receives xai-events → notifies user

---

## Implementation Status

✅ **Completed**:
- All Kafka event producers in DW API
- Complete orchestration in Agentic Core
- All consumer scripts with proper topic subscriptions
- Comprehensive documentation

🔧 **TODO** (Implementation Details):
- Actual bias detection logic in `kafka_bias_detector_consumer_example.py`
- Actual AutoML training logic in `kafka_automl_consumer_example.py`
- Actual XAI generation logic in `kafka_xai_consumer_example.py`
- User interaction mechanism in Agentic Core
- Error handling and retry logic
- Monitoring and alerting

---

## Key Design Decisions

1. **Agentic Core as Orchestrator**: Single service coordinates the entire pipeline
2. **Trigger Topics**: Separate trigger topics for each stage allow for user interaction
3. **Event Topics**: Separate event topics from DW notify completion of each stage
4. **Non-blocking Kafka**: All Kafka sends are wrapped in try-except to prevent API failures
5. **Consistent Payload Structure**: All events follow similar structure for easy processing

---

## Next Steps

1. Implement actual ML logic in each consumer
2. Add user interaction UI/API for Agentic Core
3. Add monitoring dashboard for pipeline status
4. Implement error handling and retry mechanisms
5. Add pipeline configuration (time budgets, model types, etc.)
6. Add support for pipeline cancellation
7. Add pipeline history and audit logs

---

## Troubleshooting

### Consumer not receiving messages?
- Check Kafka is running: `docker ps | grep kafka`
- Check topic exists: `kafka-topics.sh --list --bootstrap-server localhost:9092`
- Check consumer group offset: `kafka-consumer-groups.sh --describe --group <group-name> --bootstrap-server localhost:9092`

### Messages stuck in queue?
- Check consumer is running and not throwing errors
- Check consumer group ID is unique per consumer
- Check auto_offset_reset setting

### DW not producing events?
- Check Kafka producer is started in DW: Look for "Kafka producer started" in logs
- Check for Kafka errors in DW logs
- Verify topic names match config

---

## Contact & Support

For questions or issues with the Kafka orchestration, please check:
1. Application logs in each consumer
2. Kafka broker logs
3. Data Warehouse API logs

Happy orchestrating! 🚀

