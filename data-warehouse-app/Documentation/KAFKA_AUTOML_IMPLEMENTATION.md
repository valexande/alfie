# Kafka AutoML Implementation Summary

> **⚠️ NOTE**: This document is **OUTDATED**. Please refer to `KAFKA_ORCHESTRATION_COMPLETE.md` for the complete and correct implementation.
> 
> The main difference is that the AutoML consumer listens to `automl-trigger-events` (not `automl-events`), and the Agentic Core handles the XAI trigger orchestration.

---

# Kafka AutoML Implementation Summary (OUTDATED)

## Overview
This document summarizes the implementation of Kafka messaging for AI model uploads in the Data Warehouse application. Similar to how bias reports trigger Kafka events, AI model uploads now automatically send messages to the `kafka_automl_topic`.

## Changes Made

### 1. Fixed Bias Report Kafka Implementation (`app/services/kafka_service.py`)

**Issue Found**: The `bias_reports.py` was passing `target_column_name` and `task_type` to the Kafka service, but these parameters were not being accepted or included in the payload.

**Fix Applied**:
- Updated `send_bias_event()` method to accept `target_column_name` and `task_type` parameters
- Both fields are now properly included in the Kafka payload sent to the `bias-events` topic

```python
async def send_bias_event(
    self, 
    *, 
    dataset_id: str, 
    user_id: str, 
    bias_report_id: str | None, 
    has_transformation_report: bool,
    target_column_name: Optional[str] = None,  # ✅ Added
    task_type: Optional[str] = None             # ✅ Added
) -> None:
```

### 2. Added AutoML Kafka Event Handler (`app/services/kafka_service.py`)

Created a new method `send_automl_event()` that sends messages to the `kafka_automl_topic` when AI models are uploaded.

**Payload Structure**:
```json
{
  "event_type": "model.uploaded",
  "event_id": "model_uploaded_<model_id>_<timestamp>",
  "timestamp": "2025-10-10T12:00:00.000000",
  "user_id": "user123",
  "model_id": "my-model-id",
  "dataset_id": "dataset123",      // Optional - from training_dataset field
  "version": "v1",
  "framework": "pytorch",
  "model_type": "classification",
  "algorithm": "ResNet50",
  "model_size_mb": 98.5,
  "training_accuracy": 0.95,
  "validation_accuracy": 0.93,
  "test_accuracy": 0.92
}
```

### 3. Updated AI Model Upload Endpoints (`app/api/ai_models.py`)

**Changes**:
- Imported `kafka_producer_service` from the Kafka service
- Added `training_dataset` parameter to both upload endpoints (single file and folder)
- Added Kafka event sending after successful model creation in both endpoints
- Events are sent non-blocking (wrapped in try-except to prevent upload failures if Kafka is down)

**Endpoints Modified**:
1. `POST /ai-models/upload/single` - Single file upload
2. `POST /ai-models/upload/folder` - Multiple files upload (zip)

Both endpoints now:
- Accept an optional `training_dataset` form parameter (dataset_id)
- Send Kafka events to the `automl-events` topic after successful upload
- Include all relevant model metadata in the Kafka payload

### 4. Updated AutoML Consumer Example (`kafka_automl_consumer_example.py`)

Complete rewrite of the consumer example to:
- Listen to the `automl-events` topic
- Parse and process model upload events
- Fetch model metadata from the Data Warehouse API
- Optionally download model files
- Forward events to XAI topic for explainability analysis (when dataset_id is available)

## Configuration

The Kafka topics are configured in `app/core/config.py`:

```python
kafka_automl_topic: str = "automl-events"  # ✅ Already configured
```

## Testing

### 1. Start Kafka Services
Ensure Kafka is running (via docker-compose):
```bash
docker-compose up -d kafka zookeeper
```

### 2. Start the Data Warehouse API
```bash
python run.py
```

### 3. Start the AutoML Consumer (in a separate terminal)
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_AUTOML_TOPIC=automl-events \
KAFKA_CONSUMER_GROUP=automl-consumer \
python kafka_automl_consumer_example.py
```

### 4. Upload a Model

**Using curl**:
```bash
curl -X POST "http://localhost:8000/ai-models/upload/single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@model.pt" \
  -F "user_id=user123" \
  -F "model_id=test-model-1" \
  -F "name=Test Model" \
  -F "version=v1" \
  -F "framework=pytorch" \
  -F "model_type=classification" \
  -F "training_dataset=dataset123" \
  -F "training_accuracy=0.95"
```

**Using Python**:
```python
import requests

files = {'file': open('model.pt', 'rb')}
data = {
    'user_id': 'user123',
    'model_id': 'test-model-1',
    'name': 'Test Model',
    'version': 'v1',
    'framework': 'pytorch',
    'model_type': 'classification',
    'training_dataset': 'dataset123',  # Optional - links to dataset
    'training_accuracy': 0.95
}

response = requests.post(
    'http://localhost:8000/ai-models/upload/single',
    files=files,
    data=data
)
print(response.json())
```

### 5. Verify Kafka Message

Check the AutoML consumer logs - you should see:
```
AutoML Message received
  Partition=0 Offset=0
  Key=user123_test-model-1
  Event={
    "event_type": "model.uploaded",
    "event_id": "model_uploaded_test-model-1_1728561234",
    "timestamp": "2025-10-10T12:00:34.567890",
    "user_id": "user123",
    "model_id": "test-model-1",
    "dataset_id": "dataset123",
    "version": "v1",
    ...
  }
```

## Integration with XAI

The AutoML consumer is designed to forward events to the XAI (Explainable AI) system:

1. **Model Upload** → AutoML event sent to `automl-events` topic
2. **AutoML Consumer** → Processes event and forwards to `xai-events` topic
3. **XAI Consumer** → Receives event and generates explainability reports

**XAI Event Structure**:
```json
{
  "event_type": "xai.trigger",
  "event_id": "xai_trigger_<model_id>_<timestamp>",
  "timestamp": "2025-10-10T12:00:00.000000",
  "user_id": "user123",
  "model_id": "my-model-id",
  "dataset_id": "dataset123",
  "version": "v1",
  "framework": "pytorch",
  "model_type": "classification",
  "source_event": "model.uploaded"
}
```

## Key Features

✅ **Non-blocking**: Kafka events are sent asynchronously and failures don't affect model uploads  
✅ **Comprehensive Metadata**: Includes all relevant model information in the payload  
✅ **Dataset Linking**: Optional dataset_id links models to their training datasets  
✅ **XAI Integration**: Events can be forwarded to XAI system for explainability analysis  
✅ **Consistent with Bias Reports**: Follows the same pattern as the existing bias report implementation  

## Next Steps

1. Implement XAI consumer to process the forwarded events
2. Add support for model update events (if needed)
3. Add support for model deletion events (if needed)
4. Implement retry logic for failed Kafka sends
5. Add Kafka monitoring and alerting

## Files Modified

- ✅ `app/services/kafka_service.py` - Added `send_automl_event()`, fixed `send_bias_event()`
- ✅ `app/api/ai_models.py` - Added Kafka event sending to both upload endpoints
- ✅ `kafka_automl_consumer_example.py` - Complete consumer implementation
- ✅ `app/core/config.py` - Already had `kafka_automl_topic` configured

## Notes

- The `training_dataset` field is optional. If not provided, `dataset_id` in the Kafka payload will be `null`
- Make sure to include the `training_dataset` parameter when uploading models if you want XAI integration
- The Kafka message key is `{user_id}_{model_id}` for proper partitioning

