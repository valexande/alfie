# Kafka Topics Quick Reference

## Topic Overview

| # | Topic Name | Producer | Consumer | Trigger |
|---|------------|----------|----------|---------|
| 1 | `dataset-events` | DW API | Agentic Core | Dataset uploaded |
| 2 | `bias-trigger-events` | Agentic Core | Bias Detector | User confirms bias check |
| 3 | `bias-events` | DW API | Agentic Core | Bias report saved |
| 4 | `automl-trigger-events` | Agentic Core | AutoML Consumer | User confirms training |
| 5 | `automl-events` | DW API | Agentic Core | Model uploaded |
| 6 | `xai-trigger-events` | Agentic Core | XAI Consumer | User requests XAI |
| 7 | `xai-events` | DW API | Agentic Core | XAI report saved |

---

## Message Flow

```
1. Dataset Upload
   └─> dataset-events [DW → Agentic Core]

2. Bias Detection Trigger
   └─> bias-trigger-events [Agentic Core → Bias Detector]
   └─> Bias Detector saves report to DW
   └─> bias-events [DW → Agentic Core]

3. AutoML Training Trigger
   └─> automl-trigger-events [Agentic Core → AutoML Consumer]
   └─> AutoML Consumer saves model to DW
   └─> automl-events [DW → Agentic Core]

4. XAI Generation Trigger
   └─> xai-trigger-events [Agentic Core → XAI Consumer]
   └─> XAI Consumer saves report to DW
   └─> xai-events [DW → Agentic Core]

5. User Notification
   └─> Agentic Core notifies user
```

---

## Consumer Commands

### Start Agentic Core (Orchestrator)
```bash
python kafka_agentic_core_consumer_example.py
```

### Start Bias Detector
```bash
python kafka_bias_detector_consumer_example.py
```

### Start AutoML Consumer
```bash
python kafka_automl_consumer_example.py
```

### Start XAI Consumer
```bash
python kafka_xai_consumer_example.py
```

---

## Payload Examples

### 1. bias-trigger-events
```json
{
  "event_type": "bias-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "target_column_name": "target",
  "task_type": "classification",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 2. bias-events
```json
{
  "event_type": "bias.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "bias_report_id": "report123",
  "target_column_name": "target",
  "task_type": "classification",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 3. automl-trigger-events
```json
{
  "event_type": "automl-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "target_column_name": "target",
  "task_type": "classification",
  "time_budget": "10",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 4. automl-events
```json
{
  "event_type": "model.uploaded",
  "user_id": "user123",
  "model_id": "model123",
  "dataset_id": "dataset123",
  "version": "v1",
  "framework": "sklearn",
  "model_type": "classification",
  "training_accuracy": 0.95,
  "timestamp": "2025-10-10T12:00:00"
}
```

### 5. xai-trigger-events
```json
{
  "event_type": "xai-trigger.reported",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "version": "v1",
  "level": "beginner",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 6. xai-events
```json
{
  "event_type": "xai.reported",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "report_type": "model_explanation",
  "level": "beginner",
  "xai_report_id": "report123",
  "timestamp": "2025-10-10T12:00:00"
}
```

---

## Testing Flow

### 1. Upload Dataset
```bash
curl -X POST http://localhost:8000/datasets/upload \
  -F "file=@data.csv" \
  -F "user_id=user123" \
  -F "dataset_id=test1" \
  -F "name=Test Dataset"
```

**Expected**: Agentic Core receives `dataset-events` and produces `bias-trigger-events`

### 2. Bias Report Saved
**Expected**: 
- Bias Detector receives `bias-trigger-events`
- Saves bias report to DW
- DW produces `bias-events`
- Agentic Core receives and produces `automl-trigger-events`

### 3. Model Saved
**Expected**:
- AutoML Consumer receives `automl-trigger-events`
- Trains and saves model to DW
- DW produces `automl-events`
- Agentic Core receives and produces `xai-trigger-events`

### 4. XAI Report Saved
**Expected**:
- XAI Consumer receives `xai-trigger-events`
- Generates and saves report to DW
- DW produces `xai-events`
- Agentic Core receives and notifies user

---

## Monitoring

### Check Kafka Topics
```bash
kafka-topics.sh --list --bootstrap-server localhost:9092
```

### Check Consumer Groups
```bash
kafka-consumer-groups.sh --describe --all-groups --bootstrap-server localhost:9092
```

### View Topic Messages
```bash
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic dataset-events --from-beginning
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Consumer not receiving | Check consumer group ID is unique |
| Messages piling up | Check consumer is running without errors |
| DW not producing | Check Kafka producer started in DW logs |
| Wrong topic | Verify topic names in config.py |

---

## File Locations

- **Config**: `app/core/config.py`
- **Kafka Service**: `app/services/kafka_service.py`
- **Orchestrator**: `kafka_agentic_core_consumer_example.py`
- **Consumers**: `kafka_*_consumer_example.py`
- **API Endpoints**: `app/api/*.py`

For detailed documentation, see: `KAFKA_ORCHESTRATION_COMPLETE.md`

