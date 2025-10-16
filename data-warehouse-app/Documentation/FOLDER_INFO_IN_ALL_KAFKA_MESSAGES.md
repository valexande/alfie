# Folder Information in All Kafka Messages

## Overview

All Kafka messages in the ML pipeline now include `is_folder` and `file_count` fields, ensuring that folder information flows through the entire pipeline from dataset upload to XAI report generation.

---

## 🔄 Complete Message Flow

```
dataset-events
  ├─ is_folder: true/false
  └─ file_count: X
       ↓
bias-trigger-events
  ├─ is_folder: true/false
  └─ file_count: X
       ↓
bias-events
  ├─ is_folder: true/false
  └─ file_count: X
       ↓
automl-trigger-events
  ├─ is_folder: true/false
  └─ file_count: X
       ↓
automl-events
  ├─ is_folder: true/false
  └─ file_count: X
       ↓
xai-trigger-events
  ├─ is_folder: true/false
  └─ file_count: X
       ↓
xai-events
  (final step)
```

---

## 📊 Updated Message Structures

### 1. dataset-events (Already Had It)
```json
{
  "event_type": "dataset.uploaded",
  "dataset": {
    "dataset_id": "dataset123",
    "user_id": "user123",
    "is_folder": true,
    "file_count": 3,
    // ...
  }
}
```

### 2. bias-trigger-events (✨ NEW)
```json
{
  "event_type": "bias-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "target_column_name": "target",
  "task_type": "classification",
  "is_folder": true,        // ✨ Added
  "file_count": 3,          // ✨ Added
  "timestamp": "2025-10-10T12:00:00"
}
```

### 3. bias-events (✨ NEW)
```json
{
  "event_type": "bias.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "bias_report_id": "report123",
  "target_column_name": "target",
  "task_type": "classification",
  "is_folder": true,        // ✨ Added
  "file_count": 3,          // ✨ Added
  "timestamp": "2025-10-10T12:00:00"
}
```

### 4. automl-trigger-events (✨ NEW)
```json
{
  "event_type": "automl-trigger.reported",
  "dataset_id": "dataset123",
  "user_id": "user123",
  "target_column_name": "target",
  "task_type": "classification",
  "is_folder": true,        // ✨ Added
  "file_count": 3,          // ✨ Added
  "time_budget": "10",
  "timestamp": "2025-10-10T12:00:00"
}
```

### 5. automl-events (✨ NEW)
```json
{
  "event_type": "model.uploaded",
  "user_id": "user123",
  "model_id": "model123",
  "dataset_id": "dataset123",
  "version": "v1",
  "framework": "sklearn",
  "model_type": "classification",
  "is_folder": true,        // ✨ Added
  "file_count": 3,          // ✨ Added
  "training_accuracy": 0.95,
  "timestamp": "2025-10-10T12:00:00"
}
```

### 6. xai-trigger-events (✨ NEW)
```json
{
  "event_type": "xai-trigger.reported",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "version": "v1",
  "is_folder": true,        // ✨ Added
  "file_count": 3,          // ✨ Added
  "level": "beginner",
  "timestamp": "2025-10-10T12:00:00"
}
```

---

## 🔧 Implementation Details

### Backend Changes

**1. Kafka Service** (`app/services/kafka_service.py`)
- ✅ `send_bias_event()` - Added `is_folder`, `file_count` parameters
- ✅ `send_automl_event()` - Added `is_folder`, `file_count` parameters

**2. Bias Reports API** (`app/api/bias_reports.py`)
- ✅ Passes `is_folder` and `file_count` to Kafka event

**3. AI Models API** (`app/api/ai_models.py`)
- ✅ Added `_get_dataset_folder_info()` helper function
- ✅ Fetches dataset metadata to get `is_folder` info
- ✅ Passes to Kafka event when model is uploaded

**4. Bias Report Model** (`app/models/bias_report.py`)
- ✅ Added `is_folder` and `file_count` fields to `BiasReportCreate`

### Consumer Changes

**1. Agentic Core** (`kafka_agentic_core_consumer_example.py`)
- ✅ Extracts `is_folder` from dataset-events
- ✅ Includes in bias-trigger-events
- ✅ Extracts `is_folder` from bias-events
- ✅ Includes in automl-trigger-events
- ✅ Extracts `is_folder` from automl-events
- ✅ Includes in xai-trigger-events

**2. Bias Detector** (`kafka_bias_detector_consumer_example.py`)
- ✅ Extracts `is_folder` from bias-trigger-events
- ✅ Passes to bias report API

**3. AutoML Consumer** (`kafka_automl_consumer_example.py`)
- ✅ Extracts `is_folder` from automl-trigger-events
- ✅ Uses for dataset download handling

**4. XAI Consumer** (`kafka_xai_consumer_example.py`)
- ✅ Extracts `is_folder` from xai-trigger-events
- ✅ Uses for dataset download handling

---

## 🎯 How It Works

### Flow of Folder Information

```
1. User uploads folder
   └─ DW detects: is_folder=true, file_count=3

2. dataset-events produced
   └─ Includes: is_folder=true, file_count=3

3. Agentic Core receives dataset-events
   └─ Extracts: is_folder, file_count
   └─ Produces bias-trigger-events with is_folder=true

4. Bias Detector receives bias-trigger-events
   └─ Extracts: is_folder, file_count
   └─ Downloads and extracts folder
   └─ Posts report with is_folder=true

5. DW produces bias-events
   └─ Includes: is_folder=true, file_count=3

6. Agentic Core receives bias-events
   └─ Extracts: is_folder, file_count
   └─ Produces automl-trigger-events with is_folder=true

7. AutoML receives automl-trigger-events
   └─ Extracts: is_folder, file_count
   └─ Downloads and extracts folder
   └─ Uploads model with dataset_id

8. DW produces automl-events
   └─ Fetches dataset metadata
   └─ Includes: is_folder=true, file_count=3

9. Agentic Core receives automl-events
   └─ Extracts: is_folder, file_count
   └─ Produces xai-trigger-events with is_folder=true

10. XAI receives xai-trigger-events
    └─ Extracts: is_folder, file_count
    └─ Downloads and extracts folder
    └─ Generates reports

11. Complete! ✓
```

---

## ✅ What This Fixes

**Before:**
- ✅ dataset-events had `is_folder`
- ❌ bias-trigger-events didn't have `is_folder`
- ❌ Bias Detector couldn't detect folder type
- ❌ bias-events didn't have `is_folder`
- ❌ automl-trigger-events didn't have `is_folder`
- ❌ AutoML Consumer couldn't detect folder type
- ❌ And so on...

**After:**
- ✅ ALL events have `is_folder` and `file_count`
- ✅ ALL consumers can detect folder type
- ✅ ALL consumers download appropriately
- ✅ ALL consumers extract ZIP files
- ✅ Information flows through entire pipeline

---

## 🧪 Testing

### Test Folder Flow

```bash
# 1. Create folder
zip -r test.zip file1.csv file2.csv

# 2. Upload
curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@test.zip" \
  -F "dataset_id=folder-flow" \
  -F "name=Folder Flow Test"

# 3. Watch ALL consumer logs
```

**Expected in ALL consumers:**
```
Dataset type: FOLDER
File count: 2
Extracting folder dataset...
Extracted 2 files
```

---

## 📋 Files Modified

### Backend API
- ✅ `app/services/kafka_service.py` - Updated send_bias_event, send_automl_event
- ✅ `app/api/bias_reports.py` - Pass is_folder to Kafka
- ✅ `app/api/ai_models.py` - Fetch dataset info, pass to Kafka
- ✅ `app/models/bias_report.py` - Added is_folder, file_count fields

### Kafka Consumers
- ✅ `kafka_agentic_core_consumer_example.py` - Pass is_folder through all stages
- ✅ `kafka_bias_detector_consumer_example.py` - Pass is_folder to API

---

## ✅ Summary

**Problem Solved:**
- Folder information now flows through entire pipeline
- All consumers can detect and handle folders
- No more missing `is_folder` information

**Changes Made:**
- ✅ All Kafka messages include `is_folder` and `file_count`
- ✅ Agentic Core passes info through all trigger events
- ✅ Bias Detector passes info to bias report
- ✅ AI Models API fetches dataset info for automl events
- ✅ All consumers log folder type

**Result:**
Every consumer in the pipeline now knows if they're working with a single file or folder! 🎉

