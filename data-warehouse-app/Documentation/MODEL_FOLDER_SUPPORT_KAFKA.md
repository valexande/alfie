# Model Folder Support in Kafka Messages

## Overview

All Kafka messages now track whether AI models are single files or folders (with multiple files), just like datasets. This allows consumers to download and handle models appropriately.

---

## 🎯 New Fields in Kafka Messages

### automl-events
```json
{
  "event_type": "model.uploaded",
  "user_id": "user123",
  "model_id": "model123",
  "dataset_id": "dataset123",
  "is_folder": false,           // Dataset is folder?
  "file_count": 1,              // Dataset file count
  "is_model_folder": true,      // ✨ NEW - Model is folder?
  "model_file_count": 2,        // ✨ NEW - Model file count
  "framework": "sklearn",
  "training_accuracy": 0.95,
  // ...
}
```

### xai-trigger-events
```json
{
  "event_type": "xai-trigger.reported",
  "user_id": "user123",
  "dataset_id": "dataset123",
  "model_id": "model123",
  "is_folder": false,           // Dataset is folder?
  "file_count": 1,              // Dataset file count
  "is_model_folder": true,      // ✨ NEW - Model is folder?
  "model_file_count": 2,        // ✨ NEW - Model file count
  "level": "beginner",
  // ...
}
```

---

## 📊 4 Possible Combinations

### 1. Single Dataset + Single Model (Most Common)
```json
{
  "is_folder": false,
  "file_count": 1,
  "is_model_folder": false,
  "model_file_count": 1
}
```

**Use Case**: Simple ML pipeline with one CSV and one model file

### 2. Single Dataset + Model Folder
```json
{
  "is_folder": false,
  "file_count": 1,
  "is_model_folder": true,
  "model_file_count": 2
}
```

**Use Case**: Model with dependencies (model.pkl + label_encoders.pkl)

### 3. Folder Dataset + Single Model
```json
{
  "is_folder": true,
  "file_count": 3,
  "is_model_folder": false,
  "model_file_count": 1
}
```

**Use Case**: Multiple data files (train.csv, test.csv, val.csv) with single model

### 4. Folder Dataset + Model Folder
```json
{
  "is_folder": true,
  "file_count": 3,
  "is_model_folder": true,
  "model_file_count": 2
}
```

**Use Case**: Complex setup with multiple data and model files

---

## 🔧 Implementation Changes

### AI Model Metadata
**File**: `app/models/ai_model.py`

Added fields:
```python
class AIModelMetadata(BaseModel):
    # ...
    is_model_folder: bool = Field(default=False)
    model_file_count: int = Field(default=1)
```

### AI Models API
**File**: `app/api/ai_models.py`

**Single file upload:**
```python
model_metadata = AIModelMetadata(
    # ...
    is_model_folder=False,
    model_file_count=1,
)
```

**Folder upload:**
```python
model_metadata = AIModelMetadata(
    # ...
    is_model_folder=True,
    model_file_count=len(model_files),
)
```

### Kafka Service
**File**: `app/services/kafka_service.py`

Updated `send_automl_event()`:
```python
async def send_automl_event(
    # ...
    is_model_folder: Optional[bool] = False,
    model_file_count: Optional[int] = 1
):
    payload = {
        # ...
        "is_model_folder": is_model_folder,
        "model_file_count": model_file_count,
    }
```

### Agentic Core
**File**: `kafka_agentic_core_consumer_example.py`

Extracts and forwards:
```python
is_model_folder = value.get("is_model_folder", False)
model_file_count = value.get("model_file_count", 1)

# Produce XAI trigger
payload = {
    # ...
    "is_model_folder": is_model_folder,
    "model_file_count": model_file_count,
}
```

### XAI Consumer
**File**: `kafka_xai_consumer_example.py`

Handles all 4 combinations:
```python
# Get from trigger event
is_folder = event.get("is_folder", False)
is_model_folder = event.get("is_model_folder", False)

# Download dataset
dataset_bytes = download_dataset_file(...)
if is_folder:
    dataset_files = extract_dataset_folder(dataset_bytes)

# Download model
model_bytes = download_model_file(...)
if is_model_folder:
    model_files = extract_model_folder(model_bytes)
```

---

## 💻 XAI Consumer - Complete Example

```python
async def process_xai_trigger(event: dict):
    # Extract folder info
    is_folder = event.get("is_folder", False)
    is_model_folder = event.get("is_model_folder", False)
    
    # Download dataset
    dataset_bytes = download_dataset_file(user_id, dataset_id)
    
    if is_folder:
        # Extract dataset folder
        dataset_files = extract_dataset_folder(dataset_bytes, "temp_dataset")
        csv_files = [f for f in dataset_files if f.endswith('.csv')]
        df = pd.read_csv(csv_files[0])
    else:
        # Load single dataset
        df = pd.read_csv(BytesIO(dataset_bytes))
    
    # Download model
    model_bytes = download_model_file(user_id, model_id, version)
    
    if is_model_folder:
        # Extract model folder
        model_files = extract_model_folder(model_bytes, "temp_model")
        
        # Load model and dependencies
        import pickle
        model_pkl = [f for f in model_files if f.endswith('model.pkl')][0]
        encoder_pkl = [f for f in model_files if 'encoder' in f.lower()]
        
        with open(model_pkl, 'rb') as f:
            model = pickle.load(f)
        
        if encoder_pkl:
            with open(encoder_pkl[0], 'rb') as f:
                encoders = pickle.load(f)
    else:
        # Load single model
        import pickle
        model = pickle.loads(model_bytes)
    
    # Generate XAI with all resources
    # shap_values = shap.TreeExplainer(model).shap_values(df)
    # ...
```

---

## 🧪 Testing Scenarios

### Scenario 1: Single + Single
```bash
# Upload single dataset
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@data.csv" \
  -F "dataset_id=test1" \
  -F "name=Test"
```
**XAI will receive**: `is_folder=false`, `is_model_folder=false`

### Scenario 2: Single + Folder (Your Use Case!)
```bash
# Upload single dataset
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@data.csv" \
  -F "dataset_id=test2" \
  -F "name=Test"

# AutoML V2 uploads model folder (model.pkl + label_encoders.pkl)
```
**XAI will receive**: `is_folder=false`, `is_model_folder=true`

### Scenario 3: Folder + Single
```bash
# Upload folder dataset
curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@dataset.zip" \
  -F "dataset_id=test3" \
  -F "name=Test"

# AutoML V1 uploads single model.pkl
```
**XAI will receive**: `is_folder=true`, `is_model_folder=false`

### Scenario 4: Folder + Folder
```bash
# Upload folder dataset
curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@dataset.zip" \
  -F "dataset_id=test4" \
  -F "name=Test"

# AutoML V2 uploads model folder
```
**XAI will receive**: `is_folder=true`, `is_model_folder=true`

---

## 📋 Expected Console Output

### XAI Consumer with Model Folder

```
XAI Trigger Message received
Processing XAI trigger for model automl_test_1728561234
  User: testuser
  Dataset: test-dataset
  Level: beginner
  Dataset type: SINGLE FILE (1 file(s))
  Model type: FOLDER (2 file(s))

Dataset metadata fetched: Test Dataset
Dataset downloaded: 2048 bytes
Dataset is single file - ready for XAI analysis

Model metadata fetched: AutoML Model (Multi-File)
  Framework: sklearn
  Files: 2
Model downloaded: 45678 bytes
Extracting model folder...
Extracted 2 model files:
  - temp_xai_model_automl_test/model.pkl
  - temp_xai_model_automl_test/label_encoders.pkl
NOTE: All model files are available for XAI analysis

================================================================================
Generating XAI explanations
================================================================================
XAI Resources Available:
  Dataset: Single file (2048 bytes)
  Model: 2 files extracted
```

---

## 📁 Files Modified

**Backend:**
- ✅ `app/models/ai_model.py` - Added is_model_folder, model_file_count
- ✅ `app/api/ai_models.py` - Set fields appropriately for single/folder
- ✅ `app/services/kafka_service.py` - Added fields to automl-events

**Consumers:**
- ✅ `kafka_agentic_core_consumer_example.py` - Pass model folder info
- ✅ `kafka_xai_consumer_example.py` - Handle both dataset and model folders

**Documentation:**
- ✅ `Documentation/MODEL_FOLDER_SUPPORT_KAFKA.md` - This guide

---

## ✅ Summary

**Problem**: XAI consumer couldn't detect if model was single file or folder

**Solution**:
- ✅ Added `is_model_folder` and `model_file_count` to AI model metadata
- ✅ Included in automl-events Kafka messages
- ✅ Passed through xai-trigger-events
- ✅ XAI consumer now downloads and extracts appropriately

**Result**: XAI consumer can now handle all 4 combinations of single/folder for datasets and models! 🎉

**Test with AutoML V2**: Upload dataset, let AutoML V2 upload model folder, watch XAI extract both! 🚀

