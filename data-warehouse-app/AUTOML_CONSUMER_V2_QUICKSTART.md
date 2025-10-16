# AutoML Consumer V2 - Quick Start Guide

## 🎯 What is V2?

A new version of the AutoML consumer that uploads **real model folders** with multiple files instead of dummy single files.

---

## 📊 Quick Comparison

| Feature | V1 | V2 |
|---------|----|----|
| **Model Source** | Dummy `model.pkl` | Real `xai-model/model-explanation-endpoint/` |
| **Files** | 1 file | 2+ files (model.pkl, label_encoders.pkl) |
| **Upload Endpoint** | `/upload/single/{user_id}` | `/upload/folder/{user_id}` |
| **Use Case** | Basic testing | Real-world testing |
| **Consumer Group** | `automl-consumer` | `automl-consumer-v2` |

---

## 🚀 Quick Start

### Option 1: Run V2 Only

```bash
# Stop V1 if running (Ctrl+C)

# Start V2
python kafka_automl_consumer_example_v2.py
```

### Option 2: Run Both V1 and V2

```bash
# Terminal 1: V1
KAFKA_CONSUMER_GROUP=automl-consumer-v1 \
python kafka_automl_consumer_example.py

# Terminal 2: V2
KAFKA_CONSUMER_GROUP=automl-consumer-v2 \
python kafka_automl_consumer_example_v2.py
```

**Result**: Both will receive AutoML triggers and upload different models!

---

## ✅ Prerequisites

### Required Files
- ✅ `xai-model/model-explanation-endpoint/` folder exists
- ✅ Contains `model.pkl`
- ✅ Contains `label_encoders.pkl`

### If Missing

```bash
# Check if folder exists
ls -la xai-model/model-explanation-endpoint/

# If not, you can use the existing ZIP
cd xai-model
unzip model-explanation-endpoint.zip
```

---

## 🧪 Complete Test Flow

### 1. Start All Services

```bash
# Terminal 1: DW API
python run.py

# Terminal 2: Agentic Core
python kafka_agentic_core_consumer_example.py

# Terminal 3: Bias Detector
python kafka_bias_detector_consumer_example.py

# Terminal 4: AutoML V2 ⭐
python kafka_automl_consumer_example_v2.py

# Terminal 5: XAI Consumer
python kafka_xai_consumer_example.py
```

### 2. Upload Dataset

```bash
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@heart_rate.csv" \
  -F "dataset_id=v2-flow-test" \
  -F "name=V2 Flow Test"
```

### 3. Watch V2 Consumer

You should see:
```
📦 Model Files to Upload:
  ✅ model.pkl (XXX bytes)
  ✅ label_encoders.pkl (XXX bytes)

AutoML Trigger Message received (V2)
Processing AutoML trigger...
Dataset type: SINGLE FILE
Dataset loaded: 27 rows, 4 columns

TRAINING MODEL (V2 - Real Model Folder)

Using real model from: xai-model/model-explanation-endpoint
  - model.pkl (XXX bytes)
  - label_encoders.pkl (XXX bytes)

Created ZIP: temp_model_automl_v2-flow-test_XXXXX.zip
Uploading model folder to DW: automl_v2-flow-test_XXXXX

✅ MODEL FOLDER UPLOADED TO DW!
   Model ID: automl_v2-flow-test_XXXXX
   Version: v1
   Files: 2
   Total size: 0.05 MB

AutoML event will be automatically sent by the DW
```

### 4. Verify in Data Warehouse

```bash
# List models
curl "http://localhost:8000/ai-models/search/testuser" | jq

# Get specific model
curl "http://localhost:8000/ai-models/testuser/automl_v2-flow-test_XXXXX" | jq

# List model files
curl "http://localhost:8000/ai-models/testuser/automl_v2-flow-test_XXXXX/files" | jq

# Download model folder
curl -o model.zip \
  "http://localhost:8000/ai-models/testuser/automl_v2-flow-test_XXXXX/download"
```

---

## 🎯 What This Tests

1. ✅ **Dataset Upload** - Single file or folder
2. ✅ **Bias Detection** - Works normally
3. ✅ **AutoML Trigger** - Received by V2 consumer
4. ✅ **Dataset Download** - Downloaded by V2
5. ✅ **Model Folder Creation** - Real model files zipped
6. ✅ **Model Folder Upload** - Uploaded via folder endpoint
7. ✅ **AutoML Event** - Produced with folder info
8. ✅ **XAI Trigger** - Follows with correct info
9. ✅ **XAI Generation** - Uses real model
10. ✅ **Complete Pipeline** - End-to-end with real model

---

## 💻 Code Structure

### Main Components

**1. create_model_zip()** - Creates ZIP from folder
```python
def create_model_zip(source_folder: str, output_zip: str) -> str:
    """Create ZIP from model folder"""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                # Add file to ZIP with relative path
                ...
```

**2. upload_model_folder_to_dw()** - Uploads via folder endpoint
```python
def upload_model_folder_to_dw(...):
    """Upload model folder (ZIP) to DW"""
    url = f"{API_BASE}/ai-models/upload/folder/{user_id}"
    files = {'zip_file': f}
    data = {'model_id': ..., 'framework': 'sklearn', ...}
    # POST to folder endpoint
```

**3. process_automl_trigger()** - Main processing logic
```python
async def process_automl_trigger(event: dict):
    # 1. Download dataset
    # 2. Load and analyze
    # 3. Create model ZIP
    # 4. Upload to DW
    # 5. Cleanup
```

---

## 🎉 Summary

**V2 Consumer:**
- ✅ Uses real model from your use case
- ✅ Uploads multiple files (folder)
- ✅ Tests folder upload endpoint
- ✅ Complete real-world scenario
- ✅ Auto-cleanup of temp files
- ✅ Can run alongside V1

**Test your actual model:**
```bash
python kafka_automl_consumer_example_v2.py
```

Upload a dataset and watch it upload your real model folder! 🚀

