# API Changes: User ID Path Parameter & Auto-Versioning

## Overview

Two major improvements have been implemented based on partner feedback:

1. **User ID in Path Parameter**: All upload endpoints now include `user_id` as a path parameter instead of form data
2. **Auto-Versioning**: Datasets and models now automatically increment versions (v1, v2, v3, etc.)

---

## 🔄 Breaking Changes

### Before
```bash
# Old dataset upload
POST /datasets/upload
Form data: user_id, dataset_id, name, version, ...

# Old model upload
POST /ai-models/upload/single
Form data: user_id, model_id, name, version, ...

# Old XAI upload
POST /xai-reports/upload
Form data: user_id, dataset_id, model_id, ...
```

### After
```bash
# New dataset upload
POST /datasets/upload/{user_id}
Form data: dataset_id, name, ...  (no version!)

# New model upload
POST /ai-models/upload/single/{user_id}
Form data: model_id, name, ...  (no version!)

# New XAI upload
POST /xai-reports/upload/{user_id}
Form data: dataset_id, model_id, ...
```

---

## 📋 Detailed Changes

### 1. Dataset Upload Endpoint

**Endpoint**: `POST /datasets/upload/{user_id}`

**Changes**:
- ✅ `user_id` moved to path parameter
- ✅ `version` removed from form data (auto-incremented)
- ✅ First upload creates v1
- ✅ Subsequent uploads with same dataset_id create v2, v3, etc.

**Example**:
```bash
# First upload - creates v1
curl -X POST "http://localhost:8000/datasets/upload/user123" \
  -F "file=@data.csv" \
  -F "dataset_id=my-dataset" \
  -F "name=My Dataset"

# Response: version="v1"

# Second upload - creates v2
curl -X POST "http://localhost:8000/datasets/upload/user123" \
  -F "file=@data_updated.csv" \
  -F "dataset_id=my-dataset" \
  -F "name=My Dataset Updated"

# Response: version="v2"
```

---

### 2. AI Model Upload Endpoints

**Endpoints**: 
- `POST /ai-models/upload/single/{user_id}`
- `POST /ai-models/upload/folder/{user_id}`

**Changes**:
- ✅ `user_id` moved to path parameter
- ✅ `version` removed from form data (auto-incremented)
- ✅ First upload creates v1
- ✅ Subsequent uploads with same model_id create v2, v3, etc.

**Example**:
```bash
# First upload - creates v1
curl -X POST "http://localhost:8000/ai-models/upload/single/user123" \
  -F "file=@model.pkl" \
  -F "model_id=my-model" \
  -F "name=My Model" \
  -F "framework=sklearn" \
  -F "model_type=classification"

# Response: version="v1"

# Second upload - creates v2
curl -X POST "http://localhost:8000/ai-models/upload/single/user123" \
  -F "file=@model_v2.pkl" \
  -F "model_id=my-model" \
  -F "name=My Model v2" \
  -F "framework=sklearn" \
  -F "model_type=classification"

# Response: version="v2"
```

---

### 3. XAI Reports Upload Endpoint

**Endpoint**: `POST /xai-reports/upload/{user_id}`

**Changes**:
- ✅ `user_id` moved to path parameter
- ℹ️ XAI reports don't have versions (uniquely identified by user_id, dataset_id, model_id, report_type, level)

**Example**:
```bash
curl -X POST "http://localhost:8000/xai-reports/upload/user123" \
  -F "file=@report.html" \
  -F "dataset_id=my-dataset" \
  -F "model_id=my-model" \
  -F "report_type=model_explanation" \
  -F "level=beginner"
```

---

## 🔧 Implementation Details

### Auto-Versioning Logic

For both datasets and models, the system:
1. Queries all existing versions for the same `user_id` and `dataset_id`/`model_id`
2. Extracts version numbers (v1 → 1, v2 → 2, etc.)
3. Finds the maximum version number
4. Increments by 1
5. Returns `v{max + 1}`

**Example scenario**:
```
Existing versions: v1, v2, v5
Next version: v6  (not v3!)
```

**Code**:
```python
async def _get_next_dataset_version(user_id: str, dataset_id: str) -> str:
    # Find all versions
    versions = await db.datasets.find({
        "user_id": user_id,
        "dataset_id": dataset_id
    }, {"version": 1}).to_list(length=1000)
    
    if not versions:
        return "v1"
    
    # Extract numbers
    version_numbers = [int(v["version"][1:]) for v in versions if v["version"].startswith("v")]
    
    # Increment max
    return f"v{max(version_numbers) + 1}"
```

---

## 📝 Updated Consumer Scripts

All Kafka consumer scripts have been updated to use the new endpoints:

### AutoML Consumer
```python
# Before
url = f"{API_BASE}/ai-models/upload/single"
data = {'user_id': user_id, 'model_id': model_id, 'version': 'v1', ...}

# After
url = f"{API_BASE}/ai-models/upload/single/{user_id}"
data = {'model_id': model_id, ...}  # No version!
```

### XAI Consumer
```python
# Before
url = f"{API_BASE}/xai-reports/upload"
data = {'user_id': user_id, 'dataset_id': dataset_id, ...}

# After
url = f"{API_BASE}/xai-reports/upload/{user_id}"
data = {'dataset_id': dataset_id, ...}  # No user_id in form!
```

---

## ✅ Testing

### Test Auto-Versioning

```bash
# Upload same dataset multiple times
for i in {1..3}; do
  curl -X POST "http://localhost:8000/datasets/upload/testuser" \
    -F "file=@data.csv" \
    -F "dataset_id=test-dataset" \
    -F "name=Test Dataset $i"
  echo ""
done

# Check created versions
curl http://localhost:8000/datasets/testuser | jq '.[] | {dataset_id, version}'

# Expected output:
# {"dataset_id": "test-dataset", "version": "v1"}
# {"dataset_id": "test-dataset", "version": "v2"}
# {"dataset_id": "test-dataset", "version": "v3"}
```

### Test Complete Flow

```bash
# 1. Upload dataset (creates v1)
curl -X POST "http://localhost:8000/datasets/upload/flow_user" \
  -F "file=@heart_rate.csv" \
  -F "dataset_id=flow_test" \
  -F "name=Flow Test"

# 2. Watch the Kafka flow complete

# 3. Upload same dataset again (creates v2)
curl -X POST "http://localhost:8000/datasets/upload/flow_user" \
  -F "file=@heart_rate_v2.csv" \
  -F "dataset_id=flow_test" \
  -F "name=Flow Test v2"

# 4. Verify two versions exist
curl http://localhost:8000/datasets/flow_user | jq '.[] | select(.dataset_id=="flow_test") | {version, created_at}'
```

---

## 🚀 Benefits

### 1. Better REST API Design
- ✅ User ID in path follows RESTful conventions
- ✅ More intuitive URL structure
- ✅ Easier to implement route-based authorization

### 2. Simplified Client Code
- ✅ No need to track versions manually
- ✅ Just upload - system handles versioning
- ✅ Less room for user error

### 3. Consistent Versioning
- ✅ No version conflicts
- ✅ Automatic incrementing
- ✅ Works even if versions are deleted

### 4. Partner Integration
- ✅ Cleaner API for external services
- ✅ Matches industry standards
- ✅ Easier to document and explain

---

## 🔄 Migration Guide

### For API Consumers

**Old code**:
```python
import requests

response = requests.post(
    "http://localhost:8000/datasets/upload",
    files={'file': open('data.csv', 'rb')},
    data={
        'user_id': 'user123',
        'dataset_id': 'my-dataset',
        'name': 'My Dataset',
        'version': 'v1'
    }
)
```

**New code**:
```python
import requests

response = requests.post(
    "http://localhost:8000/datasets/upload/user123",  # user_id in URL
    files={'file': open('data.csv', 'rb')},
    data={
        'dataset_id': 'my-dataset',
        'name': 'My Dataset'
        # No version - auto-incremented!
    }
)
```

### For Internal Consumers

All internal Kafka consumers have been updated:
- ✅ `kafka_automl_consumer_example.py` - Updated
- ✅ `kafka_xai_consumer_example.py` - Updated

No changes needed if you're using the latest consumer scripts!

---

## 📊 Files Modified

### API Endpoints
- ✅ `app/api/datasets.py` - Path parameter + auto-versioning
- ✅ `app/api/ai_models.py` - Path parameter + auto-versioning (both endpoints)
- ✅ `app/api/xai_reports.py` - Path parameter

### Kafka Consumers
- ✅ `kafka_automl_consumer_example.py` - Updated upload function
- ✅ `kafka_xai_consumer_example.py` - Updated upload function

### Documentation
- ✅ `API_CHANGES_USER_ID_AND_VERSIONING.md` - This file

---

## ⚠️ Important Notes

1. **No duplicate version protection**: The system will create a new version even if the content is identical
2. **Version gaps allowed**: If v2 is deleted, the next upload creates v4 (not v3)
3. **Case sensitive**: user_id and dataset_id are case-sensitive
4. **No version downgrade**: Always increments - no way to create v1 if v2 exists

---

## 🎯 Summary

✅ All upload endpoints now use `{user_id}` path parameter  
✅ Automatic version incrementing for datasets and models  
✅ No manual version management needed  
✅ All consumers updated  
✅ Better REST API design  
✅ Ready for partner integration  

The changes are **backwards incompatible** but provide a much cleaner API for external integration! 🚀

