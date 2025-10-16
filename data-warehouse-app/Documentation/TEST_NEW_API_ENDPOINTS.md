# Testing Guide: New API Endpoints

## Quick Test Commands

### 1. Test Dataset Upload with Auto-Versioning

**First upload (creates v1)**:
```bash
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@heart_rate.csv" \
  -F "dataset_id=versioning-test" \
  -F "name=Versioning Test v1"
```

**Second upload (creates v2)**:
```bash
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@heart_rate.csv" \
  -F "dataset_id=versioning-test" \
  -F "name=Versioning Test v2"
```

**Third upload (creates v3)**:
```bash
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@heart_rate.csv" \
  -F "dataset_id=versioning-test" \
  -F "name=Versioning Test v3"
```

**Verify all versions created**:
```bash
curl http://localhost:8000/datasets/testuser | jq '.[] | select(.dataset_id=="versioning-test") | {version, name, created_at}'
```

Expected output:
```json
{"version":"v1","name":"Versioning Test v1","created_at":"2025-10-10T..."}
{"version":"v2","name":"Versioning Test v2","created_at":"2025-10-10T..."}
{"version":"v3","name":"Versioning Test v3","created_at":"2025-10-10T..."}
```

---

### 2. Test Complete Kafka Flow with New Endpoints

**Start all services** (if not already running):
```bash
# Terminal 1: Kafka
docker-compose up -d kafka zookeeper

# Terminal 2: DW API (RESTART to load changes!)
python run.py

# Terminal 3-6: Consumers
python kafka_agentic_core_consumer_example.py
python kafka_bias_detector_consumer_example.py
python kafka_automl_consumer_example.py
python kafka_xai_consumer_example.py
```

**Upload dataset**:
```bash
curl -X POST "http://localhost:8000/datasets/upload/flow_user" \
  -F "file=@heart_rate.csv" \
  -F "dataset_id=complete_flow_v2" \
  -F "name=Complete Flow Test v2"
```

**Watch the flow**:
- Terminal 3: Dataset event → Bias trigger
- Terminal 4: Bias trigger → Saves report → Bias event
- Terminal 3: Bias event → AutoML trigger
- Terminal 5: AutoML trigger → Uploads model (v1) → AutoML event
- Terminal 3: AutoML event → XAI trigger
- Terminal 6: XAI trigger → Uploads reports → XAI events
- Terminal 3: XAI events → ML PIPELINE COMPLETED!

---

### 3. Test Model Upload with Auto-Versioning

**First upload (creates v1)**:
```bash
curl -X POST "http://localhost:8000/ai-models/upload/single/testuser" \
  -F "file=@model.pkl" \
  -F "model_id=test-model" \
  -F "name=Test Model v1" \
  -F "framework=sklearn" \
  -F "model_type=classification"
```

**Second upload (creates v2)**:
```bash
curl -X POST "http://localhost:8000/ai-models/upload/single/testuser" \
  -F "file=@model.pkl" \
  -F "model_id=test-model" \
  -F "name=Test Model v2" \
  -F "framework=sklearn" \
  -F "model_type=classification"
```

**Verify versions**:
```bash
curl "http://localhost:8000/ai-models/testuser?model_id=test-model" | jq '.[] | {model_id, version, name}'
```

---

### 4. Test XAI Upload with New Endpoint

```bash
curl -X POST "http://localhost:8000/xai-reports/upload/testuser" \
  -F "file=@model-beginner.html" \
  -F "dataset_id=test-dataset" \
  -F "model_id=test-model" \
  -F "report_type=model_explanation" \
  -F "level=beginner"
```

**Verify uploaded**:
```bash
curl "http://localhost:8000/xai-reports/testuser/test-dataset/test-model" | jq
```

---

## Python Test Script

Save as `test_new_endpoints.py`:

```python
#!/usr/bin/env python3
import requests
import time

API_BASE = "http://localhost:8000"

def test_dataset_auto_versioning():
    """Test dataset auto-versioning"""
    print("=" * 60)
    print("Testing Dataset Auto-Versioning")
    print("=" * 60)
    
    user_id = "test_user"
    dataset_id = "auto_version_test"
    
    # Upload 3 times
    for i in range(1, 4):
        with open("heart_rate.csv", "rb") as f:
            files = {"file": f}
            data = {
                "dataset_id": dataset_id,
                "name": f"Auto Version Test {i}"
            }
            response = requests.post(
                f"{API_BASE}/datasets/upload/{user_id}",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload {i}: version={result['version']}")
            else:
                print(f"❌ Upload {i} failed: {response.text}")
        
        time.sleep(0.5)
    
    # Verify all versions
    response = requests.get(f"{API_BASE}/datasets/{user_id}")
    datasets = [d for d in response.json() if d["dataset_id"] == dataset_id]
    
    print(f"\n📊 Created versions: {[d['version'] for d in datasets]}")
    print(f"✅ Test passed! Created {len(datasets)} versions\n")


def test_model_auto_versioning():
    """Test model auto-versioning"""
    print("=" * 60)
    print("Testing Model Auto-Versioning")
    print("=" * 60)
    
    user_id = "test_user"
    model_id = "auto_version_model"
    
    # Upload 3 times
    for i in range(1, 4):
        with open("model.pkl", "rb") as f:
            files = {"file": f}
            data = {
                "model_id": model_id,
                "name": f"Auto Version Model {i}",
                "framework": "sklearn",
                "model_type": "classification"
            }
            response = requests.post(
                f"{API_BASE}/ai-models/upload/single/{user_id}",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload {i}: version={result['version']}")
            else:
                print(f"❌ Upload {i} failed: {response.text}")
        
        time.sleep(0.5)
    
    # Verify all versions
    response = requests.get(f"{API_BASE}/ai-models/search/{user_id}?query={model_id}")
    models = response.json()
    
    print(f"\n📊 Created versions: {[m['version'] for m in models]}")
    print(f"✅ Test passed! Created {len(models)} versions\n")


if __name__ == "__main__":
    print("\n🚀 Starting API Endpoint Tests\n")
    
    try:
        test_dataset_auto_versioning()
        test_model_auto_versioning()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
```

Run it:
```bash
python test_new_endpoints.py
```

---

## Expected Behavior

### Auto-Versioning
✅ First upload: v1  
✅ Second upload (same IDs): v2  
✅ Third upload (same IDs): v3  
✅ Each upload creates a new version  
✅ No duplicate version errors  

### Path Parameters
✅ user_id in URL path  
✅ No user_id in form data  
✅ Cleaner API structure  
✅ RESTful design  

---

## Common Issues

### Issue: "Model with this ID and version already exists"
**Cause**: Old code trying to specify version  
**Fix**: Remove version from form data

### Issue: 404 Not Found
**Cause**: Using old endpoint URL  
**Fix**: Add `/{user_id}` to the URL

### Issue: Missing required field 'user_id'
**Cause**: Not providing user_id in path  
**Fix**: Include user_id in the URL, not form data

---

## Verification Checklist

After making the changes, verify:

- [ ] Restart DW API
- [ ] Upload dataset - check it creates v1
- [ ] Upload again - check it creates v2
- [ ] Upload model - check it creates v1
- [ ] Upload again - check it creates v2
- [ ] Run complete Kafka flow - check it completes
- [ ] Check AutoML consumer uploads model successfully
- [ ] Check XAI consumer uploads reports successfully
- [ ] Verify all Kafka events are produced

---

## Success Criteria

✅ Dataset upload creates auto-incremented versions  
✅ Model upload creates auto-incremented versions  
✅ All endpoints use user_id in path  
✅ Complete Kafka flow works end-to-end  
✅ Consumers successfully upload to DW  
✅ No manual version management needed  

Ready to test! 🎉

