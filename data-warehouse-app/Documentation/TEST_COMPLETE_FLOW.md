# Testing the Complete Kafka Orchestration Flow

## Overview
This guide will help you test the **complete end-to-end flow** using dummy files:
- Dataset Upload → Bias Detection → AutoML Training → XAI Generation

## Prerequisites

You need these dummy files in the **root directory**:

### 1. Create Dummy Model File

Create a simple `model.pkl` file for testing:

```python
# create_dummy_model.py
import pickle

# Create a simple dummy object
dummy_model = {
    'type': 'classifier',
    'algorithm': 'RandomForest',
    'accuracy': 0.92,
    'created': 'test'
}

# Save it
with open('model.pkl', 'wb') as f:
    pickle.dump(dummy_model, f)

print("✅ Created dummy model.pkl")
```

Run it:
```bash
python create_dummy_model.py
```

### 2. Verify HTML Files Exist

You should already have these files (from your project layout):
- ✅ `model-beginner.html`
- ✅ `data-beginner.html`

If not, you can use any HTML file and rename it.

---

## Complete Testing Steps

### Step 1: Start All Services

**Terminal 1: Kafka**
```bash
docker-compose up -d kafka zookeeper
```

**Terminal 2: Data Warehouse API**
```bash
python run.py
```

Wait for: `"Kafka producer initialized"`

### Step 2: Start All Consumers

**Terminal 3: Agentic Core (Orchestrator)**
```bash
python kafka_agentic_core_consumer_example.py
```

**Terminal 4: Bias Detector**
```bash
python kafka_bias_detector_consumer_example.py
```

**Terminal 5: AutoML Consumer**
```bash
python kafka_automl_consumer_example.py
```

**Terminal 6: XAI Consumer**
```bash
python kafka_xai_consumer_example.py
```

### Step 3: Upload a Test Dataset

**Terminal 7: Test Upload**
```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@heart_rate.csv" \
  -F "user_id=test_user" \
  -F "dataset_id=flow_test_1" \
  -F "name=Complete Flow Test"
```

(Use any CSV file you have)

---

## Expected Flow & Logs

### 1. Dataset Upload ✓
**Terminal 2 (DW API):** 
```
INFO: Dataset uploaded successfully
INFO: Kafka event sent: dataset.uploaded
```

**Terminal 3 (Agentic Core):**
```
[Dataset Event] Message received
[Agentic Core] Produced bias trigger event to bias-trigger-events
```

---

### 2. Bias Detection ✓
**Terminal 4 (Bias Detector):**
```
Message received
Target column=target
Task type=classification
Loaded dataset into pandas DataFrame with shape (27, 4)
Saved bias report
```

**Terminal 2 (DW API):**
```
INFO: Bias Kafka event sent for dataset_id=flow_test_1
```

**Terminal 3 (Agentic Core):**
```
[Bias Event] Message received
[User Report] Bias report completed for dataset flow_test_1
[Agentic Core] Produced AutoML trigger event to automl-trigger-events
```

---

### 3. AutoML Training ✓
**Terminal 5 (AutoML Consumer):**
```
AutoML Trigger Message received
Processing AutoML trigger for dataset flow_test_1
Dataset downloaded: XXX bytes
Dataset loaded: 27 rows, 4 columns
================================================================================
Training model (using dummy model.pkl for testing)
  - Target column: target
  - Task type: classification
  - Dataset shape: (27, 4)
================================================================================
Uploading model to DW: automl_flow_test_1_1728561234
✅ Model uploaded to DW successfully!
   Model ID: automl_flow_test_1_1728561234
   AutoML event will be automatically sent by the DW
AutoML processing completed for dataset flow_test_1
```

**Terminal 2 (DW API):**
```
INFO: AI model uploaded successfully: automl_flow_test_1_1728561234
INFO: AutoML event sent for model_id=automl_flow_test_1_1728561234
```

**Terminal 3 (Agentic Core):**
```
[AutoML Event] Message received
[User Report] Model automl_flow_test_1_1728561234 trained successfully
  Training accuracy: 0.92
[Agentic Core] Produced XAI trigger event to xai-trigger-events
```

---

### 4. XAI Generation ✓
**Terminal 6 (XAI Consumer):**
```
XAI Trigger Message received
Processing XAI trigger for model automl_flow_test_1_1728561234
Dataset downloaded: XXX bytes
Model metadata fetched: AutoML Model - automl_flow_test_1_1728561234
================================================================================
Generating XAI explanations (using dummy HTML files for testing)
  - Model: AutoML Model - automl_flow_test_1_1728561234
  - Framework: sklearn
  - Level: beginner
================================================================================
Uploading model explanation report: model-beginner.html
✅ Model explanation report uploaded successfully!
   Report type: model_explanation
   Level: beginner
Uploading data explanation report: data-beginner.html
✅ Data explanation report uploaded successfully!
   Report type: data_explanation
   Level: beginner
   XAI events will be automatically sent by the DW
XAI processing completed for model automl_flow_test_1_1728561234
```

**Terminal 2 (DW API):**
```
INFO: XAI report uploaded successfully
INFO: XAI event sent for model_id=automl_flow_test_1_1728561234
INFO: XAI report uploaded successfully
INFO: XAI event sent for model_id=automl_flow_test_1_1728561234
```

**Terminal 3 (Agentic Core):**
```
[XAI Event] Message received
[User Report] XAI report generated for model automl_flow_test_1_1728561234
  Report type: model_explanation
  Level: beginner
[XAI Event] Message received
[User Report] XAI report generated for model automl_flow_test_1_1728561234
  Report type: data_explanation
  Level: beginner
================================================================================
ML PIPELINE COMPLETED!
  Dataset: flow_test_1
  Model: automl_flow_test_1_1728561234
  User: test_user
================================================================================
```

---

## Verify in Kafka UI

If you have Kafka UI running (usually at http://localhost:8080), you should see messages in all topics:

1. ✅ `dataset-events` (1 message)
2. ✅ `bias-trigger-events` (1 message)
3. ✅ `bias-events` (1 message)
4. ✅ `automl-trigger-events` (1 message)
5. ✅ `automl-events` (1 message)
6. ✅ `xai-trigger-events` (1 message)
7. ✅ `xai-events` (2 messages - one for each report type)

---

## Verify in Data Warehouse

Check that all resources were created:

### 1. Dataset
```bash
curl http://localhost:8000/datasets/test_user/flow_test_1
```

### 2. Bias Report
```bash
curl http://localhost:8000/bias-reports/test_user/flow_test_1
```

### 3. AI Model
```bash
curl http://localhost:8000/ai-models/test_user/automl_flow_test_1_XXXXX
```

### 4. XAI Reports
```bash
# List all XAI reports for the model
curl http://localhost:8000/xai-reports/test_user/flow_test_1/automl_flow_test_1_XXXXX

# View model explanation in browser
open http://localhost:8000/xai-reports/test_user/flow_test_1/automl_flow_test_1_XXXXX/model_explanation/beginner/view

# View data explanation in browser
open http://localhost:8000/xai-reports/test_user/flow_test_1/automl_flow_test_1_XXXXX/data_explanation/beginner/view
```

---

## Troubleshooting

### AutoML Consumer says "model.pkl not found"
```bash
# Create it:
python -c "import pickle; pickle.dump({'test': 'model'}, open('model.pkl', 'wb'))"
```

### XAI Consumer says "HTML files not found"
```bash
# Use the existing ones or create dummy ones:
echo "<html><body>Model Explanation</body></html>" > model-beginner.html
echo "<html><body>Data Explanation</body></html>" > data-beginner.html
```

### Flow stops at any stage
1. Check all consumers are running
2. Check for errors in logs
3. Check Kafka is running: `docker ps | grep kafka`
4. Restart the service that's stuck

### Kafka events not showing
1. Check DW API logs for "Kafka producer initialized"
2. Check for Kafka send errors in DW logs
3. Verify topic names match in config.py

---

## Success Criteria

✅ You should see:
1. All 6 terminals showing activity
2. Messages flowing in sequence
3. No errors in any terminal
4. "ML PIPELINE COMPLETED!" message in Agentic Core
5. All resources created in DW
6. All 7 Kafka topics have messages

---

## What's Next?

Now that the infrastructure works end-to-end, you can replace the dummy implementations with real ML logic:

1. **Bias Detector**: Implement actual bias detection algorithms
2. **AutoML Consumer**: Integrate real AutoML libraries (AutoGluon, FLAML, etc.)
3. **XAI Consumer**: Implement SHAP/LIME explanation generation
4. **Agentic Core**: Add user interaction and decision-making

The Kafka orchestration is solid and production-ready! 🎉

