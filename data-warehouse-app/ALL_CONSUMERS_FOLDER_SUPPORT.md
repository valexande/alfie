# All Kafka Consumers - Folder Support

## ✅ Implementation Complete

All Kafka consumer examples now support both single file and folder datasets with automatic unzip functionality!

---

## 🎯 Updated Consumers

### 1. Agentic Core ✅
**File**: `kafka_agentic_core_consumer_example.py`
- Checks `is_folder` in dataset events
- Extracts folder datasets
- Finds and loads CSV files
- Ready for multi-file processing

### 2. Bias Detector ✅
**File**: `kafka_bias_detector_consumer_example.py`
- Checks `is_folder` in metadata
- Extracts folder datasets
- Analyzes first CSV file found
- Ready for multi-file bias detection

### 3. AutoML Consumer ✅
**File**: `kafka_automl_consumer_example.py`
- Checks `is_folder` in dataset metadata
- Extracts folder datasets
- Uses first CSV file for training
- Ready for multi-file AutoML

### 4. XAI Consumer ✅
**File**: `kafka_xai_consumer_example.py`
- Checks `is_folder` in dataset metadata
- Extracts folder datasets
- All files available for XAI analysis
- Ready for multi-file explainability

---

## 🔧 What Each Consumer Does

### Pattern Used in All Consumers

```python
# 1. Fetch dataset metadata
metadata = fetch_dataset_metadata(user_id, dataset_id)
is_folder = metadata.get("is_folder", False)

# 2. Download dataset (file or ZIP)
file_bytes = download_dataset_file(user_id, dataset_id)

# 3. Handle based on type
if is_folder:
    # Extract ZIP
    extracted_files = extract_dataset_folder(file_bytes, f"temp_{dataset_id}")
    logger.info(f"Extracted {len(extracted_files)} files")
    
    # Find CSV files
    csv_files = [f for f in extracted_files if f.endswith('.csv')]
    
    # Process files
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Your logic here
else:
    # Single file
    df = pd.read_csv(BytesIO(file_bytes))
    # Your logic here
```

---

## 📊 Consumer-Specific Handling

### Agentic Core
**Purpose**: Orchestrate pipeline, interact with user

**Folder Handling**:
- Extracts all files
- Lists all files for user
- Finds CSV for initial analysis
- TODO: Ask user which file(s) to process

**Example Output**:
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files:
  - temp_dataset_test/train.csv
  - temp_dataset_test/test.csv
  - temp_dataset_test/metadata.json
Found 2 CSV file(s), loading first one for analysis
```

### Bias Detector
**Purpose**: Detect bias in data

**Folder Handling**:
- Extracts all files
- Finds CSV files
- Analyzes first CSV for bias
- TODO: Analyze all CSV files or combined dataset

**Example Output**:
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files
Found 2 CSV file(s), loading first one for bias analysis
Loaded CSV with shape (1000, 10)
```

### AutoML Consumer
**Purpose**: Train ML models

**Folder Handling**:
- Extracts all files
- Finds CSV files for training
- Uses first CSV as training data
- TODO: Combine multiple CSVs or use train/test splits

**Example Output**:
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files
Found 2 CSV file(s), loading first one for training
Dataset loaded: 1000 rows, 10 columns
Training model (using dummy model.pkl for testing)
```

### XAI Consumer
**Purpose**: Generate explainability reports

**Folder Handling**:
- Extracts all files
- All files available for analysis
- Can access any file for explanations
- TODO: Use specific files for different XAI methods

**Example Output**:
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files:
  - temp_xai_test/train.csv
  - temp_xai_test/test.csv
  - temp_xai_test/metadata.json
NOTE: All extracted files are available for XAI analysis
```

---

## 🔄 Complete Flow with Folders

```
1. User uploads folder (3 CSV files)
   ↓
2. DW: is_folder=true, file_count=3
   ↓
3. Kafka: dataset-events with is_folder=true
   ↓
4. Agentic Core: Downloads ZIP → Extracts → Finds CSVs
   ↓
5. Agentic Core: Produces bias-trigger
   ↓
6. Bias Detector: Downloads ZIP → Extracts → Analyzes CSV
   ↓
7. Bias Detector: Posts report → Kafka bias-events
   ↓
8. Agentic Core: Produces automl-trigger
   ↓
9. AutoML: Downloads ZIP → Extracts → Trains on CSV
   ↓
10. AutoML: Uploads model → Kafka automl-events
    ↓
11. Agentic Core: Produces xai-trigger
    ↓
12. XAI: Downloads ZIP → Extracts → All files available
    ↓
13. XAI: Generates reports → Kafka xai-events
    ↓
14. Agentic Core: ML PIPELINE COMPLETED! ✓
```

---

## 💻 Code Example - Processing Multiple Files

### Example: Process All CSV Files in Folder

```python
# In any consumer
if is_folder:
    extracted_files = extract_dataset_folder(file_bytes, "temp_dir")
    
    # Find all CSV files
    csv_files = [f for f in extracted_files if f.endswith('.csv')]
    
    # Process each CSV
    for csv_file in csv_files:
        logger.info(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Your processing logic here
        # Example: Bias detection on each file
        # Example: Train separate models
        # Example: Generate separate XAI reports
```

### Example: Combine Multiple CSV Files

```python
# In AutoML consumer - combine train/test splits
if is_folder:
    extracted_files = extract_dataset_folder(file_bytes, "temp_dir")
    csv_files = [f for f in extracted_files if f.endswith('.csv')]
    
    # Find train/test files
    train_file = next((f for f in csv_files if 'train' in f.lower()), None)
    test_file = next((f for f in csv_files if 'test' in f.lower()), None)
    
    if train_file and test_file:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
        # Train with proper train/test split
    else:
        # Fallback: use first CSV
        df = pd.read_csv(csv_files[0])
```

---

## 🧪 Testing All Consumers

### Step 1: Upload Folder Dataset
```bash
# Create test folder
zip -r test_folder.zip train.csv test.csv metadata.json

# Upload
curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@test_folder.zip" \
  -F "dataset_id=folder-flow-test" \
  -F "name=Folder Flow Test"
```

### Step 2: Watch Consumer Logs

**Agentic Core (Terminal 3):**
```
[Dataset Event] Message received
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files:
  - temp_dataset_folder-flow-test/train.csv
  - temp_dataset_folder-flow-test/test.csv
  - temp_dataset_folder-flow-test/metadata.json
Found 2 CSV file(s), loading first one for analysis
```

**Bias Detector (Terminal 4):**
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files
Found 2 CSV file(s), loading first one for bias analysis
```

**AutoML Consumer (Terminal 5):**
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files
Found 2 CSV file(s), loading first one for training
Dataset loaded: 100 rows, 5 columns
```

**XAI Consumer (Terminal 6):**
```
Dataset type: FOLDER
File count: 3
Extracting folder dataset...
Extracted 3 files
NOTE: All extracted files are available for XAI analysis
```

---

## 📋 Summary of Changes

### All 4 Consumers Updated

| Consumer | Added extract_dataset_folder() | Checks is_folder | Extracts ZIP | Processes Files |
|----------|-------------------------------|------------------|--------------|-----------------|
| **Agentic Core** | ✅ | ✅ | ✅ | ✅ |
| **Bias Detector** | ✅ | ✅ | ✅ | ✅ |
| **AutoML** | ✅ | ✅ | ✅ | ✅ |
| **XAI** | ✅ | ✅ | ✅ | ✅ |

### Common Features

All consumers now:
- ✅ Have `extract_dataset_folder()` helper function
- ✅ Check `is_folder` field from metadata
- ✅ Download dataset (automatically gets ZIP for folders)
- ✅ Extract ZIP if folder
- ✅ List all extracted files
- ✅ Find CSV files for processing
- ✅ Include example code for multi-file processing

---

## 🎉 Complete Implementation

**Backend API:**
- ✅ Upload folder endpoint
- ✅ Download with file selection
- ✅ Delete folder properly
- ✅ List files endpoint
- ✅ Kafka messages with is_folder

**All Consumer Scripts:**
- ✅ Agentic Core
- ✅ Bias Detector
- ✅ AutoML Consumer
- ✅ XAI Consumer

**Features:**
- ✅ Automatic folder detection
- ✅ Automatic ZIP extraction
- ✅ File discovery
- ✅ Example processing code
- ✅ Backward compatible with single files

**Documentation:**
- ✅ API documentation
- ✅ Kafka integration guide
- ✅ Consumer updates documented
- ✅ Testing guide

---

## 🚀 Ready to Test

**Restart all services:**
```bash
# Terminal 1: API
python run.py

# Terminal 2-5: Consumers
python kafka_agentic_core_consumer_example.py
python kafka_bias_detector_consumer_example.py
python kafka_automl_consumer_example.py
python kafka_xai_consumer_example.py
```

**Test with folder:**
```bash
zip -r test.zip file1.csv file2.csv
curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@test.zip" \
  -F "dataset_id=test-folder" \
  -F "name=Test Folder"
```

**Watch all 4 consumers extract and process the files!** 🎊

The complete system now supports folders across the entire pipeline! 🚀

