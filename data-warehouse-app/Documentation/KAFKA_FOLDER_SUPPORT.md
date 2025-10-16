# Kafka Messages - Folder Dataset Support

## Overview

Kafka dataset messages now include information about whether a dataset is a single file or a folder, allowing consumers to handle downloads appropriately.

---

## 🔄 Updated Kafka Message Structure

### Dataset Event Payload (Updated)

```json
{
  "event_type": "dataset.uploaded",
  "event_id": "dataset.uploaded_dataset123_1728561234",
  "timestamp": "2025-10-10T12:00:00.000000",
  "dataset": {
    "dataset_id": "dataset123",
    "user_id": "user123",
    "name": "My Dataset",
    "version": "v1",
    "file_type": "csv",
    "file_size": 2048,
    "is_folder": false,           // ✨ NEW
    "file_count": 1,              // ✨ NEW
    "columns": ["col1", "col2"],
    "row_count": 100,
    // ... other fields
  }
}
```

### Folder Dataset Event Payload

```json
{
  "event_type": "dataset.uploaded",
  "event_id": "dataset.uploaded_folder123_1728561234",
  "timestamp": "2025-10-10T12:00:00.000000",
  "dataset": {
    "dataset_id": "folder123",
    "user_id": "user123",
    "name": "Multi-File Dataset",
    "version": "v1",
    "file_type": "csv, json, txt",
    "file_size": 5120,
    "is_folder": true,            // ✨ NEW - indicates folder
    "file_count": 3,              // ✨ NEW - number of files
    "columns": null,              // null for folders
    "row_count": null,            // null for folders
    // ... other fields
  }
}
```

---

## 🔧 Consumer Implementation

### Check is_folder Field

```python
# Extract from Kafka message
dataset = value.get("dataset", {})
is_folder = dataset.get("is_folder", False)
file_count = dataset.get("file_count", 1)

if is_folder:
    logger.info(f"Dataset is a FOLDER with {file_count} files")
    # Handle folder logic
else:
    logger.info("Dataset is a SINGLE FILE")
    # Handle single file logic
```

### Download Based on Type

```python
import requests
from io import BytesIO

# Download dataset (API returns ZIP for folders, file for single)
file_bytes = download_dataset_file(user_id, dataset_id)

if is_folder:
    # Extract ZIP
    extracted_files = extract_dataset_folder(file_bytes, "temp_dataset")
    logger.info(f"Extracted {len(extracted_files)} files")
    
    # Process multiple files
    for file_path in extracted_files:
        # Your processing logic here
        pass
else:
    # Process single file
    df = pd.read_csv(BytesIO(file_bytes))
    # Your processing logic here
```

### Unzip Helper Function

```python
def extract_dataset_folder(zip_bytes: bytes, extract_to: str = "temp_dataset") -> list:
    """
    Extract ZIP file containing dataset folder
    
    Returns:
        List of extracted file paths
    """
    import zipfile
    import os
    from io import BytesIO
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # List extracted files
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_files.append(file_path)
    
    return extracted_files
```

---

## 📝 Updated Consumer Examples

### 1. Agentic Core Consumer

**File**: `kafka_agentic_core_consumer_example.py`

**Changes:**
- ✅ Checks `is_folder` field in dataset events
- ✅ Downloads dataset (single file or ZIP)
- ✅ Extracts ZIP if folder
- ✅ Finds and loads CSV files for analysis
- ✅ Handles multiple files in folder

**Example Output:**
```
[Dataset Event] Message received
Dataset type: FOLDER
File count: 3
Downloaded dataset: 5120 bytes
Extracting folder dataset...
Extracted 3 files:
  - temp_dataset_dataset123/data1.csv
  - temp_dataset_dataset123/data2.csv
  - temp_dataset_dataset123/metadata.json
Found 2 CSV file(s), loading first one for analysis
Loaded CSV with shape (100, 5)
```

### 2. Bias Detector Consumer

**File**: `kafka_bias_detector_consumer_example.py`

**Changes:**
- ✅ Checks `is_folder` field in bias trigger events
- ✅ Downloads dataset (single file or ZIP)
- ✅ Extracts ZIP if folder
- ✅ Finds and loads CSV files for bias analysis
- ✅ Handles multiple files in folder

**Example Output:**
```
Target column=target
Task type=classification
Dataset type: FOLDER
File count: 3
Downloaded dataset: 5120 bytes
Extracting folder dataset...
Extracted 3 files:
  - temp_bias_dataset123/train.csv
  - temp_bias_dataset123/test.csv
  - temp_bias_dataset123/metadata.json
Found 2 CSV file(s), loading first one for bias analysis
Loaded CSV with shape (1000, 10)
```

---

## 🔄 Complete Flow for Folder Datasets

```
1. User uploads folder (ZIP)
   ↓
2. DW API extracts and stores files
   ↓
3. Kafka message sent with is_folder=true
   ↓
4. Agentic Core receives message
   ↓
5. Agentic Core checks is_folder field
   ↓
6. Agentic Core downloads as ZIP
   ↓
7. Agentic Core extracts ZIP
   ↓
8. Agentic Core processes files
   ↓
9. Triggers bias detection
   ↓
10. Bias detector receives trigger
    ↓
11. Bias detector checks is_folder in metadata
    ↓
12. Bias detector downloads and extracts
    ↓
13. Bias detector analyzes files
    ↓
... (continues through pipeline)
```

---

## 💻 Code Examples

### Full Consumer Implementation

```python
import os
import zipfile
import pandas as pd
from io import BytesIO

def download_dataset_file(user_id: str, dataset_id: str) -> bytes:
    """Download dataset (returns single file or ZIP)"""
    url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/download"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def extract_dataset_folder(zip_bytes: bytes, extract_to: str) -> list:
    """Extract ZIP and return list of file paths"""
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            extracted_files.append(os.path.join(root, file))
    
    return extracted_files

async def process_dataset_event(event: dict):
    """Process dataset event - handles both single file and folder"""
    dataset = event.get("dataset", {})
    user_id = dataset.get("user_id")
    dataset_id = dataset.get("dataset_id")
    is_folder = dataset.get("is_folder", False)
    file_count = dataset.get("file_count", 1)
    
    logger.info(f"Dataset: {dataset_id}")
    logger.info(f"Type: {'FOLDER' if is_folder else 'SINGLE FILE'}")
    logger.info(f"Files: {file_count}")
    
    # Download dataset
    file_bytes = download_dataset_file(user_id, dataset_id)
    
    if is_folder:
        # Extract and process multiple files
        logger.info("Extracting folder...")
        files = extract_dataset_folder(file_bytes, f"temp_{dataset_id}")
        logger.info(f"Extracted {len(files)} files:")
        
        for file_path in files:
            logger.info(f"  - {file_path}")
        
        # Example: Process all CSV files
        csv_files = [f for f in files if f.endswith('.csv')]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {csv_file}: {df.shape}")
            # TODO: Your processing logic here
    else:
        # Process single file
        df = pd.read_csv(BytesIO(file_bytes))
        logger.info(f"Loaded dataset: {df.shape}")
        # TODO: Your processing logic here
```

---

## 🧪 Testing

### Test Single File Upload

```bash
# Upload single file
curl -X POST "http://localhost:8000/datasets/upload/testuser" \
  -F "file=@data.csv" \
  -F "dataset_id=single-test" \
  -F "name=Single File Test"

# Check Kafka message (in consumer logs)
# Should show: is_folder=false, file_count=1
```

### Test Folder Upload

```bash
# Create and upload folder
zip -r dataset.zip file1.csv file2.csv metadata.json

curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@dataset.zip" \
  -F "dataset_id=folder-test" \
  -F "name=Folder Test"

# Check Kafka message (in consumer logs)
# Should show: is_folder=true, file_count=3
```

### Verify Consumer Behavior

**Expected for single file:**
```
Dataset type: SINGLE FILE
Downloaded dataset: 1024 bytes
Loaded single file dataset with shape (100, 5)
```

**Expected for folder:**
```
Dataset type: FOLDER
File count: 3
Downloaded dataset: 5120 bytes
Extracting folder dataset...
Extracted 3 files:
  - temp_dataset_folder-test/file1.csv
  - temp_dataset_folder-test/file2.csv
  - temp_dataset_folder-test/metadata.json
Found 2 CSV file(s), loading first one for analysis
Loaded CSV with shape (100, 5)
```

---

## 📋 Files Modified

### Kafka Service
- ✅ `app/services/kafka_service.py` - Added `is_folder` and `file_count` to dataset events

### Consumer Examples
- ✅ `kafka_agentic_core_consumer_example.py` - Added folder handling and unzip
- ✅ `kafka_bias_detector_consumer_example.py` - Added folder handling and unzip

### Documentation
- ✅ `Documentation/KAFKA_FOLDER_SUPPORT.md` - This file

---

## 🎯 Key Changes

### Kafka Message
```python
# Before
"dataset": {
    "dataset_id": "...",
    "file_type": "csv",
    // ...
}

# After
"dataset": {
    "dataset_id": "...",
    "file_type": "csv",
    "is_folder": false,      // ✨ NEW
    "file_count": 1,         // ✨ NEW
    // ...
}
```

### Consumer Logic
```python
# Before
file_bytes = download_dataset_file(user_id, dataset_id)
df = pd.read_csv(BytesIO(file_bytes))

# After
file_bytes = download_dataset_file(user_id, dataset_id)

if is_folder:
    files = extract_dataset_folder(file_bytes, "temp_dir")
    # Process multiple files
else:
    df = pd.read_csv(BytesIO(file_bytes))
    # Process single file
```

---

## ✅ Benefits

1. **Smart Download** - Consumers automatically handle both types
2. **Unzip Logic** - Ready-to-use extraction code
3. **File Discovery** - Automatically finds CSV files in folders
4. **Backward Compatible** - Single file logic unchanged
5. **Easy to Extend** - Clear pattern for processing multiple files

---

## 🚀 Quick Start

### For Consumer Developers

1. **Check is_folder field** in Kafka message
2. **Download dataset** (API returns appropriate format)
3. **Extract if folder** using provided helper function
4. **Process files** as needed

```python
# In your consumer
is_folder = dataset.get("is_folder", False)
file_bytes = download_dataset_file(user_id, dataset_id)

if is_folder:
    files = extract_dataset_folder(file_bytes)
    # Your multi-file logic here
else:
    # Your single-file logic here
```

---

## 📊 Summary

**Changes:**
- ✅ Kafka messages include `is_folder` and `file_count`
- ✅ Consumers check `is_folder` before processing
- ✅ Consumers download appropriate format (file vs ZIP)
- ✅ Consumers extract ZIP for folders
- ✅ Ready-to-use unzip helper function
- ✅ Example code for processing multiple files

**Consumers Updated:**
- ✅ Agentic Core - Full folder support
- ✅ Bias Detector - Full folder support

**Result:**
The complete Kafka pipeline now supports both single file and folder datasets! 🎉

