# Dataset Folder Upload - Complete Implementation

## ✅ Implementation Complete

Full support for folder uploads with multiple files, including upload, download, delete, and file listing operations.

---

## 🎯 Features Implemented

### 1. Upload Folder ✅
- **Endpoint**: `POST /datasets/upload/folder/{user_id}`
- Upload ZIP archives with multiple files
- Auto-versioning (v1, v2, v3, etc.)
- Simplified metadata (no content analysis)
- Kafka event integration

### 2. Download Folder ✅
- **Endpoint**: `GET /datasets/{user_id}/{dataset_id}/download`
- Download entire folder as ZIP
- Download specific file by filename
- Version-specific downloads supported

### 3. Delete Folder ✅
- **Endpoint**: `DELETE /datasets/{user_id}/{dataset_id}`
- Delete all files in folder
- Delete all versions
- Cascade delete (files + metadata)

### 4. List Files ✅
- **Endpoint**: `GET /datasets/{user_id}/{dataset_id}/files`
- List all files in folder
- Shows sizes, types, hashes
- Version-specific listing supported

---

## 📊 Single File vs Folder Upload

| Feature | Single File | Folder |
|---------|-------------|--------|
| **Endpoint** | `/upload/{user_id}` | `/upload/folder/{user_id}` |
| **Input** | Single file | ZIP archive |
| **Metadata** | Full (columns, rows, types) | Simplified (file list) |
| **Download** | Direct file | ZIP or specific file |
| **Delete** | Single file | All files in folder |
| **List Files** | Single item | All files |

---

## 🚀 Quick Start

### Upload Folder

```bash
# Create ZIP
zip -r dataset.zip file1.csv file2.json metadata.txt

# Upload
curl -X POST "http://localhost:8000/datasets/upload/folder/user123" \
  -F "zip_file=@dataset.zip" \
  -F "dataset_id=my-dataset" \
  -F "name=My Dataset"
```

### List Files

```bash
curl "http://localhost:8000/datasets/user123/my-dataset/files" | jq
```

### Download Specific File

```bash
curl -o file1.csv \
  "http://localhost:8000/datasets/user123/my-dataset/download?filename=file1.csv"
```

### Download Entire Folder

```bash
curl -o my-dataset.zip \
  "http://localhost:8000/datasets/user123/my-dataset/download"
```

### Delete Dataset

```bash
curl -X DELETE "http://localhost:8000/datasets/user123/my-dataset"
```

---

## 💻 Python Example

```python
import requests
import zipfile

API_BASE = "http://localhost:8000"

# 1. Create ZIP
with zipfile.ZipFile('dataset.zip', 'w') as zipf:
    zipf.write('data1.csv')
    zipf.write('data2.csv')
    zipf.write('metadata.json')

# 2. Upload folder
with open('dataset.zip', 'rb') as f:
    files = {'zip_file': f}
    data = {
        'dataset_id': 'test-folder',
        'name': 'Test Folder Dataset'
    }
    response = requests.post(
        f"{API_BASE}/datasets/upload/folder/user123",
        files=files,
        data=data
    )
    print(f"Uploaded: {response.json()['version']}")

# 3. List files
response = requests.get(f"{API_BASE}/datasets/user123/test-folder/files")
files = response.json()
print(f"Files: {[f['filename'] for f in files]}")

# 4. Download specific file
response = requests.get(
    f"{API_BASE}/datasets/user123/test-folder/download",
    params={"filename": "data1.csv"}
)
with open("downloaded.csv", "wb") as f:
    f.write(response.content)

# 5. Download entire folder
response = requests.get(f"{API_BASE}/datasets/user123/test-folder/download")
with open("downloaded_folder.zip", "wb") as f:
    f.write(response.content)

# 6. Delete dataset
response = requests.delete(f"{API_BASE}/datasets/user123/test-folder")
print(f"Deleted: {response.json()}")
```

---

## 📋 Files Modified

### Backend Code
- ✅ `app/models/dataset.py` - Added DatasetFile, folder support
- ✅ `app/services/file_service.py` - Added folder operations
- ✅ `app/api/datasets.py` - Added folder endpoints

### Testing & Documentation
- ✅ `test_dataset_folder_operations.py` - Complete test script
- ✅ `Documentation/DATASET_FOLDER_UPLOAD.md` - Upload documentation
- ✅ `Documentation/DATASET_DOWNLOAD_DELETE.md` - Download/delete documentation
- ✅ `Documentation/DATASET_FOLDER_COMPLETE.md` - This summary

---

## 🔍 API Endpoints

### Upload
```
POST /datasets/upload/{user_id}              # Single file
POST /datasets/upload/folder/{user_id}       # Folder (ZIP)
```

### Download
```
GET /datasets/{user_id}/{dataset_id}/download
    ?filename={filename}  # Optional: specific file

GET /datasets/{user_id}/{dataset_id}/version/{version}/download
    ?filename={filename}  # Optional: specific file
```

### List Files
```
GET /datasets/{user_id}/{dataset_id}/files
    ?version={version}    # Optional: specific version
```

### Delete
```
DELETE /datasets/{user_id}/{dataset_id}              # All versions
DELETE /datasets/{user_id}/{dataset_id}/version/{version}  # Specific version
```

---

## ⚙️ Technical Implementation

### Folder Upload
```python
# Extract ZIP → Upload each file → Store metadata
dataset_files, total_size = await file_service.upload_dataset_folder(
    zip_file=zip_file,
    user_id=user_id,
    dataset_id=dataset_id,
    version=version,
    preserve_structure=True
)
```

### Folder Download
```python
# List files → Create ZIP → Stream response
zip_data = await file_service.download_folder_as_zip(folder_path)
return StreamingResponse(BytesIO(zip_data), media_type='application/zip')
```

### Folder Delete
```python
# List files → Delete each file → Remove metadata
deleted_count = await file_service.delete_folder_files(folder_path)
```

---

## 🎯 Metadata Differences

### Single File Metadata
```json
{
  "is_folder": false,
  "file_type": "csv",
  "file_size": 1024,
  "original_filename": "data.csv",
  "columns": ["col1", "col2"],
  "row_count": 100,
  "data_types": {"col1": "int64"},
  "files": null
}
```

### Folder Metadata
```json
{
  "is_folder": true,
  "file_type": "csv, json, txt",
  "file_size": 5120,
  "original_filename": "dataset.zip",
  "columns": null,
  "row_count": null,
  "data_types": null,
  "files": [
    {
      "filename": "data1.csv",
      "file_path": "datasets/user123/dataset1/v1/data1.csv",
      "file_size": 2048,
      "file_type": "csv",
      "file_hash": "abc123..."
    },
    {
      "filename": "data2.csv",
      "file_path": "datasets/user123/dataset1/v1/data2.csv",
      "file_size": 2048,
      "file_type": "csv",
      "file_hash": "def456..."
    }
  ],
  "custom_metadata": {
    "file_count": 2,
    "preserve_structure": true
  }
}
```

---

## ✅ Design Decisions

Per your requirements:

1. **Single file**: Full metadata extraction ✅
   - Columns, rows, data types analyzed
   - Automatic detection of structure

2. **Folder**: Simplified metadata ✅
   - Only file inventory (names, sizes, types)
   - No content analysis (faster)
   - No CSV parsing or column detection

3. **Download flexibility**: ✅
   - Download entire folder as ZIP
   - OR download specific file by name
   - Same pattern as AI models

4. **Delete properly**: ✅
   - Single file: Delete one file
   - Folder: Delete all files recursively
   - Cascade delete metadata

---

## 🧪 Testing

Run the complete test suite:

```bash
python test_dataset_folder_operations.py
```

Expected output:
```
📁 Creating test files...
✅ Created test files and ZIP

1️⃣ Testing Folder Upload
✅ Folder uploaded successfully!
   Version: v1
   Is folder: True
   File count: 3

2️⃣ Testing List Files
✅ Found 3 files:
   - test_data1.csv (35 bytes, csv)
   - test_data2.csv (38 bytes, csv)
   - test_metadata.json (XX bytes, json)

3️⃣ Testing Download Specific File: test_data1.csv
✅ Downloaded test_data1.csv (35 bytes)

4️⃣ Testing Download Folder as ZIP
✅ Downloaded folder as ZIP (XXX bytes)
   Files in ZIP: 3

5️⃣ Testing Delete Dataset
✅ Deleted successfully!
   Files deleted: 3
✅ Verified: Dataset no longer exists

✅ All tests completed!
```

---

## 🎉 Summary

**Complete folder support for datasets:**
- ✅ Upload multiple files as ZIP
- ✅ Simplified metadata (per your specs)
- ✅ Download entire folder or specific files
- ✅ Delete folder with all files
- ✅ List files in folder
- ✅ Full versioning support
- ✅ Kafka event integration
- ✅ Backward compatible with single files

**Feature parity with AI models:**
- ✅ Same upload pattern
- ✅ Same download pattern  
- ✅ Same delete pattern
- ✅ Same file listing

The dataset API now has complete folder support! 🚀

