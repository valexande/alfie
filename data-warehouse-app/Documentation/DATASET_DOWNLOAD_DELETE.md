# Dataset Download & Delete Operations

## Overview

This document explains how to download and delete datasets, including support for both single files and folder uploads with multiple files.

---

## 📥 Download Operations

### Download Endpoints

#### 1. Download Latest Version
**Endpoint**: `GET /datasets/{user_id}/{dataset_id}/download`

**Query Parameters**:
- `filename` (optional) - For folders: download specific file by name

**Behavior**:
- **Single file dataset**: Downloads the file
- **Folder dataset (no filename)**: Downloads all files as ZIP
- **Folder dataset (with filename)**: Downloads specific file

#### 2. Download Specific Version
**Endpoint**: `GET /datasets/{user_id}/{dataset_id}/version/{version}/download`

**Query Parameters**:
- `filename` (optional) - For folders: download specific file by name

**Behavior**: Same as latest version download

#### 3. List Files (New!)
**Endpoint**: `GET /datasets/{user_id}/{dataset_id}/files`

**Query Parameters**:
- `version` (optional) - Specific version (defaults to latest)

**Response**: Array of file metadata
```json
[
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
    "file_size": 1024,
    "file_type": "csv",
    "file_hash": "def456..."
  }
]
```

---

## 📥 Download Examples

### Single File Dataset

```bash
# Download single file
curl -o downloaded.csv \
  "http://localhost:8000/datasets/user123/my-dataset/download"
```

### Folder Dataset - Download All as ZIP

```bash
# Download entire folder as ZIP
curl -o dataset.zip \
  "http://localhost:8000/datasets/user123/my-folder-dataset/download"
```

### Folder Dataset - Download Specific File

```bash
# First, list available files
curl "http://localhost:8000/datasets/user123/my-folder-dataset/files"

# Download specific file by name
curl -o data1.csv \
  "http://localhost:8000/datasets/user123/my-folder-dataset/download?filename=data1.csv"
```

### Version-Specific Downloads

```bash
# Download specific version as ZIP
curl -o dataset_v2.zip \
  "http://localhost:8000/datasets/user123/my-folder-dataset/version/v2/download"

# Download specific file from specific version
curl -o data1.csv \
  "http://localhost:8000/datasets/user123/my-folder-dataset/version/v2/download?filename=data1.csv"
```

---

## 🗑️ Delete Operations

### Delete Endpoints

#### 1. Delete All Versions
**Endpoint**: `DELETE /datasets/{user_id}/{dataset_id}`

**Behavior**:
- Deletes all versions of the dataset
- For single files: Deletes each file
- For folders: Deletes all files in each folder
- Removes metadata from MongoDB

#### 2. Delete Specific Version
**Endpoint**: `DELETE /datasets/{user_id}/{dataset_id}/version/{version}`

**Behavior**:
- Deletes specific version only
- For single file: Deletes the file
- For folder: Deletes all files in folder
- Removes metadata from MongoDB

---

## 🗑️ Delete Examples

### Delete All Versions

```bash
# Delete all versions of dataset
curl -X DELETE "http://localhost:8000/datasets/user123/my-dataset"

# Response:
{
  "message": "All 3 version(s) of dataset deleted successfully",
  "versions_deleted": 3,
  "files_deleted": 5,  # Total files across all versions
  "files_failed": 0
}
```

### Delete Specific Version

```bash
# Delete specific version
curl -X DELETE "http://localhost:8000/datasets/user123/my-dataset/version/v2"

# Response:
{
  "message": "Dataset deleted successfully",
  "files_deleted": 3,  # Files in this version
  "metadata_deleted": true
}
```

---

## 💻 Python Examples

### Download Folder as ZIP

```python
import requests

response = requests.get(
    "http://localhost:8000/datasets/user123/my-folder-dataset/download"
)

if response.status_code == 200:
    with open("dataset.zip", "wb") as f:
        f.write(response.content)
    print("✅ Downloaded folder as ZIP")
```

### Download Specific File from Folder

```python
import requests

# List available files first
response = requests.get(
    "http://localhost:8000/datasets/user123/my-folder-dataset/files"
)
files = response.json()
print(f"Available files: {[f['filename'] for f in files]}")

# Download specific file
response = requests.get(
    "http://localhost:8000/datasets/user123/my-folder-dataset/download",
    params={"filename": "data1.csv"}
)

if response.status_code == 200:
    with open("data1.csv", "wb") as f:
        f.write(response.content)
    print("✅ Downloaded specific file")
```

### Delete Dataset with All Versions

```python
import requests

response = requests.delete(
    "http://localhost:8000/datasets/user123/my-dataset"
)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Deleted {result['versions_deleted']} versions")
    print(f"   Files deleted: {result['files_deleted']}")
```

---

## 🔄 Complete Workflow Example

### Upload, List, Download, Delete

```python
import requests
import zipfile
import os

API_BASE = "http://localhost:8000"
user_id = "user123"
dataset_id = "test-folder-dataset"

# Step 1: Create and upload ZIP
print("1️⃣ Creating ZIP...")
with zipfile.ZipFile('dataset.zip', 'w') as zipf:
    zipf.write('data1.csv')
    zipf.write('data2.csv')
    zipf.write('metadata.json')

with open('dataset.zip', 'rb') as f:
    files = {'zip_file': f}
    data = {'dataset_id': dataset_id, 'name': 'Test Dataset'}
    response = requests.post(
        f"{API_BASE}/datasets/upload/folder/{user_id}",
        files=files,
        data=data
    )
    print(f"✅ Uploaded: {response.json()['version']}")

# Step 2: List files in folder
print("\n2️⃣ Listing files...")
response = requests.get(f"{API_BASE}/datasets/{user_id}/{dataset_id}/files")
files = response.json()
for f in files:
    print(f"   - {f['filename']} ({f['file_size']} bytes)")

# Step 3: Download specific file
print("\n3️⃣ Downloading specific file...")
response = requests.get(
    f"{API_BASE}/datasets/{user_id}/{dataset_id}/download",
    params={"filename": "data1.csv"}
)
with open("downloaded_data1.csv", "wb") as f:
    f.write(response.content)
print("✅ Downloaded: data1.csv")

# Step 4: Download entire folder as ZIP
print("\n4️⃣ Downloading entire folder...")
response = requests.get(f"{API_BASE}/datasets/{user_id}/{dataset_id}/download")
with open("downloaded_folder.zip", "wb") as f:
    f.write(response.content)
print("✅ Downloaded: folder as ZIP")

# Step 5: Delete dataset
print("\n5️⃣ Deleting dataset...")
response = requests.delete(f"{API_BASE}/datasets/{user_id}/{dataset_id}")
result = response.json()
print(f"✅ Deleted: {result['files_deleted']} files")
```

---

## 🎯 Feature Summary

### Download Features

| Feature | Single File | Folder |
|---------|-------------|--------|
| **Download dataset** | ✅ Returns file | ✅ Returns ZIP |
| **Download specific file** | N/A | ✅ By filename |
| **Download by version** | ✅ Supported | ✅ Supported |
| **List files** | ✅ Single item | ✅ All files |

### Delete Features

| Feature | Single File | Folder |
|---------|-------------|--------|
| **Delete all versions** | ✅ Deletes file | ✅ Deletes all files |
| **Delete specific version** | ✅ Deletes file | ✅ Deletes all files in folder |
| **Cascade delete** | ✅ File + metadata | ✅ All files + metadata |

---

## 📋 API Endpoint Summary

### Download
```
GET  /datasets/{user_id}/{dataset_id}/download
     ?filename={filename}  # Optional: specific file from folder

GET  /datasets/{user_id}/{dataset_id}/version/{version}/download
     ?filename={filename}  # Optional: specific file from folder

GET  /datasets/{user_id}/{dataset_id}/files
     ?version={version}    # Optional: specific version
```

### Delete
```
DELETE /datasets/{user_id}/{dataset_id}
       Deletes all versions

DELETE /datasets/{user_id}/{dataset_id}/version/{version}
       Deletes specific version
```

---

## ⚙️ Technical Details

### Folder Download as ZIP
- Creates ZIP in memory (no temp files)
- Preserves folder structure
- Uses streaming response for efficiency
- Compression: ZIP_DEFLATED

### Folder Delete
- Recursively deletes all files
- Uses MinIO prefix-based listing
- Logs each deleted file
- Returns count of deleted files

### File Listing
- Returns metadata for all files
- Includes file sizes, types, hashes
- Works for both single files and folders
- Cached in dataset metadata (fast)

---

## 🧪 Testing

### Test Download

```bash
# Upload folder
curl -X POST "http://localhost:8000/datasets/upload/folder/testuser" \
  -F "zip_file=@test.zip" \
  -F "dataset_id=test-folder" \
  -F "name=Test Folder"

# List files
curl "http://localhost:8000/datasets/testuser/test-folder/files" | jq

# Download entire folder
curl -o downloaded_folder.zip \
  "http://localhost:8000/datasets/testuser/test-folder/download"

# Download specific file
curl -o specific_file.csv \
  "http://localhost:8000/datasets/testuser/test-folder/download?filename=data1.csv"
```

### Test Delete

```bash
# Delete specific version
curl -X DELETE "http://localhost:8000/datasets/testuser/test-folder/version/v1"

# Verify deleted
curl "http://localhost:8000/datasets/testuser/test-folder"
# Should return 404
```

---

## 🔍 Behavior Details

### Download Folder Without Filename
```bash
curl "http://localhost:8000/datasets/user123/folder-dataset/download"
```
**Result**: ZIP file containing all files with preserved structure

### Download Folder With Filename
```bash
curl "http://localhost:8000/datasets/user123/folder-dataset/download?filename=data1.csv"
```
**Result**: Single file `data1.csv`

### Download Single File (Ignores Filename)
```bash
curl "http://localhost:8000/datasets/user123/single-dataset/download?filename=ignored"
```
**Result**: The single file (filename parameter ignored)

### Delete Folder
```bash
curl -X DELETE "http://localhost:8000/datasets/user123/folder-dataset"
```
**Result**: All files in all folders deleted recursively

---

## ✅ What Was Implemented

### File Service (`app/services/file_service.py`)
- ✅ `delete_folder_files()` - Delete all files in folder
- ✅ `download_folder_as_zip()` - Download folder as ZIP

### API Endpoints (`app/api/datasets.py`)
- ✅ Updated download endpoints - Support filename parameter
- ✅ Updated delete endpoints - Handle folders properly
- ✅ Added list files endpoint - Show files in dataset

### Features
- ✅ Download specific file from folder by name
- ✅ Download entire folder as ZIP
- ✅ Delete folder with all files
- ✅ List all files in folder
- ✅ Works with versioning
- ✅ Backward compatible with single files

---

## 🎉 Summary

**Download Operations:**
- ✅ Single file download
- ✅ Folder download as ZIP
- ✅ Specific file download from folder
- ✅ Version-specific downloads
- ✅ List files in folder

**Delete Operations:**
- ✅ Delete single file
- ✅ Delete folder with all files
- ✅ Delete all versions
- ✅ Delete specific version
- ✅ Cascade delete (files + metadata)

The dataset API now has full feature parity with AI models API! 🚀

