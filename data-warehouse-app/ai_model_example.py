#!/usr/bin/env python3
"""
AI Model Management Example

This script demonstrates how to use the AI Model endpoints for:
1. Uploading single model files (PKL, ONNX, H5, etc.)
2. Uploading model folders as ZIP files
3. Downloading model files and folders
4. Managing model metadata
5. Searching and filtering models

Usage:
    python ai_model_example.py
"""

import os
import requests
import json
import tempfile
import zipfile
from pathlib import Path

# Configuration
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
USER_ID = "user1"
MODEL_ID = "my_classification_model"

def create_sample_model_files():
    """Create sample model files for demonstration"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create sample model files
    model_files = {
        "model.pkl": b"dummy_pickle_data",
        "weights.h5": b"dummy_h5_data", 
        "config.json": json.dumps({"model_type": "classification", "classes": 10}).encode(),
        "requirements.txt": b"numpy==1.21.0\nscikit-learn==1.0.0\npandas==1.3.0",
        "README.md": b"# My Model\nThis is a sample classification model."
    }
    
    for filename, content in model_files.items():
        file_path = temp_dir / filename
        file_path.write_bytes(content)
    
    return temp_dir

def create_model_folder_zip(temp_dir):
    """Create a ZIP file from the model folder"""
    zip_path = temp_dir.parent / "model_folder.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    return zip_path

def upload_single_model_file():
    """Example: Upload a single model file"""
    print("=== Uploading Single Model File ===")
    
    # Create a sample PKL file
    temp_dir = Path(tempfile.mkdtemp())
    model_file = temp_dir / "model.pkl"
    model_file.write_bytes(b"dummy_pickle_model_data")
    
    url = f"{API_BASE}/ai-models/upload/single"
    
    with open(model_file, 'rb') as f:
        files = {'file': ('model.pkl', f, 'application/octet-stream')}
        data = {
            'user_id': USER_ID,
            'model_id': f"{MODEL_ID}_single",
            'name': 'Single Model Upload Example',
            'description': 'Example of uploading a single PKL model file',
            'version': 'v1',
            'framework': 'sklearn',
            'model_type': 'classification',
            'algorithm': 'RandomForest',
            'is_primary': True,
            'tags': 'example,pkl,classification',
            'training_accuracy': 0.95,
            'validation_accuracy': 0.92,
            'test_accuracy': 0.90,
            'python_version': '3.9'
        }
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model uploaded successfully!")
            print(f"   Model ID: {result['model_id']}")
            print(f"   Files: {len(result['files'])}")
            print(f"   Size: {result['model_size_mb']:.2f} MB")
            return result['model_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

def upload_model_folder():
    """Example: Upload a model folder as ZIP"""
    print("\n=== Uploading Model Folder ===")
    
    # Create sample model files
    temp_dir = create_sample_model_files()
    
    # Create ZIP file
    zip_path = create_model_folder_zip(temp_dir)
    
    url = f"{API_BASE}/ai-models/upload/folder"
    
    with open(zip_path, 'rb') as f:
        files = {'zip_file': ('model_folder.zip', f, 'application/zip')}
        data = {
            'user_id': USER_ID,
            'model_id': f"{MODEL_ID}_folder",
            'name': 'Model Folder Upload Example',
            'description': 'Example of uploading a model folder with multiple files',
            'version': 'v1',
            'framework': 'sklearn',
            'model_type': 'classification',
            'algorithm': 'RandomForest',
            'preserve_structure': True,
            'tags': 'example,folder,multi-file',
            'training_accuracy': 0.96,
            'validation_accuracy': 0.93,
            'test_accuracy': 0.91,
            'python_version': '3.9'
        }
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model folder uploaded successfully!")
            print(f"   Model ID: {result['model_id']}")
            print(f"   Files: {len(result['files'])}")
            print(f"   Size: {result['model_size_mb']:.2f} MB")
            print("   File list:")
            for file_info in result['files']:
                print(f"     - {file_info['filename']} ({file_info['file_size']} bytes)")
            return result['model_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

def get_model_metadata(model_id):
    """Example: Get model metadata"""
    print(f"\n=== Getting Model Metadata: {model_id} ===")
    
    url = f"{API_BASE}/ai-models/{USER_ID}/{model_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        model = response.json()
        print(f"✅ Model metadata retrieved!")
        print(f"   Name: {model['name']}")
        print(f"   Framework: {model['framework']}")
        print(f"   Type: {model['model_type']}")
        print(f"   Version: {model['version']}")
        print(f"   Files: {len(model['files'])}")
        print(f"   Tags: {model['tags']}")
        print(f"   Training Accuracy: {model.get('training_accuracy', 'N/A')}")
        return model
    else:
        print(f"❌ Failed to get model metadata: {response.status_code}")
        return None

def list_model_files(model_id):
    """Example: List files available for a model"""
    print(f"\n=== Listing Files for Model: {model_id} ===")
    
    url = f"{API_BASE}/ai-models/{USER_ID}/{model_id}/files"
    response = requests.get(url)
    
    if response.status_code == 200:
        files = response.json()
        print(f"✅ Found {len(files)} files:")
        for file_info in files:
            print(f"   - {file_info['filename']} ({file_info['file_size']} bytes)")
            if file_info.get('is_primary'):
                print(f"     * Primary file")
        return files
    else:
        print(f"❌ Failed to list files: {response.status_code}")
        return []

def download_model_file(model_id, filename=None):
    """Example: Download model file or entire model as ZIP"""
    print(f"\n=== Downloading Model: {model_id} ===")
    
    if filename:
        # Download specific file
        url = f"{API_BASE}/ai-models/{USER_ID}/{model_id}/download"
        params = {'filename': filename}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            print(f"✅ Downloaded file: {filename}")
            print(f"   Size: {len(response.content)} bytes")
        else:
            print(f"❌ Failed to download file: {response.status_code}")
    else:
        # Download entire model as ZIP
        url = f"{API_BASE}/ai-models/{USER_ID}/{model_id}/download"
        response = requests.get(url)
        
        if response.status_code == 200:
            zip_filename = f"{model_id}_downloaded.zip"
            with open(zip_filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded model as ZIP: {zip_filename}")
            print(f"   Size: {len(response.content)} bytes")
        else:
            print(f"❌ Failed to download model: {response.status_code}")

def list_user_models():
    """Example: List all models for a user"""
    print(f"\n=== Listing Models for User: {USER_ID} ===")
    
    url = f"{API_BASE}/ai-models/{USER_ID}"
    response = requests.get(url)
    
    if response.status_code == 200:
        models = response.json()
        print(f"✅ Found {len(models)} models:")
        for model in models:
            print(f"   - {model['model_id']} ({model['framework']}, {model['model_type']}) - {model['name']}")
        return models
    else:
        print(f"❌ Failed to list models: {response.status_code}")
        return []

def search_models():
    """Example: Search models with filters"""
    print(f"\n=== Searching Models ===")
    
    url = f"{API_BASE}/ai-models/search/{USER_ID}"
    params = {
        'query': 'classification',
        'framework': 'sklearn',
        'tags': 'example'
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        models = response.json()
        print(f"✅ Found {len(models)} models matching search criteria:")
        for model in models:
            print(f"   - {model['model_id']}: {model['name']}")
        return models
    else:
        print(f"❌ Search failed: {response.status_code}")
        return []

def update_model(model_id):
    """Example: Update model metadata"""
    print(f"\n=== Updating Model: {model_id} ===")
    
    url = f"{API_BASE}/ai-models/{USER_ID}/{model_id}"
    data = {
        'name': 'Updated Model Name',
        'description': 'Updated description with new information',
        'tags': ['updated', 'example', 'classification'],
        'is_production_ready': True,
        'test_accuracy': 0.94
    }
    
    response = requests.put(url, json=data)
    
    if response.status_code == 200:
        updated_model = response.json()
        print(f"✅ Model updated successfully!")
        print(f"   New name: {updated_model['name']}")
        print(f"   Production ready: {updated_model['is_production_ready']}")
        return updated_model
    else:
        print(f"❌ Update failed: {response.status_code}")
        return None

def main():
    """Main function to run all examples"""
    print("🚀 AI Model Management API Examples")
    print("=" * 50)
    
    try:
        # Test API connection
        response = requests.get(f"{API_BASE}/health")
        if response.status_code != 200:
            print(f"❌ API not accessible at {API_BASE}")
            return
        
        print(f"✅ Connected to API at {API_BASE}")
        
        # Upload examples
        single_model_id = upload_single_model_file()
        folder_model_id = upload_model_folder()
        
        if single_model_id:
            # Get metadata
            get_model_metadata(single_model_id)
            
            # List files
            list_model_files(single_model_id)
            
            # Download examples
            download_model_file(single_model_id, "model.pkl")
            download_model_file(single_model_id)
            
            # Update model
            update_model(single_model_id)
        
        if folder_model_id:
            # Get metadata
            get_model_metadata(folder_model_id)
            
            # List files
            files = list_model_files(folder_model_id)
            
            # Download examples (download specific files from the folder)
            if files:
                # Download the first file as an example
                first_file = files[0]['filename']
                download_model_file(folder_model_id, first_file)
            
            # Download entire model as ZIP
            download_model_file(folder_model_id)
        
        # List and search
        list_user_models()
        search_models()
        
        print("\n🎉 All examples completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to API at {API_BASE}")
        print("   Make sure the API server is running")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
