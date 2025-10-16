#!/usr/bin/env python3
"""
Test script for dataset folder upload, download, and delete operations

This script demonstrates:
1. Creating a test ZIP file
2. Uploading folder to DW
3. Listing files in folder
4. Downloading specific files
5. Downloading entire folder as ZIP
6. Deleting the dataset

Usage:
  python test_dataset_folder_operations.py
"""

import requests
import zipfile
import os
import json
import tempfile

API_BASE = "http://localhost:8000"
USER_ID = "test_user"
DATASET_ID = "folder-ops-test"


def create_test_files():
    """Create test files for upload"""
    print("📁 Creating test files...")
    
    # Create test data files
    with open("test_data1.csv", "w") as f:
        f.write("id,name,value\n1,Alice,100\n2,Bob,200\n")
    
    with open("test_data2.csv", "w") as f:
        f.write("id,category,score\n1,A,95\n2,B,87\n")
    
    with open("test_metadata.json", "w") as f:
        json.dump({
            "description": "Test dataset",
            "created": "2025-10-10",
            "files": ["test_data1.csv", "test_data2.csv"]
        }, f)
    
    # Create ZIP
    with zipfile.ZipFile("test_dataset.zip", "w") as zipf:
        zipf.write("test_data1.csv")
        zipf.write("test_data2.csv")
        zipf.write("test_metadata.json")
    
    print("✅ Created test files and ZIP")
    return ["test_data1.csv", "test_data2.csv", "test_metadata.json", "test_dataset.zip"]


def cleanup_test_files(files):
    """Remove test files"""
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def test_upload_folder():
    """Test folder upload"""
    print("\n" + "=" * 60)
    print("1️⃣ Testing Folder Upload")
    print("=" * 60)
    
    with open("test_dataset.zip", "rb") as f:
        files = {'zip_file': ('test_dataset.zip', f, 'application/zip')}
        data = {
            'dataset_id': DATASET_ID,
            'name': 'Folder Operations Test',
            'description': 'Testing folder upload, download, and delete',
            'preserve_structure': 'true',
            'tags': 'test,folder'
        }
        
        response = requests.post(
            f"{API_BASE}/datasets/upload/folder/{USER_ID}",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Folder uploaded successfully!")
            print(f"   Version: {result['version']}")
            print(f"   Is folder: {result['is_folder']}")
            print(f"   Total size: {result['file_size']} bytes")
            print(f"   File count: {result['custom_metadata']['file_count']}")
            return result
        else:
            print(f"❌ Upload failed: {response.text}")
            return None


def test_list_files():
    """Test listing files in folder"""
    print("\n" + "=" * 60)
    print("2️⃣ Testing List Files")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/datasets/{USER_ID}/{DATASET_ID}/files")
    
    if response.status_code == 200:
        files = response.json()
        print(f"✅ Found {len(files)} files:")
        for f in files:
            print(f"   - {f['filename']} ({f['file_size']} bytes, {f['file_type']})")
        return files
    else:
        print(f"❌ List failed: {response.text}")
        return []


def test_download_specific_file(filename):
    """Test downloading specific file from folder"""
    print("\n" + "=" * 60)
    print(f"3️⃣ Testing Download Specific File: {filename}")
    print("=" * 60)
    
    response = requests.get(
        f"{API_BASE}/datasets/{USER_ID}/{DATASET_ID}/download",
        params={"filename": filename}
    )
    
    if response.status_code == 200:
        output_file = f"downloaded_{filename}"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded {filename} ({len(response.content)} bytes)")
        print(f"   Saved as: {output_file}")
        return output_file
    else:
        print(f"❌ Download failed: {response.text}")
        return None


def test_download_folder_as_zip():
    """Test downloading entire folder as ZIP"""
    print("\n" + "=" * 60)
    print("4️⃣ Testing Download Folder as ZIP")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/datasets/{USER_ID}/{DATASET_ID}/download")
    
    if response.status_code == 200:
        with open("downloaded_folder.zip", "wb") as f:
            f.write(response.content)
        
        # Verify ZIP contents
        with zipfile.ZipFile("downloaded_folder.zip", "r") as zipf:
            file_list = zipf.namelist()
            print(f"✅ Downloaded folder as ZIP ({len(response.content)} bytes)")
            print(f"   Files in ZIP: {len(file_list)}")
            for fname in file_list:
                print(f"     - {fname}")
        
        return "downloaded_folder.zip"
    else:
        print(f"❌ Download failed: {response.text}")
        return None


def test_delete_dataset():
    """Test deleting dataset"""
    print("\n" + "=" * 60)
    print("5️⃣ Testing Delete Dataset")
    print("=" * 60)
    
    response = requests.delete(f"{API_BASE}/datasets/{USER_ID}/{DATASET_ID}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Deleted successfully!")
        print(f"   Versions deleted: {result['versions_deleted']}")
        print(f"   Files deleted: {result['files_deleted']}")
        
        # Verify deletion
        response = requests.get(f"{API_BASE}/datasets/{USER_ID}/{DATASET_ID}")
        if response.status_code == 404:
            print("✅ Verified: Dataset no longer exists")
        else:
            print("⚠️  Warning: Dataset still exists after deletion")
    else:
        print(f"❌ Delete failed: {response.text}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 Dataset Folder Operations Test Suite")
    print("=" * 60)
    
    # Create test files
    test_files = create_test_files()
    
    try:
        # Run tests
        upload_result = test_upload_folder()
        
        if upload_result:
            files = test_list_files()
            
            if files:
                # Download first file
                test_download_specific_file(files[0]['filename'])
            
            test_download_folder_as_zip()
            test_delete_dataset()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup test files
        print("\n🧹 Cleaning up test files...")
        cleanup_test_files(test_files)
        cleanup_test_files([
            "downloaded_test_data1.csv",
            "downloaded_test_data2.csv",
            "downloaded_test_metadata.json",
            "downloaded_folder.zip"
        ])
        print("✅ Cleanup complete")


if __name__ == "__main__":
    main()

