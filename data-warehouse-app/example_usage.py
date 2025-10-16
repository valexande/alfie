#!/usr/bin/env python3
"""
Example usage of the Data Warehouse API
"""
import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def create_user(user_data):
    """Create a new user"""
    response = requests.post(f"{BASE_URL}/users/", json=user_data)
    if response.status_code == 200:
        print(f"✅ User created: {response.json()}")
    else:
        print(f"❌ Failed to create user: {response.text}")
    return response

def upload_dataset(file_path, user_id, dataset_id, name, description=None, version="v1", tags=None):
    """Upload a dataset file"""
    files = {"file": open(file_path, "rb")}
    data = {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "name": name,
        "version": version
    }
    
    if description:
        data["description"] = description
    if tags:
        data["tags"] = ",".join(tags)
    
    response = requests.post(f"{BASE_URL}/datasets/upload", files=files, data=data)
    files["file"].close()
    
    if response.status_code == 200:
        print(f"✅ Dataset uploaded: {response.json()['dataset_id']}")
    else:
        print(f"❌ Failed to upload dataset: {response.text}")
    return response

def get_user_datasets(user_id):
    """Get all datasets for a user"""
    response = requests.get(f"{BASE_URL}/datasets/{user_id}")
    if response.status_code == 200:
        datasets = response.json()
        print(f"📁 Found {len(datasets)} datasets for user {user_id}")
        for dataset in datasets:
            print(f"  - {dataset['name']} ({dataset['dataset_id']}) - {dataset['file_type']}")
    else:
        print(f"❌ Failed to get datasets: {response.text}")
    return response

def search_datasets(user_id, query=None, tags=None, file_type=None):
    """Search datasets"""
    params = {}
    if query:
        params["query"] = query
    if tags:
        params["tags"] = ",".join(tags)
    if file_type:
        params["file_type"] = file_type
    
    response = requests.get(f"{BASE_URL}/datasets/search/{user_id}", params=params)
    if response.status_code == 200:
        datasets = response.json()
        print(f"🔍 Search found {len(datasets)} datasets")
        for dataset in datasets:
            print(f"  - {dataset['name']} ({dataset['dataset_id']})")
    else:
        print(f"❌ Failed to search datasets: {response.text}")
    return response

def main():
    """Example usage workflow"""
    print("🚀 Data Warehouse API Example Usage\n")
    
    # 1. Create a user
    print("1. Creating user...")
    user_data = {
        "user_id": "user1",
        "username": "john_doe",
        "email": "john@example.com",
        "full_name": "John Doe",
        "password": "secure_password"
    }
    create_user(user_data)
    print()
    
    # 2. Create a sample CSV file for demonstration
    print("2. Creating sample data file...")
    sample_csv = """name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago"""
    
    sample_file = Path("sample_data.csv")
    with open(sample_file, "w") as f:
        f.write(sample_csv)
    print(f"✅ Created sample file: {sample_file}")
    print()
    
    # 3. Upload dataset
    print("3. Uploading dataset...")
    upload_dataset(
        file_path=sample_file,
        user_id="user1",
        dataset_id="sample_dataset",
        name="Sample User Data",
        description="Sample dataset with user information",
        version="v1",
        tags=["sample", "users", "demo"]
    )
    print()
    
    # 4. Get user datasets
    print("4. Retrieving user datasets...")
    get_user_datasets("user1")
    print()
    
    # 5. Search datasets
    print("5. Searching datasets...")
    search_datasets("user1", query="sample", tags=["demo"])
    print()
    
    # 6. Get specific dataset
    print("6. Getting specific dataset...")
    response = requests.get(f"{BASE_URL}/datasets/user1/sample_dataset")
    if response.status_code == 200:
        dataset = response.json()
        print(f"📊 Dataset Details:")
        print(f"  Name: {dataset['name']}")
        print(f"  File Type: {dataset['file_type']}")
        print(f"  Size: {dataset['file_size']} bytes")
        print(f"  Columns: {dataset['columns']}")
        print(f"  Rows: {dataset['row_count']}")
        print(f"  Tags: {dataset['tags']}")
    print()
    
    # 7. Download dataset
    print("7. Downloading dataset...")
    response = requests.get(f"{BASE_URL}/datasets/user1/sample_dataset/download")
    if response.status_code == 200:
        with open("downloaded_sample.csv", "wb") as f:
            f.write(response.content)
        print("✅ Dataset downloaded as 'downloaded_sample.csv'")
    print()
    
    # Cleanup
    print("🧹 Cleaning up...")
    sample_file.unlink(missing_ok=True)
    Path("downloaded_sample.csv").unlink(missing_ok=True)
    print("✅ Cleanup completed")

if __name__ == "__main__":
    main()
