#!/usr/bin/env python3
"""
Debug script to test Data Warehouse upload API
"""

import requests
import os

def test_upload_api():
    """Test the Data Warehouse upload API directly"""
    print("Testing Data Warehouse upload API...")
    
    # Test parameters
    user_id = "1"
    dataset_id = "test_dataset"
    model_id = "test_model"
    report_type = "model_explanation"
    level = "expert"
    
    # Create a simple test HTML file
    test_html = """
    <html>
    <head><title>Test Report</title></head>
    <body>
        <h1>Test XAI Report</h1>
        <p>This is a test report for debugging upload issues.</p>
    </body>
    </html>
    """
    
    test_file_path = "test_report.html"
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_html)
    
    try:
        url = f"http://localhost:8000/xai-reports/upload/{user_id}"
        print(f"Upload URL: {url}")
        
        with open(test_file_path, 'rb') as f:
            files = {'file': (test_file_path, f, 'text/html')}
            data = {
                'dataset_id': dataset_id,
                'model_id': model_id,
                'report_type': report_type,
                'level': level
            }
            
            print(f"Request data: {data}")
            print(f"Files: {list(files.keys())}")
            
            response = requests.post(url, files=files, data=data, timeout=30)
            
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response text: {response.text}")
            
            if response.status_code == 200:
                print("Upload successful!")
                return True
            else:
                print(f"Upload failed with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"Upload test failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_api_health():
    """Test if Data Warehouse API is accessible"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Data Warehouse health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Data Warehouse health check failed: {e}")
        return False

def main():
    print("Data Warehouse Upload API Debug Test")
    print("=" * 50)
    
    # Test API health first
    if not test_api_health():
        print("Data Warehouse API is not accessible. Make sure it's running on localhost:8000")
        return 1
    
    # Test upload
    success = test_upload_api()
    
    if success:
        print("\nUpload test completed successfully!")
    else:
        print("\nUpload test failed. Check the Data Warehouse API logs.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
