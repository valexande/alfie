#!/usr/bin/env python3
"""
Debug test for UC-2 Flask API
"""

import requests
import os

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=10)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_simple_model_request():
    """Test with minimal data"""
    print("\nTesting with minimal data...")
    
    # Create a minimal test CSV
    test_csv = """heart_rate,yawning,looks_straigh,eyes_closed,alert,gender,age,ethnicity,race
75,0,1,0,0,Male,25,Non-Hispanic,White
80,1,0,1,1,Female,30,Hispanic,Black"""
    
    with open("test_data.csv", "w") as f:
        f.write(test_csv)
    
    try:
        with open("csv-pkl-json/model.pkl", 'rb') as mf, open("csv-pkl-json/label_encoders.pkl", 'rb') as ef, open("test_data.csv", 'rb') as df:
            files = {
                'model_file': mf,
                'encoder_file': ef,
                'data_file': df
            }
            data = {'user_level': 'expert'}
            
            print("Sending minimal request...")
            response = requests.post("http://localhost:5000/explain-uc2-model", files=files, data=data, timeout=60)
            
            print(f"Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response: {response.text[:500]}")
            else:
                print("Success!")
                print(f"Response length: {len(response.text)}")
            
            return response.status_code == 200
    except Exception as e:
        print(f"Request failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")

def main():
    print("Debug Test for UC-2 Flask API")
    print("=" * 40)
    
    # Test health first
    if not test_health():
        print("Health check failed - is the Flask app running?")
        return 1
    
    # Test with minimal data
    success = test_simple_model_request()
    
    if success:
        print("\nDebug test passed!")
    else:
        print("\nDebug test failed - check Flask container logs")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
