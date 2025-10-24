#!/usr/bin/env python3
"""
Test script for the new /explain-uc2-data endpoint
"""

import requests
import os

def test_data_endpoint():
    """Test the driver data analysis endpoint"""
    print("Testing driver data analysis endpoint...")
    
    # Check if required files exist
    frame_file = "../uc-2-data-explanation/csv-json/frames-cleaned.csv"
    hr_file = "../uc-2-data-explanation/csv-json/heart_rate.csv"
    
    if not all(os.path.exists(f) for f in [frame_file, hr_file]):
        print("Required files not found:")
        for f in [frame_file, hr_file]:
            print(f"   - {f}: {'EXISTS' if os.path.exists(f) else 'MISSING'}")
        return False
    
    try:
        with open(frame_file, 'rb') as ff, open(hr_file, 'rb') as hf:
            files = {
                'frame_file': ff,
                'hr_file': hf
            }
            data = {'user_level': 'expert'}
            
            print("Sending request to Flask API...")
            response = requests.post("http://localhost:5000/explain-uc2-data", files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                print("Driver data analysis test passed!")
                print(f"   Response length: {len(response.text)} characters")
                print(f"   Content type: {response.headers.get('content-type', 'unknown')}")
                
                # Save the response to a file for inspection
                with open("output/driver_analysis_response.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("   Response saved to: output/driver_analysis_response.html")
                return True
            else:
                print(f"Driver data analysis test failed: {response.status_code}")
                print(f"   Response: {response.text[:500]}...")
                return False
    except Exception as e:
        print(f"Driver data analysis test failed: {e}")
        return False

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

def main():
    print("Testing UC-2 Flask API - Data Analysis Endpoint")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("Health check failed - is the Flask app running?")
        return 1
    
    # Test data endpoint
    success = test_data_endpoint()
    
    print("\n" + "=" * 60)
    if success:
        print("Test completed successfully!")
        print("Check output/driver_analysis_response.html for the generated report")
    else:
        print("Test failed. Check the logs above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
