#!/usr/bin/env python3
"""
Test script for UC-2 Flask API
Tests the health endpoint and demonstrates API usage
"""

import requests
import json
import os
from pathlib import Path

def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_explanation(base_url, user_level="expert"):
    """Test the model explanation endpoint"""
    print(f"🔍 Testing model explanation endpoint (level: {user_level})...")
    
    # Check if required files exist
    model_file = "csv-pkl-json/model.pkl"
    encoder_file = "csv-pkl-json/label_encoders.pkl"
    data_file = "csv-pkl-json/alert-data-uc2-demographics.csv"
    
    if not all(os.path.exists(f) for f in [model_file, encoder_file, data_file]):
        print("❌ Required files not found. Please ensure the following files exist:")
        print(f"   - {model_file}")
        print(f"   - {encoder_file}")
        print(f"   - {data_file}")
        return False
    
    try:
        with open(model_file, 'rb') as mf, open(encoder_file, 'rb') as ef, open(data_file, 'rb') as df:
            files = {
                'model_file': mf,
                'encoder_file': ef,
                'data_file': df
            }
            data = {'user_level': user_level}
            
            response = requests.post(f"{base_url}/explain-uc2-model", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                print("✅ Model explanation test passed!")
                print(f"   Response length: {len(response.text)} characters")
                return True
            else:
                print(f"❌ Model explanation test failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False
    except Exception as e:
        print(f"❌ Model explanation test failed: {e}")
        return False

def test_driver_analysis(base_url, user_level="expert"):
    """Test the driver analysis endpoint"""
    print(f"🔍 Testing driver analysis endpoint (level: {user_level})...")
    
    # Check if required files exist
    frame_file = "csv-pkl-json/frames-cleaned.csv"
    hr_file = "csv-pkl-json/heart_rate.csv"
    
    if not all(os.path.exists(f) for f in [frame_file, hr_file]):
        print("❌ Required files not found. Please ensure the following files exist:")
        print(f"   - {frame_file}")
        print(f"   - {hr_file}")
        return False
    
    try:
        with open(frame_file, 'rb') as ff, open(hr_file, 'rb') as hf:
            files = {
                'frame_file': ff,
                'hr_file': hf
            }
            data = {'user_level': user_level}
            
            response = requests.post(f"{base_url}/explain-uc2-data", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                print("✅ Driver analysis test passed!")
                print(f"   Response length: {len(response.text)} characters")
                return True
            else:
                print(f"❌ Driver analysis test failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False
    except Exception as e:
        print(f"❌ Driver analysis test failed: {e}")
        return False

def main():
    """Main test function"""
    base_url = "http://localhost:5000"
    
    print("🚀 Starting UC-2 Flask API Tests")
    print("=" * 50)
    
    # Test health endpoint
    health_ok = test_health_endpoint(base_url)
    
    if not health_ok:
        print("\n❌ Health check failed. Is the Flask app running?")
        print("   Try: docker-compose up uc2-flask-app")
        return 1
    
    print("\n" + "=" * 50)
    
    # Test model explanation
    model_ok = test_model_explanation(base_url, "expert")
    
    print("\n" + "=" * 50)
    
    # Test driver analysis
    driver_ok = test_driver_analysis(base_url, "expert")
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Model Explanation: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"   Driver Analysis: {'✅ PASS' if driver_ok else '❌ FAIL'}")
    
    if all([health_ok, model_ok, driver_ok]):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
