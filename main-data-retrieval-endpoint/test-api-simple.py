#!/usr/bin/env python3
"""
Simple test script for UC-2 Flask API
"""

import requests
import os

def test_model_explanation():
    """Test the model explanation endpoint"""
    print("Testing model explanation endpoint...")
    
    # Check if required files exist
    model_file = "csv-pkl-json/model.pkl"
    encoder_file = "csv-pkl-json/label_encoders.pkl"
    data_file = "../uc-2-data-explanation/csv-json/alert-data-uc2-demographics.csv"
    
    if not all(os.path.exists(f) for f in [model_file, encoder_file, data_file]):
        print("Required files not found:")
        for f in [model_file, encoder_file, data_file]:
            print(f"   - {f}: {'EXISTS' if os.path.exists(f) else 'MISSING'}")
        return False
    
    try:
        with open(model_file, 'rb') as mf, open(encoder_file, 'rb') as ef, open(data_file, 'rb') as df:
            files = {
                'model_file': mf,
                'encoder_file': ef,
                'data_file': df
            }
            data = {'user_level': 'expert'}
            
            print("Sending request to Flask API...")
            response = requests.post("http://localhost:5000/explain-uc2-model", files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                print("Model explanation test passed!")
                print(f"   Response length: {len(response.text)} characters")
                print(f"   Content type: {response.headers.get('content-type', 'unknown')}")
                
                # Save the response to a file for inspection
                with open("output/model_explanation_response.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("   Response saved to: output/model_explanation_response.html")
                return True
            else:
                print(f"Model explanation test failed: {response.status_code}")
                print(f"   Response: {response.text[:500]}...")
                return False
    except Exception as e:
        print(f"Model explanation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Testing UC-2 Flask API")
    print("=" * 50)
    
    # Test model explanation
    success = test_model_explanation()
    
    print("\n" + "=" * 50)
    if success:
        print("Test completed successfully!")
        print("Check output/model_explanation_response.html for the generated report")
    else:
        print("Test failed. Check the logs above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
