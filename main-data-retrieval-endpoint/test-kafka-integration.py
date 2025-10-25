#!/usr/bin/env python3
"""
Test script for Kafka consumer integration with Flask endpoints
"""

import os
import sys
import tempfile
import shutil

# Add the scripts directory to the path
sys.path.append('scripts')

from kafka_call import call_model_explanation_endpoint, call_driver_data_endpoint

def test_flask_integration():
    """Test the Kafka consumer functions with Flask endpoints"""
    print("Testing Kafka consumer integration with Flask endpoints...")
    
    # Create temporary directories with test data
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        model_dir = os.path.join(temp_dir, 'model')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy test files
        test_files = [
            ('../uc-2-data-explanation/csv-json/alert-data.csv', data_dir),
            ('../uc-2-data-explanation/csv-json/frames-cleaned.csv', data_dir),
            ('../uc-2-data-explanation/csv-json/heart_rate.csv', data_dir),
            ('csv-pkl-json/model.pkl', model_dir),
            ('csv-pkl-json/label_encoders.pkl', model_dir)
        ]
        
        for src, dst in test_files:
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"Warning: {src} not found")
        
        try:
            # Test model explanation endpoint
            print("\nTesting model explanation endpoint...")
            model_html = call_model_explanation_endpoint(data_dir, model_dir, user_level='expert')
            print(f"Model explanation: {len(model_html)} characters")
            
            # Test driver data endpoint
            print("\nTesting driver data endpoint...")
            driver_html = call_driver_data_endpoint(data_dir, user_level='expert')
            print(f"Driver data analysis: {len(driver_html)} characters")
            
            print("\nAll tests passed!")
            return True
            
        except Exception as e:
            print(f"\nTest failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("Testing Kafka Consumer Integration")
    print("=" * 50)
    
    success = test_flask_integration()
    
    if success:
        print("\nIntegration test completed successfully!")
        print("The Kafka consumer can successfully call the Flask endpoints.")
    else:
        print("\nIntegration test failed.")
        print("Check the Flask container is running and accessible.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
