"""
Test script for Flask API
Tests all endpoints with sample requests
"""

import requests
import json
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    

def test_predict(image_path):
    """Test prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Prediction Endpoint")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/api/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Flood Detected: {result['prediction']['flood_detected']}")
        print(f"  Flood Percentage: {result['prediction']['flood_percentage']:.2f}%")
        print(f"  Risk Level: {result['prediction']['risk_level']}")
        print(f"  Max Confidence: {result['prediction']['max_confidence']:.3f}")
        print(f"  Avg Confidence: {result['prediction']['avg_confidence']:.3f}")
    else:
        print(f"Error: {response.json()}")


def test_predict_coordinates(image_path, bounds="40.7128,-74.0060,40.7828,-73.9260"):
    """Test coordinate prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Coordinate Prediction Endpoint")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'bounds': bounds}
        response = requests.post(f"{BASE_URL}/api/predict-coordinates", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Flood Detected: {result['prediction']['flood_detected']}")
        print(f"  Flood Percentage: {result['prediction']['flood_percentage']:.2f}%")
        print(f"  Risk Level: {result['prediction']['risk_level']}")
        print(f"  Number of Flood Zones: {result['metadata']['num_flood_zones']}")
    else:
        print(f"Error: {response.json()}")


def test_batch_predict(image_paths):
    """Test batch prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Batch Prediction Endpoint")
    print("="*50)
    
    files = []
    for path in image_paths:
        if Path(path).exists():
            files.append(('images', open(path, 'rb')))
        else:
            print(f"Warning: Image not found at {path}")
    
    if not files:
        print("Error: No valid images found")
        return
    
    response = requests.post(f"{BASE_URL}/api/batch-predict", files=files)
    
    # Close files
    for _, f in files:
        f.close()
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Results:")
        print(f"  Total Images: {result['total_images']}")
        for r in result['results']:
            if r['success']:
                print(f"\n  {r['filename']}:")
                print(f"    Flood: {r['flood_detected']}")
                print(f"    Percentage: {r['flood_percentage']:.2f}%")
                print(f"    Risk: {r['risk_level']}")
            else:
                print(f"\n  {r['filename']}: ERROR - {r['error']}")
    else:
        print(f"Error: {response.json()}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Flood Segmentation API Test Suite")
    print("="*50)
    
    # Test health endpoint
    try:
        test_health()
    except Exception as e:
        print(f"Health test failed: {e}")
    
    # Test prediction with sample image
    # Update this path to your test image
    test_image = "test_image.jpg"
    
    print("\n\nNote: To test prediction endpoints, provide a valid image path:")
    print(f"  test_predict('{test_image}')")
    print(f"  test_predict_coordinates('{test_image}')")
    print(f"  test_batch_predict(['{test_image}', 'test_image2.jpg'])")
    
    print("\n\nExample usage:")
    print("  python test_api.py")
