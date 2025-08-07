"""
Test basic ML packages (without chemistry packages)
"""

def test_basic_imports():
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib.pyplot as plt
        print("[OK] Pandas working")
        print("[OK] NumPy working") 
        print("[OK] Scikit-learn working")
        print("[OK] Matplotlib working")
        
        print("SUCCESS: Basic ML packages ready!")
        return True
        
    except ImportError as e:
        print(f"ERROR: {e}")
        return False

def test_advanced_imports():
    try:
        import fastapi
        print("[OK] FastAPI working")
        
        import mlflow
        print("[OK] MLflow working")
        
        print("SUCCESS: Advanced packages ready!")
        
    except ImportError as e:
        print(f"INFO: Advanced package not installed yet: {e}")

if __name__ == "__main__":
    print("=== Testing Basic ML Packages ===")
    basic_ok = test_basic_imports()
    
    print("\n=== Testing Advanced Packages ===")
    test_advanced_imports()
    
    if basic_ok:
        print("\n Ready to start ML development!")