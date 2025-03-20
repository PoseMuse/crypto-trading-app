import ccxt
import numpy as np
import pandas as pd

def test_environment_imports():
    # Test ccxt is accessible
    assert hasattr(ccxt, 'Exchange'), "ccxt doesn't have expected attribute"
    
    # Test numpy is accessible
    assert np.__version__, "NumPy is not accessible"
    
    # Test pandas is accessible
    assert pd.__version__, "Pandas is not accessible"
    
    print("All required packages are installed and accessible!") 