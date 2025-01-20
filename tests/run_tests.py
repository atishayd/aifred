import unittest
import sys
import os

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

if __name__ == '__main__':
    result = run_tests()
    sys.exit(not result.wasSuccessful()) 