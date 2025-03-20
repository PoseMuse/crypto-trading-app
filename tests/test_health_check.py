import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from src.health_check_endpoint import HealthCheckHandler

class TestHealthCheck(unittest.TestCase):
    """Test cases for health check endpoint."""

    def test_build_response(self):
        """Test that the health check response contains expected fields."""
        # Create a mock handler
        handler = MagicMock()
        # Add the method to test
        handler._build_response = HealthCheckHandler._build_response.__get__(handler, HealthCheckHandler)
        
        # Call the method
        response = handler._build_response()
        
        # Verify structure of response
        self.assertIsInstance(response, dict)
        self.assertIn('status', response)
        self.assertEqual(response['status'], 'ok')
        self.assertIn('version', response)
        self.assertIn('uptime_seconds', response)
        self.assertIn('hostname', response)
        self.assertIn('timestamp', response)
        
    @patch('sys.stdout', new_callable=StringIO)
    def test_handler_methods_exist(self, mock_stdout):
        """Test that the handler has the expected methods."""
        # Verify the handler has the expected methods
        self.assertTrue(hasattr(HealthCheckHandler, 'do_GET'))
        self.assertTrue(hasattr(HealthCheckHandler, '_set_headers'))
        self.assertTrue(hasattr(HealthCheckHandler, '_build_response'))

if __name__ == '__main__':
    unittest.main() 