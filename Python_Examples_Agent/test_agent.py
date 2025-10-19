"""
Basic tests for the agent implementation.

Note: These tests validate the structure and basic functionality.
Full integration tests require a valid Anthropic API key.
"""

import unittest
from unittest.mock import patch, MagicMock
import os


class TestAgentStructure(unittest.TestCase):
    """Test the basic structure and components of the agent."""
    
    def test_imports(self):
        """Test that the agent module can be imported."""
        try:
            import agent
            self.assertTrue(hasattr(agent, 'create_agent'))
            self.assertTrue(hasattr(agent, 'run_agent'))
            # Tools are now decorated functions - they have invoke method
            self.assertTrue(hasattr(agent.get_current_time, 'invoke'))
            self.assertTrue(hasattr(agent.calculate, 'invoke'))
        except ImportError as e:
            self.fail(f"Failed to import agent module: {e}")
    
    def test_get_current_time(self):
        """Test the get_current_time tool."""
        from agent import get_current_time
        # Tools are decorated with @tool, so we invoke them
        result = get_current_time.invoke({})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_calculate_valid_expression(self):
        """Test the calculator tool with valid expressions."""
        from agent import calculate
        
        test_cases = [
            ("2 + 2", "4"),
            ("10 * 5", "50"),
            ("100 / 4", "25.0"),
        ]
        
        for expression, expected in test_cases:
            result = calculate.invoke({"expression": expression})
            self.assertEqual(result, expected)
    
    def test_calculate_invalid_expression(self):
        """Test the calculator tool with invalid expressions."""
        from agent import calculate
        result = calculate.invoke({"expression": "invalid expression"})
        self.assertTrue(result.startswith("Error:"))
    
    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        env_example_path = os.path.join(test_dir, ".env.example")
        self.assertTrue(os.path.exists(env_example_path))
    
    def test_requirements_exists(self):
        """Test that requirements.txt exists and contains necessary packages."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        requirements_path = os.path.join(test_dir, "requirements.txt")
        self.assertTrue(os.path.exists(requirements_path))
        
        with open(requirements_path, 'r') as f:
            content = f.read()
            self.assertIn("langchain", content)
            self.assertIn("anthropic", content)
            self.assertIn("dotenv", content)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_agent_without_api_key(self):
        """Test that create_agent raises error without API key."""
        from agent import create_agent
        
        with self.assertRaises(ValueError) as context:
            create_agent()
        
        self.assertIn("ANTHROPIC_API_KEY", str(context.exception))


if __name__ == "__main__":
    unittest.main()
