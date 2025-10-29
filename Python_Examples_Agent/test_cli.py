"""
Tests for the Copilot CLI implementation.

These tests validate the CLI structure and functionality.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import CopilotCLI, main


class TestCopilotCLI(unittest.TestCase):
    """Test the CopilotCLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = CopilotCLI(verbose=False)
    
    def test_initialization(self):
        """Test CLI initialization with default parameters."""
        cli = CopilotCLI()
        self.assertEqual(cli.model, "claude-3-5-sonnet-20241022")
        self.assertEqual(cli.temperature, 0.7)
        self.assertFalse(cli.verbose)
        self.assertIsNone(cli.agent)
        self.assertEqual(cli.session_history, [])
    
    def test_initialization_with_custom_params(self):
        """Test CLI initialization with custom parameters."""
        cli = CopilotCLI(
            model="claude-3-opus-20240229",
            temperature=0.5,
            verbose=True
        )
        self.assertEqual(cli.model, "claude-3-opus-20240229")
        self.assertEqual(cli.temperature, 0.5)
        self.assertTrue(cli.verbose)
    
    @patch('cli.create_agent')
    def test_initialize_agent_success(self, mock_create_agent):
        """Test successful agent initialization."""
        mock_create_agent.return_value = MagicMock()
        result = self.cli.initialize_agent()
        self.assertTrue(result)
        self.assertIsNotNone(self.cli.agent)
    
    @patch('cli.create_agent')
    def test_initialize_agent_failure(self, mock_create_agent):
        """Test agent initialization failure."""
        mock_create_agent.side_effect = Exception("API key not found")
        result = self.cli.initialize_agent()
        self.assertFalse(result)
    
    @patch('cli.create_agent')
    def test_query_success(self, mock_create_agent):
        """Test successful query processing."""
        # Setup mock agent
        mock_agent = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "The current time is 2025-10-29 12:00:00"
        mock_agent.invoke.return_value = {'messages': [mock_message]}
        mock_create_agent.return_value = mock_agent
        
        # Initialize agent
        self.cli.initialize_agent()
        
        response = self.cli.query("What is the current time?")
        
        self.assertEqual(response, "The current time is 2025-10-29 12:00:00")
        self.assertEqual(len(self.cli.session_history), 1)
        self.assertEqual(self.cli.session_history[0]['query'], "What is the current time?")
        mock_agent.invoke.assert_called_once()
    
    def test_query_without_agent_initialization(self):
        """Test query when agent is not initialized."""
        response = self.cli.query("What is the current time?")
        self.assertIn("Agent not initialized", response)
    
    @patch('cli.create_agent')
    def test_query_error_handling(self, mock_create_agent):
        """Test query error handling."""
        # Setup mock agent that raises an exception
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("Network error")
        mock_create_agent.return_value = mock_agent
        
        # Initialize agent
        self.cli.initialize_agent()
        response = self.cli.query("What is the current time?")
        
        self.assertIn("Error processing query", response)
    
    def test_show_history_empty(self):
        """Test showing history when empty."""
        with patch('sys.stdout', new=StringIO()) as fake_output:
            self.cli.show_history()
            output = fake_output.getvalue()
            self.assertIn("No conversation history", output)
    
    def test_show_history_with_entries(self):
        """Test showing history with entries."""
        self.cli.session_history = [
            {
                'timestamp': '2025-10-29T12:00:00',
                'query': 'Test query',
                'response': 'Test response'
            }
        ]
        
        with patch('sys.stdout', new=StringIO()) as fake_output:
            self.cli.show_history()
            output = fake_output.getvalue()
            self.assertIn("Test query", output)
            self.assertIn("Test response", output)
    
    def test_show_help(self):
        """Test showing help information."""
        with patch('sys.stdout', new=StringIO()) as fake_output:
            self.cli.show_help()
            output = fake_output.getvalue()
            self.assertIn("Available Commands", output)
            self.assertIn("help", output)
            self.assertIn("history", output)
            self.assertIn("exit", output)
    
    @patch('cli.create_agent')
    def test_one_shot_mode(self, mock_create_agent):
        """Test one-shot mode."""
        # Setup mock agent
        mock_agent = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "42"
        mock_agent.invoke.return_value = {'messages': [mock_message]}
        mock_create_agent.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as fake_output:
            exit_code = self.cli.one_shot_mode("Calculate 6 * 7")
            output = fake_output.getvalue()
            
            self.assertEqual(exit_code, 0)
            self.assertIn("42", output)


class TestCLIMain(unittest.TestCase):
    """Test the main CLI entry point."""
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('sys.argv', ['cli.py', '--help'])
    def test_help_argument(self):
        """Test --help argument."""
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 0)
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('sys.argv', ['cli.py', '--version'])
    def test_version_argument(self):
        """Test --version argument."""
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 0)
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('sys.argv', ['cli.py'])
    def test_missing_api_key(self):
        """Test behavior when API key is missing."""
        with patch('sys.stderr', new=StringIO()) as fake_stderr:
            exit_code = main()
            error_output = fake_stderr.getvalue()
            
            self.assertEqual(exit_code, 1)
            self.assertIn("ANTHROPIC_API_KEY", error_output)
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('sys.argv', ['cli.py', '--temperature', '1.5'])
    def test_invalid_temperature(self):
        """Test invalid temperature value."""
        with patch('sys.stderr', new=StringIO()) as fake_stderr:
            exit_code = main()
            error_output = fake_stderr.getvalue()
            
            self.assertEqual(exit_code, 1)
            self.assertIn("Temperature must be between", error_output)
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('sys.argv', ['cli.py', 'What is the time?'])
    @patch('cli.CopilotCLI.one_shot_mode')
    def test_one_shot_mode_invocation(self, mock_one_shot):
        """Test that one-shot mode is invoked correctly."""
        mock_one_shot.return_value = 0
        exit_code = main()
        
        self.assertEqual(exit_code, 0)
        mock_one_shot.assert_called_once_with('What is the time?')
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('sys.argv', ['cli.py', '--interactive'])
    @patch('cli.CopilotCLI.interactive_mode')
    def test_interactive_mode_invocation(self, mock_interactive):
        """Test that interactive mode is invoked correctly."""
        mock_interactive.return_value = 0
        exit_code = main()
        
        self.assertEqual(exit_code, 0)
        mock_interactive.assert_called_once()


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI with mocked agent."""
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('cli.create_agent')
    @patch('builtins.input', side_effect=['help', 'exit'])
    def test_interactive_mode_help_command(self, mock_input, mock_create_agent):
        """Test interactive mode with help command."""
        mock_create_agent.return_value = MagicMock()
        
        cli = CopilotCLI()
        
        with patch('sys.stdout', new=StringIO()) as fake_output:
            exit_code = cli.interactive_mode()
            output = fake_output.getvalue()
            
            self.assertEqual(exit_code, 0)
            self.assertIn("Available Commands", output)
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('cli.create_agent')
    @patch('builtins.input', side_effect=['history', 'exit'])
    def test_interactive_mode_history_command(self, mock_input, mock_create_agent):
        """Test interactive mode with history command."""
        mock_create_agent.return_value = MagicMock()
        
        cli = CopilotCLI()
        
        with patch('sys.stdout', new=StringIO()) as fake_output:
            exit_code = cli.interactive_mode()
            output = fake_output.getvalue()
            
            self.assertEqual(exit_code, 0)
            self.assertIn("No conversation history", output)
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('cli.create_agent')
    @patch('builtins.input', side_effect=['quit'])
    def test_interactive_mode_quit_variations(self, mock_input, mock_create_agent):
        """Test interactive mode with different quit commands."""
        mock_create_agent.return_value = MagicMock()
        
        cli = CopilotCLI()
        
        with patch('sys.stdout', new=StringIO()) as fake_output:
            exit_code = cli.interactive_mode()
            output = fake_output.getvalue()
            
            self.assertEqual(exit_code, 0)
            self.assertIn("Goodbye", output)


if __name__ == "__main__":
    unittest.main()
