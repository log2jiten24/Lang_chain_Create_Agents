#!/usr/bin/env python3
"""
Demo script for the Copilot CLI

This script demonstrates the CLI features without requiring an API key.
It shows the help output, version, and other CLI capabilities.
"""

import subprocess
import sys
import os

def run_command(cmd_list, description, cmd_display=None):
    """Run a command and display the output."""
    print("=" * 70)
    print(f"Demo: {description}")
    print("=" * 70)
    # Use cmd_display if provided, otherwise join the list
    display_cmd = cmd_display if cmd_display else ' '.join(cmd_list)
    print(f"$ {display_cmd}\n")
    
    result = subprocess.run(
        cmd_list,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print()
    return result.returncode

def main():
    """Run CLI demonstrations."""
    print("\n")
    print("#" * 70)
    print("# COPILOT CLI DEMONSTRATION")
    print("#" * 70)
    print()
    
    # Change to Python_Examples_Agent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, "Python_Examples_Agent"))
    
    # Demo 1: Version
    run_command(
        ["python", "cli.py", "--version"],
        "Show CLI version",
        "python cli.py --version"
    )
    
    # Demo 2: Help
    run_command(
        ["python", "cli.py", "--help"],
        "Display help information with all options",
        "python cli.py --help"
    )
    
    # Demo 3: Temperature validation
    run_command(
        ["python", "cli.py", "--temperature", "1.5", "test"],
        "Temperature validation (should fail with invalid temperature)",
        "python cli.py --temperature 1.5 'test'"
    )
    
    # Demo 4: Missing API key error (if no key is set)
    if not os.getenv('ANTHROPIC_API_KEY'):
        run_command(
            ["python", "cli.py", "What is the current time?"],
            "One-shot query without API key (shows helpful error message)",
            "python cli.py 'What is the current time?'"
        )
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nTo use the CLI with an actual API key:")
    print("1. Create a .env file: cp .env.example .env")
    print("2. Add your Anthropic API key to the .env file")
    print("3. Run: python Python_Examples_Agent/cli.py")
    print("\nFor more information, see README.md")
    print()

if __name__ == "__main__":
    main()
