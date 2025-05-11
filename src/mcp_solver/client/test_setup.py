#!/usr/bin/env python3
"""
Test script for verifying the client installation of MCP-Solver.
This script checks:
  1. Configuration files existence
  2. Required API keys for supported LLM platforms (based on chosen model)
  3. Client module dependencies
  4. Basic client functionality
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import importlib.util

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt

# Try to import specific client modules
try:
    from ..client.llm_factory import LLMFactory, ModelInfo
except ImportError:
    # Add parent path to import path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from mcp_solver.client.llm_factory import LLMFactory, ModelInfo

# Default model for testing
DEFAULT_MODEL = "AT:claude-3-7-sonnet-20250219"
# LM Studio model for testing
LMSTUDIO_MODEL = "LM:ministral-8b-instruct-2410@http://localhost:1234/v1"

class SetupTest:
    def __init__(self):
        self.successes: List[Tuple[str, str]] = []  # (test_name, details)
        self.failures: List[Tuple[str, str]] = []   # (test_name, error_details)
        self.base_dir = Path(__file__).resolve().parents[3]
        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"

    def print_result(self, test_name: str, success: bool, details: Optional[str] = None):
        """Print a test result with color and proper formatting."""
        mark = "✓" if success else "✗"
        color = self.GREEN if success else self.RED
        print(f"{color}{mark} {test_name}{self.RESET}")
        if details:
            print(f"  └─ {details}")

    def record_test(self, test_name: str, success: bool, details: Optional[str] = None):
        """Record a test result and print it."""
        if success:
            self.successes.append((test_name, details if details else ""))
        else:
            self.failures.append((test_name, details if details else "Test failed"))
        self.print_result(test_name, success, None if success else details)

    def check_file(self, file_name: str) -> bool:
        """Check if a file exists in the base directory."""
        file_path = self.base_dir / file_name
        return file_path.exists()

    def test_configuration_files(self):
        """Test for the presence of required configuration files."""
        print(f"\n{self.BOLD}Configuration Files:{self.RESET}")
        
        # Test prompt files using the centralized prompt loader
        prompts_to_test = [
            ("mzn", "instructions"),
            ("mzn", "review"),
            ("pysat", "instructions"),
            ("pysat", "review"),
            ("z3", "instructions"),
            ("z3", "review")
        ]
        
        for mode, prompt_type in prompts_to_test:
            try:
                # Attempt to load the prompt using the prompt loader
                content = load_prompt(mode, prompt_type)
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    True,
                    f"Successfully loaded ({len(content)} characters)"
                )
            except FileNotFoundError:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    f"Prompt file not found but not required for client"
                )
            except Exception as e:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    f"Error loading prompt: {str(e)}"
                )
        
        # Check for other files
        files = [
            (".env", False)  # Optional but recommended
        ]
        
        for file, required in files:
            exists = self.check_file(file)
            if required:
                self.record_test(
                    f"Configuration file: {file}",
                    exists,
                    None if exists else f"Required file not found at {self.base_dir / file}"
                )
            else:
                self.record_test(
                    f"Configuration file: {file}",
                    True,
                    "Optional file" + (" found" if exists else " not found")
                )

    def test_client_dependencies(self):
        """Test that required client dependencies are installed."""
        print(f"\n{self.BOLD}Client Dependencies:{self.RESET}")
        
        dependencies = [
            "langchain_core",
            "langgraph",
            "langchain_openai",
            "langchain_anthropic",
            "langchain_google_genai",
            "dotenv"
        ]
        
        for package in dependencies:
            try:
                importlib.import_module(package)
                self.record_test(
                    f"Package: {package}",
                    True
                )
            except ImportError as e:
                self.record_test(
                    f"Package: {package}",
                    False,
                    f"Error importing {package}: {str(e)}"
                )

    def test_api_keys(self, model_code: str):
        """Test that the API key required for the specified model is available."""
        print(f"\n{self.BOLD}API Keys:{self.RESET}")
        
        # Use the factory's check_api_key_available method
        try:
            key_available, key_name = LLMFactory.check_api_key_available(model_code)
            
            self.record_test(
                f"API Key for {model_code}",
                key_available,
                f"API key {key_name} is {'available' if key_available else 'not available'}"
            )
            
            # Print message for user if key not found
            if not key_available:
                print(f"  To use {model_code}, please set the {key_name} environment variable")
                print(f"  You can add it to your .env file or set it in your environment")
                
        except Exception as e:
            self.record_test(
                "API Key check",
                False,
                f"Error checking API key: {str(e)}"
            )

    def test_basic_functionality(self, model_code: str):
        """Test basic client factory functionality."""
        print(f"\n{self.BOLD}Basic Functionality:{self.RESET}")
        
        # Test model code parsing
        try:
            model_info = LLMFactory.parse_model_code(model_code)
            self.record_test(
                "Model code parsing",
                True,
                f"Successfully parsed model code: platform={model_info.platform}, provider={model_info.provider}, model={model_info.model_name}"
            )
        except Exception as e:
            self.record_test(
                "Model code parsing",
                False,
                f"Error parsing model code: {str(e)}"
            )
            return
        
        # Test getting expected model type
        try:
            model_type, provider_name = LLMFactory.get_expected_model_type(model_code)
            self.record_test(
                "Model type identification",
                True,
                f"Model will use {provider_name} ({model_type.__name__})"
            )
        except Exception as e:
            self.record_test(
                "Model type identification",
                False,
                f"Error identifying model type: {str(e)}"
            )
            return
            
        # Test model instantiation without actually calling the API
        key_available, _ = LLMFactory.check_api_key_available(model_code)
        if key_available:
            try:
                success, message, _ = LLMFactory.test_create_model(model_code)
                self.record_test(
                    "Model instantiation",
                    success,
                    message
                )
            except Exception as e:
                self.record_test(
                    "Model instantiation",
                    False,
                    f"Error during model creation: {str(e)}"
                )
        else:
            # Skip test if key is not available
            self.record_test(
                "Model instantiation",
                True,  # Mark as success to avoid failure
                "Skipped model instantiation test since API key is not available"
            )

    def run_all_tests(self, model_code: str = DEFAULT_MODEL):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver Client Setup Test ==={self.RESET}")
        print(f"Testing with model code: {model_code}")
        
        self.test_configuration_files()
        self.test_client_dependencies()
        self.test_api_keys(model_code)
        self.test_basic_functionality(model_code)
        
        print(f"\n{self.BOLD}=== Test Summary ==={self.RESET}")
        print(f"Passed: {len(self.successes)}")
        print(f"Failed: {len(self.failures)}")
        
        if self.failures:
            print(f"\n{self.BOLD}Failed Tests:{self.RESET}")
            for test, details in self.failures:
                print(f"\n{self.RED}✗ {test}{self.RESET}")
                print(f"  └─ {details}")
            print("\nClient setup incomplete. Please fix the issues above before proceeding.")
            sys.exit(1)
        else:
            print(f"\n{self.GREEN}✓ All client tests passed successfully!{self.RESET}")
            print("\nClient system is ready to use MCP-Solver with LLM integration.")
            sys.exit(0)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the MCP-Solver client setup')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Model code to test (default: {DEFAULT_MODEL})')
    parser.add_argument('--mc', type=str,
                        help='Direct model code (e.g., OR:mistralai/ministral-3b). Format: \'<platform>:<provider>/<model>\'. '
                             'Overrides --model if provided.')
    parser.add_argument('--lmstudio', action='store_true',
                        help='Test LM Studio local model integration')
    args = parser.parse_args()

    test = SetupTest()
    if args.mc:
        print(f"Testing with custom model code: {args.mc}")
        test.run_all_tests(args.mc)
    elif args.lmstudio:
        print(f"Testing with LM Studio model: {LMSTUDIO_MODEL}")
        test.run_all_tests(LMSTUDIO_MODEL)
    else:
        test.run_all_tests(args.model)

if __name__ == "__main__":
    main() 