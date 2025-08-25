#!/usr/bin/env python3
"""
Test script for verifying the client installation of MCP-Solver.
This script checks:
  1. Configuration files existence
  2. Required API keys for supported LLM platforms (based on chosen model)
  3. Client module dependencies
  4. Basic client functionality
  5. Tool calling capability (optional)
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Import our centralized prompt loader
from mcp_solver.core.prompt_loader import load_prompt


# Try to import specific client modules
try:
    from ..client.llm_factory import LLMFactory, ModelInfo
    from ..client.tool_capability_detector import (
        ToolCallCapability,
        ToolCapabilityDetector,
    )
except ImportError:
    # Add parent path to import path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from mcp_solver.client.llm_factory import LLMFactory
    from mcp_solver.client.tool_capability_detector import (
        ToolCallCapability,
        ToolCapabilityDetector,
    )

# Default model for testing
DEFAULT_MODEL = "AT:claude-sonnet-4-20250514"  # Anthropic Claude Sonnet 4

# LM Studio model for testing
LMSTUDIO_MODEL = "LM:ministral-8b-instruct-2410@http://localhost:1234/v1"


class SetupTest:
    def __init__(self):
        self.successes: list[tuple[str, str]] = []  # (test_name, details)
        self.failures: list[tuple[str, str]] = []  # (test_name, error_details)
        self.base_dir = Path(__file__).resolve().parents[3]
        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"
        self.capability_detector = ToolCapabilityDetector()

    def print_result(self, test_name: str, success: bool, details: str | None = None):
        """Print a test result with color and proper formatting."""
        mark = "✓" if success else "✗"
        color = self.GREEN if success else self.RED
        print(f"{color}{mark} {test_name}{self.RESET}")
        if details:
            print(f"  └─ {details}")

    def record_test(self, test_name: str, success: bool, details: str | None = None):
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
            ("z3", "review"),
            ("asp", "instructions"),
            ("asp", "review"),
        ]

        for mode, prompt_type in prompts_to_test:
            try:
                # Attempt to load the prompt using the prompt loader
                content = load_prompt(mode, prompt_type)
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    True,
                    f"Successfully loaded ({len(content)} characters)",
                )
            except FileNotFoundError:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    "Prompt file not found but not required for client",
                )
            except Exception as e:
                self.record_test(
                    f"Prompt file: {mode}/{prompt_type}.md",
                    False,
                    f"Error loading prompt: {e!s}",
                )

        # Check for other files
        files = [(".env", False)]  # Optional but recommended

        for file, required in files:
            exists = self.check_file(file)
            if required:
                self.record_test(
                    f"Configuration file: {file}",
                    exists,
                    (
                        None
                        if exists
                        else f"Required file not found at {self.base_dir / file}"
                    ),
                )
            else:
                self.record_test(
                    f"Configuration file: {file}",
                    True,
                    "Optional file" + (" found" if exists else " not found"),
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
            "dotenv",
        ]

        for package in dependencies:
            try:
                importlib.import_module(package)
                self.record_test(f"Package: {package}", True)
            except ImportError as e:
                self.record_test(
                    f"Package: {package}", False, f"Error importing {package}: {e!s}"
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
                f"API key {key_name} is {'available' if key_available else 'not available'}",
            )

            # Print message for user if key not found
            if not key_available:
                print(
                    f"  To use {model_code}, please set the {key_name} environment variable"
                )
                print(
                    "  You can add it to your .env file or set it in your environment"
                )

        except Exception as e:
            self.record_test("API Key check", False, f"Error checking API key: {e!s}")

    def test_basic_functionality(self, model_code: str):
        """Test basic client factory functionality."""
        print(f"\n{self.BOLD}Basic Functionality:{self.RESET}")

        # Test model code parsing
        try:
            model_info = LLMFactory.parse_model_code(model_code)
            self.record_test(
                "Model code parsing",
                True,
                f"Successfully parsed model code: platform={model_info.platform}, provider={model_info.provider}, model={model_info.model_name}",
            )
        except Exception as e:
            self.record_test(
                "Model code parsing", False, f"Error parsing model code: {e!s}"
            )
            return

        # Test getting expected model type
        try:
            model_type, provider_name = LLMFactory.get_expected_model_type(model_code)
            self.record_test(
                "Model type identification",
                True,
                f"Model will use {provider_name} ({model_type.__name__})",
            )
        except Exception as e:
            self.record_test(
                "Model type identification",
                False,
                f"Error identifying model type: {e!s}",
            )
            return

        # Test model instantiation without actually calling the API
        key_available, _ = LLMFactory.check_api_key_available(model_code)
        if key_available:
            try:
                success, message, _ = LLMFactory.test_create_model(model_code)
                self.record_test("Model instantiation", success, message)
            except Exception as e:
                self.record_test(
                    "Model instantiation",
                    False,
                    f"Error during model creation: {e!s}",
                )
        else:
            # Skip test if key is not available
            self.record_test(
                "Model instantiation",
                True,  # Mark as success to avoid failure
                "Skipped model instantiation test since API key is not available",
            )

    def test_model_completion(self, model_code: str):
        """Test if the model can perform a basic completion task."""
        print(f"\n{self.BOLD}Basic Model Completion Test:{self.RESET}")

        # Skip the test if API key is not available
        key_available, _ = LLMFactory.check_api_key_available(model_code)
        if not key_available:
            self.record_test(
                "Basic model completion",
                True,  # Mark as success to avoid failure
                "Skipped basic model completion test since API key is not available",
            )
            return

        try:
            # Create the model instance - let the LLM factory handle implementation details
            model = LLMFactory.create_model(model_code)

            # Prepare a simple prompt for completion
            messages = [{"role": "user", "content": "What is 123.45 plus 67.89?"}]

            # Try to invoke the model with the prompt
            response = model.invoke(messages)

            # Check if response has content
            if hasattr(response, "content") and response.content.strip():
                self.record_test(
                    "Basic model completion",
                    True,
                    f"Model responded with: {response.content[:100]}...",
                )
            else:
                self.record_test(
                    "Basic model completion",
                    False,
                    f"Model response was empty or in an unexpected format: {response}",
                )

        except Exception as e:
            self.record_test(
                "Basic model completion",
                False,
                f"Error during basic model completion test: {e!s}",
            )

    def test_tool_capability(self, model_code: str):
        """Test the model's tool calling capability and categorize it."""
        print(f"\n{self.BOLD}Tool Capability Test:{self.RESET}")

        # Skip the test if API key is not available
        key_available, _ = LLMFactory.check_api_key_available(model_code)
        if not key_available:
            self.record_test(
                "Tool capability",
                True,  # Mark as success to avoid failure
                "Skipped tool capability test since API key is not available",
            )
            return

        try:
            # Create the model instance
            model = LLMFactory.create_model(model_code)

            # Detect the model's capability
            capability = self.capability_detector.detect_capability(model, model_code)

            # Record the result
            if capability == ToolCallCapability.NATIVE:
                self.record_test(
                    "Tool capability",
                    True,
                    f"Model supports native tool calling ({capability})",
                )
            elif capability == ToolCallCapability.JSON_ONLY:
                self.record_test(
                    "Tool capability",
                    True,
                    f"Model supports JSON output but not native tool calling ({capability})",
                )
            else:  # NONE
                self.record_test(
                    "Tool capability",
                    False,
                    f"Model has limited tool capability ({capability})",
                )

            # Store the capability for other tests to use
            self.model_capability = capability

        except Exception as e:
            self.record_test(
                "Tool capability", False, f"Error during tool capability test: {e!s}"
            )
            # Set a default capability in case of error
            self.model_capability = ToolCallCapability.NONE

    def test_json_extraction(self, model_code: str):
        """Test if tool calls can be extracted from the model's JSON output."""
        print(f"\n{self.BOLD}JSON Tool Call Test:{self.RESET}")

        # Skip if we already know the model can't output JSON
        if (
            hasattr(self, "model_capability")
            and self.model_capability == ToolCallCapability.NONE
        ):
            self.record_test(
                "JSON tool call extraction",
                False,
                "Skipped because model doesn't support JSON output",
            )
            return

        # Skip the test if API key is not available
        key_available, _ = LLMFactory.check_api_key_available(model_code)
        if not key_available:
            self.record_test(
                "JSON tool call extraction",
                True,  # Mark as success to avoid failure
                "Skipped JSON tool call test since API key is not available",
            )
            return

        try:
            # Create the model instance
            model = LLMFactory.create_model(model_code)

            # Prepare a prompt that should trigger a tool call in JSON format
            messages = [
                {
                    "role": "user",
                    "content": """
                Please respond with a JSON object that represents a tool call to calculate 
                123.45 plus 67.89. Use this format:
                
                {
                  "function_call": {
                    "name": "calculator",
                    "arguments": {
                      "x": 123.45,
                      "y": 67.89,
                      "operation": "add"
                    }
                  }
                }
                """,
                }
            ]

            # Invoke the model
            response = model.invoke(messages)

            # Get the response content
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Try to extract a tool call
            has_tool_call, tool_info = self.capability_detector.extract_tool_call(
                response_text
            )

            if has_tool_call:
                # Check if it has the expected structure
                tool_name = tool_info.get("name", "")
                args = tool_info.get("arguments", {})

                if isinstance(args, str):
                    # Try to parse the arguments if they're a string
                    try:
                        args = json.loads(args)
                    except:
                        pass

                # Check if the tool call looks valid
                calculator_terms = ["calculator", "calc", "add", "plus"]
                is_calculator = any(
                    term.lower() in tool_name.lower() for term in calculator_terms
                )

                if is_calculator and isinstance(args, dict):
                    x_val = args.get("x", None)
                    y_val = args.get("y", None)

                    if (
                        x_val is not None
                        and y_val is not None
                        and (
                            abs(float(x_val) - 123.45) < 1.0
                            or abs(float(y_val) - 67.89) < 1.0
                        )
                    ):
                        self.record_test(
                            "JSON tool call extraction",
                            True,
                            f"Successfully extracted valid tool call: {tool_name} with args {args}",
                        )
                    else:
                        self.record_test(
                            "JSON tool call extraction",
                            False,
                            f"Extracted tool call but arguments don't match expected: {args}",
                        )
                else:
                    self.record_test(
                        "JSON tool call extraction",
                        False,
                        f"Extracted tool call but structure doesn't match expected: {tool_info}",
                    )
            else:
                self.record_test(
                    "JSON tool call extraction",
                    False,
                    "No tool call could be extracted from the model's output",
                )

        except Exception as e:
            self.record_test(
                "JSON tool call extraction",
                False,
                f"Error during JSON tool call extraction test: {e!s}",
            )

    def run_all_tests(
        self, model_code: str = DEFAULT_MODEL, test_tool_calling: bool = False
    ):
        """Run all setup tests and display results."""
        print(f"{self.BOLD}=== MCP-Solver Client Setup Test ==={self.RESET}")
        print(f"Testing with model code: {model_code}")

        self.test_configuration_files()
        self.test_client_dependencies()
        self.test_api_keys(model_code)
        self.test_basic_functionality(model_code)

        # Run tool calling tests if requested
        if test_tool_calling:
            # First test basic model completion
            self.test_model_completion(model_code)

            # Then test tool capability detection
            self.test_tool_capability(model_code)

            # Test JSON tool call extraction if the model supports JSON output
            if (
                hasattr(self, "model_capability")
                and self.model_capability != ToolCallCapability.NONE
            ):
                self.test_json_extraction(model_code)

        print(f"\n{self.BOLD}=== Test Summary ==={self.RESET}")
        print(f"Passed: {len(self.successes)}")
        print(f"Failed: {len(self.failures)}")

        # Show summary of model capability if tool calling was tested
        if test_tool_calling and hasattr(self, "model_capability"):
            print(f"\n{self.BOLD}=== Model Compatibility Summary ==={self.RESET}")
            capability = self.model_capability

            if capability == ToolCallCapability.NATIVE:
                print(
                    f"{self.GREEN}This model supports native tool calling.{self.RESET}"
                )
                print("✅ Compatible with all MCP-Solver features")
                print("✅ Will work with client.py's ReAct agent out of the box")

            elif capability == ToolCallCapability.JSON_ONLY:
                print(
                    f"{self.GREEN}This model can produce proper JSON for tool calls but doesn't invoke tools natively.{self.RESET}"
                )
                print("✅ Compatible with client.py through LangGraph's adaptation")
                print("✅ Will work with JSON extraction approach")
                print("ℹ️  May require enhanced system prompts for optimal performance")

            else:  # NONE
                print(
                    f"{self.RED}This model has limited tool calling capabilities.{self.RESET}"
                )
                print("⚠️  May not work well with client.py's ReAct agent")
                print("⚠️  Consider using a different model for best results")

        if self.failures:
            print(f"\n{self.BOLD}Failed Tests:{self.RESET}")
            for test, details in self.failures:
                print(f"\n{self.RED}✗ {test}{self.RESET}")
                print(f"  └─ {details}")
            print(
                "\nClient setup incomplete. Please fix the issues above before proceeding."
            )
            sys.exit(1)
        else:
            print(f"\n{self.GREEN}✓ All client tests passed successfully!{self.RESET}")
            print("\nClient system is ready to use MCP-Solver with LLM integration.")
            sys.exit(0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the MCP-Solver client setup")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model code to test (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mc",
        type=str,
        help="Direct model code (e.g., OR:mistralai/ministral-3b or LM:model@url). "
        "Format: '<platform>:<provider>/<model>' or 'LM:model@url' for local models. "
        "Overrides --model if provided.",
    )
    parser.add_argument(
        "--test-tool-calling",
        action="store_true",
        help="Test if tool calling works with the specified model",
    )
    args = parser.parse_args()

    test = SetupTest()
    model_code = args.mc if args.mc else args.model

    print(f"Testing with model code: {model_code}")
    test.run_all_tests(model_code, test_tool_calling=args.test_tool_calling)


if __name__ == "__main__":
    main()
