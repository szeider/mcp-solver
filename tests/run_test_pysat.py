#!/usr/bin/env python3
"""
Runs PySAT problems through the test-client
"""
import os
import subprocess
import glob
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Import test configuration
from test_config import (
    MCP_CLIENT_DIR, 
    DEFAULT_TIMEOUT, 
    PYSAT_PROMPT_FILE,
    PYSAT_PROBLEMS_DIR,
    get_prompt_path
)

def validate_files_exist():
    """Check that required files and directories exist"""
    if not os.path.exists(get_prompt_path(PYSAT_PROMPT_FILE)):
        print(f"Error: Prompt file '{PYSAT_PROMPT_FILE}' not found")
        return False
    
    if not os.path.exists(PYSAT_PROBLEMS_DIR):
        print(f"Error: Problems directory '{PYSAT_PROBLEMS_DIR}' not found")
        return False
    
    if not os.path.exists(MCP_CLIENT_DIR):
        print(f"Error: MCP client directory '{MCP_CLIENT_DIR}' not found")
        return False
        
    return True

def run_problem(problem_path, save_results=False):
    """Run a single problem through test-client"""
    problem_name = os.path.basename(problem_path).replace('.md', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"Testing problem: {problem_name}")
    print(f"{'='*60}")
    
    # Get absolute path to the problem file
    abs_problem_path = os.path.abspath(problem_path)
    
    # Create the command with PySAT specific args
    cmd = f"cd {MCP_CLIENT_DIR} && uv run test-client --query {abs_problem_path} --prompt {get_prompt_path(PYSAT_PROMPT_FILE)} --server 'uv run mcp-solver-pysat'"
    
    # Set up result saving if enabled
    if save_results:
        # Define output directory for saving results
        output_dir = os.path.join(os.path.dirname(__file__), "results", "pysat")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file paths with timestamp
        response_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_response.txt")
        model_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_model.py")
    
    # Run the test-client-pysat command
    try:
        print(f"\nExecuting: {cmd}\n")
        print(f"{'-'*60}\n[CLIENT OUTPUT START]\n{'-'*60}")
        
        # Use Popen to capture output in real-time
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                            text=True, bufsize=1, universal_newlines=True) as process:
            
            # Initialize variables to capture model and response if saving is enabled
            if save_results:
                model_content = ""
                agent_response = ""
                capture_model = False
                capture_response = False
            
            # Set up threads to read stdout and stderr
            def read_stream(stream, prefix):
                if save_results:
                    nonlocal model_content, agent_response, capture_model, capture_response
                
                for line in stream:
                    print(f"{prefix}: {line}", end='')
                    
                    # If saving is enabled, parse output to extract model and response
                    if save_results and prefix == "STDOUT":
                        # Capture the final model content
                        if "Current model:" in line:
                            capture_model = True
                            model_content = ""  # Reset to get only the latest model
                            continue
                        elif capture_model:
                            if line.strip().startswith("TOOL:"):
                                capture_model = False
                            else:
                                # Remove the "STDOUT: " prefix if present
                                cleaned_line = line.replace("STDOUT: ", "")
                                model_content += cleaned_line
                        
                        # Capture agent response
                        if "Agent reply received" in line:
                            capture_response = True
                            continue
                        elif capture_response and not line.strip().startswith("[CLIENT OUTPUT END]"):
                            if "------------------------------------------------------------" in line:
                                capture_response = False
                            else:
                                # Remove the "STDOUT: " prefix if present
                                cleaned_line = line.replace("STDOUT: ", "")
                                agent_response += cleaned_line
            
            import threading
            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT"))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR"))
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for threads to complete
            stdout_thread.join()
            stderr_thread.join()
            
            # Wait for process to complete
            process.wait()
            success = process.returncode == 0
        
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END]\n{'-'*60}")
        
        # Save the model and agent response if enabled
        if save_results:
            if model_content:
                with open(model_file, 'w') as f:
                    f.write(model_content)
                print(f"\nSaved final model to: {model_file}")
                
            if agent_response:
                with open(response_file, 'w') as f:
                    f.write(agent_response)
                print(f"\nSaved agent response to: {response_file}")
        
        if success:
            print(f"\n✅ Successfully tested: {problem_name}")
            return True
        else:
            print(f"\n❌ Failed to test {problem_name} (Exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END - ERROR]\n{'-'*60}")
        print(f"\n❌ Failed to test {problem_name}: {str(e)}")
        return False

def main():
    """Main function to run PySAT tests"""
    parser = argparse.ArgumentParser(description="Run PySAT problems through test-client")
    parser.add_argument("--problem", help="Specific problem to run (without .md extension)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, 
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--save", "-s", action="store_true", help="Save test results to files")
    args = parser.parse_args()
    
    # Validate required files exist
    if not validate_files_exist():
        return 1
    
    # Get problem files based on command-line arguments
    if args.problem:
        problem_path = os.path.join(PYSAT_PROBLEMS_DIR, f"{args.problem}")
        if not problem_path.endswith('.md'):
            problem_path += '.md'
            
        if not os.path.exists(problem_path):
            print(f"Error: Problem file '{problem_path}' not found")
            return 1
        problem_files = [problem_path]
    else:
        problem_files = glob.glob(f"{PYSAT_PROBLEMS_DIR}/*.md")
    
    if not problem_files:
        print(f"Error: No problem files found in {PYSAT_PROBLEMS_DIR}")
        return 1
    
    print(f"Found {len(problem_files)} PySAT problem(s) to test")
    
    # Track results
    success_count = 0
    failed_tests = []
    
    # Run each test
    for problem_file in sorted(problem_files):
        if run_problem(problem_file, save_results=args.save):
            success_count += 1
        else:
            failed_tests.append(os.path.basename(problem_file))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: PySAT Tests ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}")
    print(f"Total tests: {len(problem_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(problem_files) - success_count}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    if len(problem_files) == success_count:
        print("\n🎉 All PySAT tests passed! 🎉")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 