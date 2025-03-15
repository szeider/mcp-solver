"""
Test all MCP Solver modes.

This script tests if the MCP Solver server can be started in all three modes:
1. MiniZinc mode (default)
2. Z3 mode (--z3 --lite)
3. PySAT mode (--pysat --lite)
"""

import sys
import os
import time
import subprocess
import signal
import threading

# Add parent directory to path to ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def start_server(mode=None):
    """
    Start the MCP Solver server in the specified mode.
    
    Args:
        mode: One of "minizinc" (default), "z3", or "pysat"
    
    Returns:
        The process object
    """
    # Use Python module invocation to ensure we use the installed package
    cmd = ["uv", "run", "python", "-m", "mcp_solver"]
    
    # Add appropriate flags based on mode
    if mode == "z3":
        cmd.extend(["--z3", "--lite"])
    elif mode == "pysat":
        cmd.extend(["--pysat", "--lite"])
    
    # Start the server process
    process = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True,
                              bufsize=1)
    
    return process

def monitor_output(process, success_marker, timeout=10):
    """Monitor the process output for a success marker."""
    start_time = time.time()
    success = False
    output_lines = []
    
    def read_output():
        nonlocal success
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line.strip())
            if success_marker in line:
                success = True
                break
    
    # Start reading in a separate thread
    t = threading.Thread(target=read_output)
    t.daemon = True
    t.start()
    
    # Wait for success or timeout
    while time.time() - start_time < timeout and not success:
        if not t.is_alive() and process.poll() is not None:
            # Process ended before success marker was found
            break
        time.sleep(0.1)
    
    return success, output_lines

def test_mode(mode):
    """Test a specific mode."""
    mode_name = mode if mode else "minizinc"
    print(f"\n=== Testing {mode_name.upper()} Mode ===")
    
    # Determine the success marker based on mode
    if mode == "z3":
        success_marker = "Using Z3 model manager"
    elif mode == "pysat":
        success_marker = "Using PySAT model manager"
    else:
        success_marker = "Using MiniZinc model manager"
    
    # Start the server
    print(f"Starting MCP Solver in {mode_name} mode...")
    process = start_server(mode)
    
    # Monitor for success marker
    success, output_lines = monitor_output(process, success_marker)
    
    # Terminate the process
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    # Check stderr if we didn't find the success marker in stdout
    if not success:
        for line in iter(process.stderr.readline, ''):
            output_lines.append(f"STDERR: {line.strip()}")
            if success_marker in line:
                success = True
                break
    
    # Print collected output
    print("\nServer output:")
    for line in output_lines:
        print(f"  {line}")
    
    # Report result
    if success:
        print(f"\n✅ SUCCESS: Server started successfully in {mode_name} mode")
    else:
        print(f"\n❌ FAILURE: Could not verify that server started in {mode_name} mode")
    
    return success

def run_all_tests():
    """Run tests for all modes."""
    results = {}
    
    # Test all three modes
    modes = [None, "z3", "pysat"]  # None represents the default MiniZinc mode
    
    for mode in modes:
        mode_name = mode if mode else "minizinc"
        results[mode_name] = test_mode(mode)
    
    # Print summary
    print("\n=== Test Summary ===")
    all_pass = True
    
    for mode, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{mode.upper()}: {status}")
        all_pass = all_pass and success
    
    return all_pass

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 