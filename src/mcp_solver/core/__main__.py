"""
Main entry point for running the MCP Solver package.
"""

import logging
import sys


def main():
    """Default entry point for MiniZinc mode"""
    from .server import main as server_main

    return server_main()


def main_mzn():
    """Entry point for MiniZinc mode (alias of main for consistency)"""
    return main()


def main_z3():
    """Entry point for Z3 mode"""
    try:
        import z3
    except ImportError:
        print("Z3 dependencies not installed. Please install with:")
        print("    uv pip install -e '.[z3]'")
        return 1

    from .server import main as server_main

    # Set command line arguments for Z3 mode
    sys.argv = [sys.argv[0], "--z3"]
    return server_main()


def main_pysat():
    """Entry point for PySAT mode"""
    try:
        import pysat
    except ImportError:
        print("PySAT dependencies not installed. Please install with:")
        print("    uv pip install -e '.[pysat]'")
        return 1

    from .server import main as server_main

    # Set command line arguments for PySAT mode
    sys.argv = [sys.argv[0], "--pysat"]
    return server_main()


def main_maxsat():
    """Entry point for MaxSAT optimization mode"""
    from .server import main as server_main

    # Set command line arguments for MaxSAT mode
    sys.argv = [sys.argv[0], "--maxsat"]
    return server_main()


def main_asp():
    """Entry point for ASP mode"""
    from .server import main as server_main

    # Set command line arguments for ASP mode
    sys.argv = [sys.argv[0], "--asp"]
    return server_main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.error(f"Error in main: {e}")
        sys.exit(1)
