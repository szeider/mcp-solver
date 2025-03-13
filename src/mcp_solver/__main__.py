"""
Main entry point for running the MCP Solver package.
"""

import sys
import logging
import asyncio
from .server import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.error(f"Error in main: {e}")
        sys.exit(1) 