"""
Main entry point for Gray Swan Arena.

This module is a wrapper around the main_di module, which provides the
actual implementation of the Gray Swan Arena pipeline with dependency injection.
"""

import sys

from cybersec_agents.grayswan.main_di import main

if __name__ == "__main__":
    sys.exit(main()) 