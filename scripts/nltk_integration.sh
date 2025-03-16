#!/usr/bin/env python
"""
NLTK Dependencies Installation Script for Gray Swan Arena.

This script ensures that all required NLTK packages are available
and installs them if necessary. It's designed to be run during application
initialization to ensure a consistent environment.
"""

import os
import sys
import logging
import platform
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nltk_dependencies")

# ANSI color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(message: str) -> None:
    """Print a success message in green."""
    print(f"{GREEN}{message}{RESET}")

def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    print(f"{YELLOW}{message}{RESET}")

def print_error(message: str) -> None:
    """Print an error message in red."""
    print(f"{RED}{message}{RESET}")

def print_info(message: str) -> None:
    """Print an info message in blue."""
    print(f"{BLUE}{message}{RESET}")

def print_section(title: str) -> None:
    """Print a section title."""
    width = 80
    print(f"\n{BLUE}{'=' * width}{RESET}")
    print(f"{BLUE}{title.center(width)}{RESET}")
    print(f"{BLUE}{'=' * width}{RESET}\n")

def initialize_nltk(packages: Optional[List[str]] = None, quiet: bool = False) -> Dict[str, bool]:
    """
    Initialize NLTK and download required packages.
    
    Args:
        packages: List of NLTK packages to download, or None to use default list
        quiet: Whether to suppress output
        
    Returns:
        Dictionary with status for each package
    """
    if packages is None:
        packages: List[str] = ["vader_lexicon", "punkt", "stopwords"]
    
    results: Dict[str, bool] = {}
    
    try:
        import nltk
        
        # Get NLTK data path
        nltk_data_path = nltk.data.path
        
        # Try to create a writable directory if needed
        home_dir = os.path.expanduser("~")
        nltk_data_dir = os.path.join(home_dir, "nltk_data")
        
        if not os.path.exists(nltk_data_dir):
            try:
                os.makedirs(nltk_data_dir, exist_ok=True)
                if nltk_data_dir not in nltk_data_path:
                    nltk.data.path.append(nltk_data_dir)
            except Exception as e:
                if not quiet:
                    logger.warning(f"Could not create NLTK data directory: {e}")
        
        # Download each package
        for package in packages:
            try:
                # Check if the package is already downloaded
                try:
                    nltk.data.find(f"tokenizers/{package}.zip" if package == "punkt" else 
                                   f"corpora/{package}.zip" if package == "stopwords" else
                                   f"sentiment/{package}.zip")
                    results[package] = True
                    if not quiet:
                        logger.info(f"NLTK package '{package}' is already available.")
                except LookupError:
                    # Download the package
                    nltk.download(package, quiet=quiet)
                    # Verify the download was successful
                    try:
                        nltk.data.find(f"tokenizers/{package}.zip" if package == "punkt" else 
                                       f"corpora/{package}.zip" if package == "stopwords" else
                                       f"sentiment/{package}.zip")
                        results[package] = True
                        if not quiet:
                            logger.info(f"Successfully downloaded NLTK package '{package}'.")
                    except LookupError:
                        results[package] = False
                        if not quiet:
                            logger.error(f"Failed to download NLTK package '{package}'.")
            except Exception as e:
                results[package] = False
                if not quiet:
                    logger.error(f"Error downloading NLTK package '{package}': {e}")
    
    except ImportError:
        logger.error("NLTK is not installed. Install with 'pip install nltk'.")
        for package in packages:
            results[package] = False
    
    return results

def ensure_vader_lexicon() -> bool:
    """
    Ensure the VADER lexicon is available for sentiment analysis.
    
    Returns:
        True if the VADER lexicon is available, False otherwise
    """
    try:
        # Try to import NLTK's SentimentIntensityAnalyzer
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # Test the analyzer with a simple sentence
        sia.polarity_scores("Test sentence.")
        return True
    except Exception as e:
        logger.error(f"Error ensuring VADER lexicon: {e}")
        # Try to initialize NLTK and download the VADER lexicon
        results = initialize_nltk(["vader_lexicon"])
        return results.get("vader_lexicon", False)

def get_sentiment_analyzer():
    """
    Get a SentimentIntensityAnalyzer instance if available.
    
    Returns:
        A SentimentIntensityAnalyzer instance or None if not available
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"Could not create SentimentIntensityAnalyzer: {e}")
        return None

def update_project_dependencies() -> Dict[str, bool]:
    """
    Update project dependency files to include NLTK.
    
    Returns:
        Dictionary mapping file paths to whether they were updated
    """
    results = {}
    
    # Check if we need to update pyproject.toml
    try:
        pyproject_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "pyproject.toml")
        if os.path.exists(pyproject_path):
            with open(pyproject_path, 'r') as f:
                content = f.read()
                
            # Check if nltk is already in dependencies
            if "nltk" not in content:
                # This is just a simple check - in practice, you might want to use
                # a proper TOML parser to add the dependency in the right place
                results[pyproject_path] = False
                logger.info(f"NLTK is not in {pyproject_path}, but we're not modifying it")
            else:
                results[pyproject_path] = False
                logger.info(f"NLTK is already in {pyproject_path}")
    except Exception as e:
        logger.error(f"Error checking pyproject.toml: {e}")
        results["pyproject.toml"] = False
    
    # Check if we need to update requirements.txt
    try:
        requirements_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "requirements.txt")
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r') as f:
                content = f.readlines()
                
            # Check if nltk is already in requirements
            has_nltk = any(line.strip().startswith("nltk") for line in content)
            
            if not has_nltk:
                # Add nltk to requirements.txt
                with open(requirements_path, 'a') as f:
                    f.write("nltk>=3.8.1\n")
                results[requirements_path] = True
                logger.info(f"Added NLTK to {requirements_path}")
            else:
                results[requirements_path] = False
                logger.info(f"NLTK is already in {requirements_path}")
    except Exception as e:
        logger.error(f"Error updating requirements.txt: {e}")
        results["requirements.txt"] = False
    
    return results

def create_nltk_init_script() -> Tuple[bool, Optional[str]]:
    """
    Create the NLTK utilities module if it doesn't exist.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        utils_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "cybersec_agents", "utils")
        if not os.path.exists(utils_dir):
            os.makedirs(utils_dir, exist_ok=True)
            
        nltk_utils_path = os.path.join(utils_dir, "nltk_utils.py")
        
        # Define the content of the NLTK utilities module
        content = '''"""
NLTK Utilities for cybersec-agents.

This module provides utilities for working with NLTK, including
initialization, text processing, and sentiment analysis.
"""

import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def initialize_nltk(packages: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Initialize NLTK and download required packages.
    
    Args:
        packages: List of NLTK packages to download, or None to use default list
        
    Returns:
        Dictionary with status for each package
    """
    if packages is None:
        packages = ["vader_lexicon", "punkt", "stopwords"]
    
    results = {}
    
    try:
        import nltk
        
        # Create the NLTK data directory if it doesn't exist
        home_dir = os.path.expanduser("~")
        nltk_data_dir = os.path.join(home_dir, "nltk_data")
        
        if not os.path.exists(nltk_data_dir):
            try:
                os.makedirs(nltk_data_dir, exist_ok=True)
                if nltk_data_dir not in nltk.data.path:
                    nltk.data.path.append(nltk_data_dir)
            except Exception as e:
                logger.warning(f"Could not create NLTK data directory: {e}")
        
        # Download each package
        for package in packages:
            try:
                # Check if the package is already downloaded
                try:
                    nltk.data.find(f"tokenizers/{package}" if package == "punkt" else 
                                   f"corpora/{package}" if package == "stopwords" else
                                   f"sentiment/{package}")
                    results[package] = True
                    logger.debug(f"NLTK package '{package}' is already available.")
                except LookupError:
                    # Download the package
                    nltk.download(package, quiet=True)
                    # Verify the download was successful
                    try:
                        nltk.data.find(f"tokenizers/{package}" if package == "punkt" else 
                                       f"corpora/{package}" if package == "stopwords" else
                                       f"sentiment/{package}")
                        results[package] = True
                        logger.info(f"Successfully downloaded NLTK package '{package}'.")
                    except LookupError:
                        results[package] = False
                        logger.error(f"Failed to download NLTK package '{package}'.")
            except Exception as e:
                results[package] = False
                logger.error(f"Error downloading NLTK package '{package}': {e}")
    
    except ImportError:
        logger.error("NLTK is not installed. Install with 'pip install nltk'.")
        for package in packages:
            results[package] = False
    
    return results

def ensure_vader_lexicon() -> bool:
    """
    Ensure the VADER lexicon is available for sentiment analysis.
    
    Returns:
        True if the VADER lexicon is available, False otherwise
    """
    try:
        # Try to import NLTK's SentimentIntensityAnalyzer
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # Test the analyzer with a simple sentence
        sia.polarity_scores("Test sentence.")
        return True
    except Exception as e:
        logger.error(f"Error ensuring VADER lexicon: {e}")
        # Try to initialize NLTK and download the VADER lexicon
        results = initialize_nltk(["vader_lexicon"])
        return results.get("vader_lexicon", False)

def get_sentiment_analyzer():
    """
    Get a SentimentIntensityAnalyzer instance if available.
    
    Returns:
        A SentimentIntensityAnalyzer instance or None if not available
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"Could not create SentimentIntensityAnalyzer: {e}")
        return None

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze the sentiment of the given text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    analyzer = get_sentiment_analyzer()
    if analyzer is None:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    
    return analyzer.polarity_scores(text)

def get_nltk_info() -> Dict[str, Any]:
    """
    Get information about the NLTK installation.
    
    Returns:
        Dictionary with information
    """
    info = {
        "nltk_installed": False,
        "data_path": [],
        "packages": {}
    }
    
    try:
        import nltk
        info["nltk_installed"] = True
        info["data_path"] = nltk.data.path
        
        # Check for common packages
        packages = ["vader_lexicon", "punkt", "stopwords"]
        for package in packages:
            try:
                nltk.data.find(f"tokenizers/{package}" if package == "punkt" else 
                               f"corpora/{package}" if package == "stopwords" else
                               f"sentiment/{package}")
                info["packages"][package] = True
            except LookupError:
                info["packages"][package] = False
    except ImportError:
        pass
    
    return info
'''
        
        # Write the content to the file
        with open(nltk_utils_path, 'w') as f:
            f.write(content)
            
        # Create __init__.py in utils directory if it doesn't exist
        init_path = os.path.join(utils_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write('"""\nUtilities for cybersec-agents.\n"""\n')
                
        return True, None
    except Exception as e:
        return False, str(e)

def run_installation(
    force: bool = False, quiet: bool = False, data_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full NLTK installation process.
    
    Args:
        force: Whether to force reinstallation of packages
        quiet: Whether to suppress output
        data_dir: Optional directory to store NLTK data
        
    Returns:
        Dictionary with results of the installation process
    """
    results = {
        "nltk_initialization": {},
        "vader_lexicon": False,
        "sentiment_analyzer": False,
        "dependency_updates": {},
        "nltk_utils": {}
    }
    
    # Welcome message
    if not quiet:
        print_section("NLTK Dependencies Installation")
        
    # Initialize NLTK and download required packages
    if not quiet:
        print_section("Initializing NLTK")
        
    packages = ["vader_lexicon", "punkt", "stopwords"]
    results["nltk_initialization"] = initialize_nltk(packages, quiet=quiet)
    
    all_packages_available = all(results["nltk_initialization"].values())
    if all_packages_available:
        if not quiet:
            print_success("All NLTK packages are available")
    else:
        missing_packages = [pkg for pkg, status in results["nltk_initialization"].items() if not status]
        if not quiet:
            print_warning(f"Some NLTK packages could not be downloaded: {', '.join(missing_packages)}")
            
    # Check if VADER lexicon is working
    if not quiet:
        print_section("Testing VADER Lexicon")
        
    results["vader_lexicon"] = ensure_vader_lexicon()
    if results["vader_lexicon"]:
        if not quiet:
            print_success("VADER lexicon is working")
    else:
        if not quiet:
            print_warning("VADER lexicon is not working properly")
            
    # Check if SentimentIntensityAnalyzer is working
    if not quiet:
        print_section("Testing Sentiment Analyzer")
        
    sia = get_sentiment_analyzer()
    results["sentiment_analyzer"] = sia is not None
    
    if results["sentiment_analyzer"]:
        # Test the analyzer on a simple sentence
        sentiment = sia.polarity_scores("This is a great day!")
        if not quiet:
            print_success("SentimentIntensityAnalyzer is working")
            print_info(f"Sample sentiment: {sentiment}")
    else:
        if not quiet:
            print_error("SentimentIntensityAnalyzer is not working correctly")

    # Update project dependencies
    if not quiet:
        print_section("Updating Project Dependencies")

    dependency_updates = update_project_dependencies()
    results["dependency_updates"] = dependency_updates

    for file, updated in dependency_updates.items():
        if updated:
            if not quiet:
                print_success(f"Updated {file} to include NLTK")
        else:
            if not quiet:
                print_info(f"No updates needed for {file}")

    # Create NLTK utilities module
    if not quiet:
        print_section("Creating NLTK Utilities Module")

    success, error = create_nltk_init_script()
    results["nltk_utils"] = {"created": success, "error": error}

    if success:
        if not quiet:
            print_success("Created NLTK utilities module")
    else:
        if not quiet:
            print_error(f"Failed to create NLTK utilities module: {error}")

    # Summary
    if not quiet:
        print_section("Installation Summary")
        
        if all([
            all_packages_available,
            results["vader_lexicon"],
            results["sentiment_analyzer"]
        ]):
            print_success("NLTK dependencies installation completed successfully")
        else:
            print_warning("NLTK dependencies installation completed with some issues")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install NLTK dependencies")
    parser.add_argument("--force", action="store_true", help="Force reinstallation of packages")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--data-dir", help="Directory to store NLTK data")
    
    args = parser.parse_args()
    
    run_installation(args.force, args.quiet, args.data_dir)