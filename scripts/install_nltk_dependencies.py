#!/usr/bin/env python
"""
NLTK Dependencies Installer

This script provides a robust installation process for NLTK and required data packages,
with a focus on the VADER lexicon used for sentiment analysis in the Gray Swan Arena framework.

Features:
- Checks and installs NLTK if not already installed
- Downloads required NLTK data packages with fallback mechanisms
- Handles permission issues by trying multiple installation directories
- Verifies successful installation
- Updates project dependencies to include NLTK

Usage:
    python scripts/install_nltk_dependencies.py [--force] [--quiet] [--data-dir DIR]

Options:
    --force     Force reinstallation even if already installed
    --quiet     Suppress detailed output
    --data-dir  Specify custom NLTK data directory
"""

import os
import sys
import platform
import subprocess
import argparse
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Check if we're running in a terminal that supports colors
def supports_color() -> bool:
    """Check if the terminal supports color output."""
    if platform.system() == 'Windows':
        return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

# Color formatting function
def colorize(text: str, color: str) -> str:
    """Add color to text if supported."""
    if supports_color():
        return f"{color}{text}{Colors.ENDC}"
    return text

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + colorize("=" * 80, Colors.BOLD))
    print(colorize(f" {text} ", Colors.BOLD + Colors.HEADER))
    print(colorize("=" * 80, Colors.BOLD) + "\n")

def print_section(text: str) -> None:
    """Print a formatted section header."""
    print("\n" + colorize("-" * 40, Colors.BOLD))
    print(colorize(f" {text} ", Colors.BOLD + Colors.BLUE))
    print(colorize("-" * 40, Colors.BOLD) + "\n")

def print_success(text: str) -> None:
    """Print a success message."""
    print(colorize("✓ " + text, Colors.GREEN))

def print_warning(text: str) -> None:
    """Print a warning message."""
    print(colorize("⚠ " + text, Colors.YELLOW))

def print_error(text: str) -> None:
    """Print an error message."""
    print(colorize("✗ " + text, Colors.RED))

def print_info(text: str) -> None:
    """Print an informational message."""
    print(colorize("ℹ " + text, Colors.BLUE))

def check_nltk_installation() -> Tuple[bool, str, Optional[str]]:
    """
    Check if NLTK is installed and get its version.
    
    Returns:
        Tuple of (is_installed, version_str, error_message)
    """
    try:
        import nltk
        return True, nltk.__version__, None
    except ImportError as e:
        return False, "", str(e)

def install_nltk(quiet: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Install NLTK using pip.
    
    Args:
        quiet: Whether to suppress output
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if not quiet:
            print_info("Installing NLTK...")
        
        # Use subprocess to install NLTK
        cmd = [sys.executable, "-m", "pip", "install", "nltk"]
        if quiet:
            cmd.append("--quiet")
        
        subprocess.check_call(cmd)
        
        # Verify installation
        try:
            import nltk
            if not quiet:
                print_success(f"Successfully installed NLTK version {nltk.__version__}")
            return True, None
        except ImportError as e:
            return False, f"NLTK installation succeeded but import failed: {str(e)}"
    except Exception as e:
        return False, f"Failed to install NLTK: {str(e)}"

def get_nltk_data_dirs() -> List[str]:
    """
    Get all NLTK data directories.
    
    Returns:
        List of directory paths
    """
    try:
        import nltk.data
        return nltk.data.path
    except ImportError:
        # If NLTK is not installed, return default paths
        home_dir = os.path.expanduser("~")
        return [
            os.path.join(home_dir, "nltk_data"),
            os.path.join(sys.prefix, "nltk_data"),
            os.path.join(sys.prefix, "share", "nltk_data"),
            os.path.join(sys.prefix, "lib", "nltk_data"),
            os.path.join(os.path.dirname(sys.executable), "nltk_data"),
        ]

def check_directory_permissions(directory: str) -> Dict[str, bool]:
    """
    Check if a directory exists and has read/write permissions.
    
    Args:
        directory: Path to check
        
    Returns:
        Dictionary with permission status
    """
    dir_path = Path(directory)
    
    # Check if directory exists
    exists = dir_path.exists()
    
    # If it doesn't exist, check if parent directory exists and is writable
    if not exists:
        parent_dir = dir_path.parent
        parent_exists = parent_dir.exists()
        parent_writable = os.access(parent_dir, os.W_OK) if parent_exists else False
        return {
            "exists": False,
            "readable": False,
            "writable": False,
            "parent_exists": parent_exists,
            "parent_writable": parent_writable,
            "can_create": parent_exists and parent_writable
        }
    
    # Check permissions
    readable = os.access(directory, os.R_OK)
    writable = os.access(directory, os.W_OK)
    
    return {
        "exists": True,
        "readable": readable,
        "writable": writable,
        "parent_exists": True,
        "parent_writable": True,
        "can_create": True
    }

def find_writable_directory() -> Optional[str]:
    """
    Find a writable directory for NLTK data.
    
    Returns:
        Path to a writable directory, or None if none found
    """
    # Try standard NLTK data directories
    for directory in get_nltk_data_dirs():
        perms = check_directory_permissions(directory)
        if perms["exists"] and perms["writable"]:
            return directory
        elif perms["can_create"]:
            try:
                os.makedirs(directory, exist_ok=True)
                return directory
            except:
                pass
    
    # Try user's home directory
    home_dir = os.path.expanduser("~")
    nltk_data_dir = os.path.join(home_dir, "nltk_data")
    try:
        os.makedirs(nltk_data_dir, exist_ok=True)
        return nltk_data_dir
    except:
        pass
    
    # Try temporary directory
    try:
        temp_dir = tempfile.gettempdir()
        nltk_data_dir = os.path.join(temp_dir, "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        return nltk_data_dir
    except:
        pass
    
    # Try current directory
    try:
        nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        return nltk_data_dir
    except:
        pass
    
    return None

def download_nltk_data(package: str, download_dir: Optional[str] = None, quiet: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Download NLTK data package with fallback mechanisms.
    
    Args:
        package: Name of the NLTK data package to download
        download_dir: Directory to download to (None for default)
        quiet: Whether to suppress output
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        import nltk
        
        # Try to find the package first
        try:
            nltk.data.find(package)
            if not quiet:
                print_success(f"Package '{package}' is already downloaded")
            return True, None
        except LookupError:
            # Package not found, need to download
            pass
        
        if not quiet:
            print_info(f"Downloading NLTK package: {package}")
        
        # Try downloading to specified directory
        if download_dir:
            try:
                os.makedirs(download_dir, exist_ok=True)
                nltk.download(package, download_dir=download_dir, quiet=quiet)
                
                # Add the directory to NLTK's search path
                if download_dir not in nltk.data.path:
                    nltk.data.path.append(download_dir)
                
                # Verify download
                try:
                    nltk.data.find(package)
                    if not quiet:
                        print_success(f"Successfully downloaded '{package}' to {download_dir}")
                    return True, None
                except LookupError:
                    if not quiet:
                        print_warning(f"Download to {download_dir} failed verification")
                    # Continue to fallback methods
            except Exception as e:
                if not quiet:
                    print_warning(f"Failed to download to {download_dir}: {str(e)}")
                # Continue to fallback methods
        
        # Try downloading to default location
        try:
            nltk.download(package, quiet=quiet)
            
            # Verify download
            try:
                nltk.data.find(package)
                if not quiet:
                    print_success(f"Successfully downloaded '{package}' to default location")
                return True, None
            except LookupError:
                if not quiet:
                    print_warning("Download to default location failed verification")
                # Continue to fallback methods
        except Exception as e:
            if not quiet:
                print_warning(f"Failed to download to default location: {str(e)}")
            # Continue to fallback methods
        
        # Try to find a writable directory
        writable_dir = find_writable_directory()
        if writable_dir:
            try:
                nltk.download(package, download_dir=writable_dir, quiet=quiet)
                
                # Add the directory to NLTK's search path
                if writable_dir not in nltk.data.path:
                    nltk.data.path.append(writable_dir)
                
                # Verify download
                try:
                    nltk.data.find(package)
                    if not quiet:
                        print_success(f"Successfully downloaded '{package}' to {writable_dir}")
                    return True, None
                except LookupError:
                    return False, f"Download to {writable_dir} failed verification"
            except Exception as e:
                return False, f"Failed to download to {writable_dir}: {str(e)}"
        else:
            return False, "Could not find a writable directory for NLTK data"
    except ImportError:
        return False, "NLTK is not installed"

def manual_download_nltk_data(package: str, target_dir: str, quiet: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Manually download NLTK data package using urllib.
    
    Args:
        package: Name of the NLTK data package to download
        target_dir: Directory to download to
        quiet: Whether to suppress output
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        import urllib.request
        import zipfile
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Map package names to URLs
        package_urls = {
            "vader_lexicon": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip",
            "punkt": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip",
            "stopwords": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip"
        }
        
        if package not in package_urls:
            return False, f"No manual download URL for package: {package}"
        
        url = package_urls[package]
        
        if not quiet:
            print_info(f"Manually downloading {package} from {url}")
        
        # Download the zip file
        zip_path = os.path.join(target_dir, f"{package}.zip")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        package_type = "sentiment" if package == "vader_lexicon" else "tokenizers" if package == "punkt" else "corpora"
        extract_dir = os.path.join(target_dir, package_type)
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up
        os.remove(zip_path)
        
        if not quiet:
            print_success(f"Successfully manually downloaded and extracted {package}")
        
        return True, None
    except Exception as e:
        return False, f"Manual download failed: {str(e)}"

def verify_nltk_data(package: str) -> bool:
    """
    Verify that an NLTK data package is available.
    
    Args:
        package: Name of the NLTK data package to verify
        
    Returns:
        True if the package is available, False otherwise
    """
    try:
        import nltk.data
        nltk.data.find(package)
        return True
    except (ImportError, LookupError):
        return False

def verify_sentiment_analyzer() -> bool:
    """
    Verify that SentimentIntensityAnalyzer works correctly.
    
    Returns:
        True if the analyzer works, False otherwise
    """
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sia.polarity_scores("This is a test.")
        return True
    except Exception:
        return False

def update_project_dependencies() -> Dict[str, bool]:
    """
    Update project dependency files to include NLTK.
    
    Returns:
        Dictionary with update status for each file
    """
    results = {
        "requirements.txt": False,
        "setup.py": False
    }
    
    # Update requirements.txt
    if os.path.exists("requirements.txt"):
        try:
            with open("requirements.txt", "r") as f:
                content = f.read()
            
            if "nltk" not in content:
                with open("requirements.txt", "a") as f:
                    f.write("\n# NLP libraries\nnltk==3.8.1  # Required for sentiment analysis\n")
                results["requirements.txt"] = True
        except Exception:
            pass
    
    # Update setup.py
    if os.path.exists("setup.py"):
        try:
            with open("setup.py", "r") as f:
                content = f.read()
            
            if "nltk" not in content:
                # Find the install_requires section
                if "install_requires=[" in content:
                    updated_content = content.replace(
                        "install_requires=[",
                        "install_requires=[\n        \"nltk\",  # Required for sentiment analysis"
                    )
                    
                    with open("setup.py", "w") as f:
                        f.write(updated_content)
                    
                    results["setup.py"] = True
        except Exception:
            pass
    
    return results

def create_nltk_init_script() -> Tuple[bool, Optional[str]]:
    """
    Create an initialization script for NLTK data.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        script_dir = os.path.join("cybersec_agents", "utils")
        os.makedirs(script_dir, exist_ok=True)
        
        script_path = os.path.join(script_dir, "nltk_utils.py")
        
        with open(script_path, "w") as f:
            f.write("""\"\"\"
NLTK utilities for cybersec_agents.

This module provides utilities for initializing NLTK and downloading required data.
\"\"\"

import os
import sys
import logging
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def initialize_nltk(packages: List[str] = None, quiet: bool = True) -> Dict[str, bool]:
    \"\"\"
    Initialize NLTK and download required data packages.
    
    Args:
        packages: List of NLTK packages to download (default: vader_lexicon, punkt, stopwords)
        quiet: Whether to suppress output
        
    Returns:
        Dictionary with status for each package
    \"\"\"
    if packages is None:
        packages = ["vader_lexicon", "punkt", "stopwords"]
    
    results = {}
    
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
                # Check if package is already downloaded
                try:
                    nltk.data.find(package)
                    results[package] = True
                    continue
                except LookupError:
                    pass
                
                # Try to download the package
                nltk.download(package, quiet=quiet)
                
                # Verify download
                try:
                    nltk.data.find(package)
                    results[package] = True
                except LookupError:
                    # Try downloading to specific directory
                    nltk.download(package, download_dir=nltk_data_dir, quiet=quiet)
                    
                    # Verify again
                    try:
                        nltk.data.find(package)
                        results[package] = True
                    except LookupError:
                        results[package] = False
            except Exception as e:
                if not quiet:
                    logger.warning(f"Error downloading {package}: {e}")
                results[package] = False
    
    except ImportError as e:
        if not quiet:
            logger.error(f"NLTK import error: {e}")
        for package in packages:
            results[package] = False
    
    return results

def get_sentiment_analyzer():
    \"\"\"
    Get a SentimentIntensityAnalyzer instance with proper initialization.
    
    Returns:
        SentimentIntensityAnalyzer instance or None if not available
    \"\"\"
    try:
        # Initialize NLTK
        initialize_nltk(["vader_lexicon"])
        
        # Import and initialize SentimentIntensityAnalyzer
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception as e:
        logger.warning(f"Could not initialize SentimentIntensityAnalyzer: {e}")
        return None

def analyze_sentiment(text: str) -> Dict[str, float]:
    \"\"\"
    Analyze sentiment of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores or empty dict if analysis fails
    \"\"\"
    try:
        sia = get_sentiment_analyzer()
        if sia:
            return sia.polarity_scores(text)
        return {}
    except Exception:
        return {}

# Initialize NLTK when module is imported
initialize_nltk()
""")
        
        return True, None
    except Exception as e:
        return False, str(e)

def update_affected_modules() -> Dict[str, Any]:
    """
    Update affected modules to use the new NLTK utilities.
    
    Returns:
        Dictionary with update status for each module
    """
    results = {
        "updated_modules": [],
        "errors": []
    }
    
    # Find affected modules
    affected_modules = []
    for root, _, files in os.walk("cybersec_agents"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "nltk.download('vader_lexicon'" in content:
                            affected_modules.append(file_path)
                except Exception:
                    # Skip files that can't be read
                    pass
    
    # Update each affected module
    for module_path in affected_modules:
        try:
            with open(module_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Replace direct NLTK download calls with utility function
            updated_content = content.replace(
                "nltk.download('vader_lexicon', quiet=True)",
                "from cybersec_agents.utils.nltk_utils import initialize_nltk\ninitialize_nltk(['vader_lexicon'])"
            )
            
            # Replace SentimentIntensityAnalyzer initialization
            if "from nltk.sentiment import SentimentIntensityAnalyzer" in content:
                updated_content = updated_content.replace(
                    "from nltk.sentiment import SentimentIntensityAnalyzer",
                    "from cybersec_agents.utils.nltk_utils import get_sentiment_analyzer"
                )
                updated_content = updated_content.replace(
                    "sia = SentimentIntensityAnalyzer()",
                    "sia = get_sentiment_analyzer()"
                )
            
            # Write updated content
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            
            results["updated_modules"].append(module_path)
        except Exception as e:
            results["errors"].append(f"Error updating {module_path}: {str(e)}")
    
    return results

def run_installation(force: bool = False, quiet: bool = False, data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the complete installation process.
    
    Args:
        force: Whether to force reinstallation
        quiet: Whether to suppress output
        data_dir: Custom NLTK data directory
        
    Returns:
        Dictionary with installation results
    """
    results = {}
    
    # Check NLTK installation
    if not quiet:
        print_section("Checking NLTK Installation")
    
    nltk_installed, nltk_version, nltk_error = check_nltk_installation()
    results["nltk_installation"] = {
        "installed": nltk_installed,
        "version": nltk_version,
        "error": nltk_error
    }
    
    if nltk_installed and not force:
        if not quiet:
            print_success(f"NLTK is already installed (version {nltk_version})")
    else:
        # Install NLTK
        if not quiet:
            if force:
                print_info("Forcing NLTK reinstallation")
            else:
                print_info("NLTK is not installed, installing now")
        
        success, error = install_nltk(quiet)
        results["nltk_installation"]["installed"] = success
        results["nltk_installation"]["error"] = error
        
        if not success:
            if not quiet:
                print_error(f"Failed to install NLTK: {error}")
            return results
    
    # Download required NLTK data
    if not quiet:
        print_section("Installing NLTK Data Packages")
    
    packages = ["vader_lexicon", "punkt", "stopwords"]
    results["nltk_data"] = {}
    
    for package in packages:
        if not quiet:
            print_info(f"Processing package: {package}")
        
        # Check if package is already installed
        if verify_nltk_data(package) and not force:
            if not quiet:
                print_success(f"Package '{package}' is already installed")
            results["nltk_data"][package] = {
                "installed": True,
                "method": "already_installed"
            }
            continue
        
        # Try standard download
        success, error = download_nltk_data(package, data_dir, quiet)
        if success:
            results["nltk_data"][package] = {
                "installed": True,
                "method": "standard_download"
            }
            continue
        
        # Try manual download if standard download failed
        if not quiet:
            print_warning(f"Standard download failed for {package}, trying manual download")
        
        # Determine target directory
        target_dir = data_dir
        if not target_dir:
            target_dir = find_writable_directory()
            if not target_dir:
                home_dir = os.path.expanduser("~")
                target_dir = os.path.join(home_dir, "nltk_data")
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except:
                    if not quiet:
                        print_error(f"Could not create directory: {target_dir}")
                    results["nltk_data"][package] = {
                        "installed": False,
                        "error": "Could not create directory",
                        "method": "manual_download_failed"
                    }
                    continue
        
        success, error = manual_download_nltk_data(package, target_dir, quiet)
        if success:
            # Add the directory to NLTK's search path
            try:
                import nltk.data
                if target_dir not in nltk.data.path:
                    nltk.data.path.append(target_dir)
            except ImportError:
                pass
            
            # Verify installation
            if verify_nltk_data(package):
                results["nltk_data"][package] = {
                    "installed": True,
                    "method": "manual_download"
                }
                if not quiet:
                    print_success(f"Manual download successful for {package}")
            else:
                results["nltk_data"][package] = {
                    "installed": False,
                    "error": "Manual download succeeded but verification failed",
                    "method": "manual_download_failed"
                }
                if not quiet:
                    print_error(f"Manual download succeeded but verification failed for {package}")
        else:
            results["nltk_data"][package] = {
                "installed": False,
                "error": error,
                "method": "manual_download_failed"
            }
            if not quiet:
                print_error(f"Manual download failed for {package}: {error}")
    
    # Verify SentimentIntensityAnalyzer
    if not quiet:
        print_section("Verifying SentimentIntensityAnalyzer")
    
    sia_works = verify_sentiment_analyzer()
    results["sentiment_analyzer"] = {
        "working": sia_works
    }
    
    if sia_works:
        if not quiet:
            print_success("SentimentIntensityAnalyzer is working correctly")
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
    results["nltk_utils"] = {
        "created": success,
        "error": error
    }
    
    if success:
        if not quiet:
            print_success("Created NLTK utilities module")
    else:
        if not quiet:
            print_error(f"Failed to create NLTK utilities module: {error}")
    
    # Update affected modules
    if not quiet:
        print_section("Updating Affected Modules")
    
    module_updates = update_affected_modules()
    results["module_updates"] = module_updates
    
    if module_updates["updated_modules"]:
        if not quiet:
            print_success(f"Updated {len(module_updates['updated_modules'])} modules to use NLTK utilities")
            for module in module_updates["updated_modules"]:
                print_info(f"  - {module}")
    else:
        if not quiet:
            print_info("No modules needed updating")
    
    if module_updates["errors"]:
        if not quiet:
            print_warning(f"Encountered {len(module_updates['errors'])} errors while updating modules")
            for error in module_updates["errors"]:
                print_error(f"  - {error}")
    
    return results

def main():
    """Main function to run the installation script."""
    parser = argparse.ArgumentParser(description="NLTK Dependencies Installer")
    parser.add_argument("--force", action="store_true", help="Force reinstallation even if already installed")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--data-dir", type=str, help="Specify custom NLTK data directory")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, help="Save results to a file")
    args = parser.parse_args()
    
    if not args.quiet:
        print_header("NLTK Dependencies Installer")
        print_info(f"Python version: {platform.python_version()}")
        print_info(f"Platform: {platform.platform()}")
        print_info(f"System: {platform.system()} {platform.release()}")
    
    results = run_installation(args.force, args.quiet, args.data_dir)
    
    # Print summary
    if not args.quiet:
        print_section("Installation Summary")
        
        nltk_ok = results["nltk_installation"]["installed"]
        vader_ok = results["nltk_data"].get("vader_lexicon", {}).get("installed", False)
        sia_ok = results["sentiment_analyzer"]["working"]
        
        if nltk_ok and vader_ok and sia_ok:
            print_success("Installation successful! NLTK and VADER lexicon are working correctly.")
        elif nltk_ok and not vader_ok:
            print_warning("NLTK is installed but VADER lexicon installation failed.")
        elif not nltk_ok:
            print_error("NLTK installation failed.")
        elif vader_ok and not sia_ok:
            print_warning("VADER lexicon is installed but SentimentIntensityAnalyzer is not working.")
    
    # Output as JSON if requested
    if args.json:
        json_results = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_results)
            if not args.quiet:
                print_info(f"Results saved to {args.output}")
        else:
            print(json_results)
    elif args.output:
        with open(args.output, "w") as f:
            f.write("NLTK Dependencies Installation Results\n")
            f.write("====================================\n\n")
            
            f.write("NLTK Installation:\n")
            f.write(f"  Installed: {results['nltk_installation']['installed']}\n")
            if results['nltk_installation']['version']:
                f.write(f"  Version: {results['nltk_installation']['version']}\n")
            
            f.write("\nNLTK Data Packages:\n")
            for package, status in results['nltk_data'].items():
                f.write(f"  {package}: {status['installed']}\n")
                if not status['installed'] and status.get('error'):
                    f.write(f"    Error: {status['error']}\n")
            
            f.write("\nSentimentIntensityAnalyzer:\n")
            f.write(f"  Working: {results['sentiment_analyzer']['working']}\n")
            
            f.write("\nProject Updates:\n")
            f.write(f"  Updated requirements.txt: {results['dependency_updates']['requirements.txt']}\n")
            f.write(f"  Updated setup.py: {results['dependency_updates']['setup.py']}\n")
            f.write(f"  Created NLTK utilities: {results['nltk_utils']['created']}\n")
            f.write(f"  Updated modules: {len(results['module_updates']['updated_modules'])}\n")
            
            if not args.quiet:
                print_info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()