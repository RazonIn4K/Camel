#!/usr/bin/env python
"""
NLTK Diagnostics Script

This script performs comprehensive diagnostics on NLTK installation and data availability,
with a focus on the VADER lexicon used for sentiment analysis in the Gray Swan Arena framework.

It checks:
1. NLTK installation status
2. NLTK data directory configuration
3. VADER lexicon availability
4. Permissions and write access to NLTK data directories
5. Environment variables affecting NLTK

Usage:
    python scripts/nltk_diagnostics.py [--verbose] [--fix]

Options:
    --verbose   Show detailed diagnostic information
    --fix       Attempt to fix common issues automatically
"""

import os
import sys
import platform
import subprocess
import argparse
import tempfile
import json
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

def check_vader_lexicon() -> Dict[str, Any]:
    """
    Check if VADER lexicon is available and where it's located.
    
    Returns:
        Dictionary with status information
    """
    result = {
        "available": False,
        "location": None,
        "error": None,
        "size": None,
        "download_attempted": False,
        "download_success": False
    }
    
    try:
        import nltk.data
        
        # Check if VADER lexicon is already downloaded
        try:
            vader_path = nltk.data.find('vader_lexicon')
            result["available"] = True
            result["location"] = str(vader_path)
            
            # Get file size
            try:
                result["size"] = os.path.getsize(vader_path)
            except (OSError, IOError):
                result["size"] = "Unknown"
                
        except LookupError as e:
            result["error"] = str(e)
            
            # Try to download VADER lexicon
            try:
                result["download_attempted"] = True
                nltk.download('vader_lexicon', quiet=True)
                
                # Check again after download attempt
                try:
                    vader_path = nltk.data.find('vader_lexicon')
                    result["available"] = True
                    result["location"] = str(vader_path)
                    result["download_success"] = True
                    
                    # Get file size
                    try:
                        result["size"] = os.path.getsize(vader_path)
                    except (OSError, IOError):
                        result["size"] = "Unknown"
                        
                except LookupError as e2:
                    result["error"] = f"After download attempt: {str(e2)}"
                    
            except Exception as download_error:
                result["error"] = f"Download error: {str(download_error)}"
    
    except ImportError as e:
        result["error"] = f"NLTK import error: {str(e)}"
    
    return result

def check_sentiment_analyzer() -> Dict[str, Any]:
    """
    Check if SentimentIntensityAnalyzer can be initialized and used.
    
    Returns:
        Dictionary with status information
    """
    result = {
        "can_import": False,
        "can_initialize": False,
        "can_analyze": False,
        "sample_result": None,
        "error": None
    }
    
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        result["can_import"] = True
        
        try:
            sia = SentimentIntensityAnalyzer()
            result["can_initialize"] = True
            
            try:
                sample_text = "This is a test sentence for sentiment analysis."
                sentiment = sia.polarity_scores(sample_text)
                result["can_analyze"] = True
                result["sample_result"] = sentiment
            except Exception as e:
                result["error"] = f"Analysis error: {str(e)}"
                
        except Exception as e:
            result["error"] = f"Initialization error: {str(e)}"
            
    except ImportError as e:
        result["error"] = f"Import error: {str(e)}"
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result

def check_environment_variables() -> Dict[str, Any]:
    """
    Check environment variables that might affect NLTK.
    
    Returns:
        Dictionary with environment variable information
    """
    env_vars = {
        "NLTK_DATA": os.environ.get("NLTK_DATA"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "HOME": os.environ.get("HOME"),
        "USERPROFILE": os.environ.get("USERPROFILE"),  # Windows
        "APPDATA": os.environ.get("APPDATA"),  # Windows
        "XDG_DATA_HOME": os.environ.get("XDG_DATA_HOME"),  # Linux
    }
    
    return env_vars

def check_disk_space(directory: str) -> Dict[str, Any]:
    """
    Check available disk space in a directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        Dictionary with disk space information
    """
    result = {
        "directory": directory,
        "total_space": None,
        "available_space": None,
        "error": None
    }
    
    try:
        if os.path.exists(directory):
            if platform.system() == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(directory), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes)
                )
                result["total_space"] = total_bytes.value
                result["available_space"] = free_bytes.value
            else:
                stat = os.statvfs(directory)
                result["total_space"] = stat.f_frsize * stat.f_blocks
                result["available_space"] = stat.f_frsize * stat.f_bavail
        else:
            result["error"] = f"Directory {directory} does not exist"
    except Exception as e:
        result["error"] = str(e)
    
    return result

def test_write_permissions(directory: str) -> Dict[str, Any]:
    """
    Test if we can write to a directory by creating a temporary file.
    
    Args:
        directory: Directory to test
        
    Returns:
        Dictionary with test results
    """
    result = {
        "directory": directory,
        "can_write": False,
        "error": None
    }
    
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            result["error"] = f"Could not create directory: {str(e)}"
            return result
    
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=True) as tmp:
            tmp.write(b"test")
            result["can_write"] = True
    except Exception as e:
        result["error"] = str(e)
    
    return result

def check_project_dependencies() -> Dict[str, Any]:
    """
    Check if NLTK is properly listed in project dependencies.
    
    Returns:
        Dictionary with dependency information
    """
    result = {
        "requirements_txt": False,
        "setup_py": False,
        "pyproject_toml": False,
        "errors": []
    }
    
    # Check requirements.txt
    try:
        if os.path.exists("requirements.txt"):
            with open("requirements.txt", "r") as f:
                content = f.read()
                result["requirements_txt"] = "nltk" in content
    except Exception as e:
        result["errors"].append(f"Error checking requirements.txt: {str(e)}")
    
    # Check setup.py
    try:
        if os.path.exists("setup.py"):
            with open("setup.py", "r") as f:
                content = f.read()
                result["setup_py"] = "nltk" in content
    except Exception as e:
        result["errors"].append(f"Error checking setup.py: {str(e)}")
    
    # Check pyproject.toml
    try:
        if os.path.exists("pyproject.toml"):
            with open("pyproject.toml", "r") as f:
                content = f.read()
                result["pyproject_toml"] = "nltk" in content
    except Exception as e:
        result["errors"].append(f"Error checking pyproject.toml: {str(e)}")
    
    return result

def check_affected_modules() -> Dict[str, List[str]]:
    """
    Check which modules in the project use NLTK and VADER.
    
    Returns:
        Dictionary with affected modules
    """
    affected_modules = {
        "nltk_import": [],
        "vader_lexicon": [],
        "sentiment_analyzer": []
    }
    
    for root, _, files in os.walk("cybersec_agents"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "import nltk" in content or "from nltk" in content:
                            affected_modules["nltk_import"].append(file_path)
                        if "vader_lexicon" in content:
                            affected_modules["vader_lexicon"].append(file_path)
                        if "SentimentIntensityAnalyzer" in content:
                            affected_modules["sentiment_analyzer"].append(file_path)
                except Exception:
                    # Skip files that can't be read
                    pass
    
    return affected_modules

def attempt_fix_nltk_installation(verbose: bool = False) -> Dict[str, Any]:
    """
    Attempt to fix NLTK installation issues.
    
    Args:
        verbose: Whether to show verbose output
        
    Returns:
        Dictionary with fix results
    """
    result = {
        "nltk_installed": False,
        "vader_downloaded": False,
        "errors": []
    }
    
    # 1. Install NLTK if not already installed
    try:
        import nltk
        result["nltk_installed"] = True
        if verbose:
            print_info(f"NLTK is already installed (version {nltk.__version__})")
    except ImportError:
        try:
            if verbose:
                print_info("Installing NLTK...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            try:
                import nltk
                result["nltk_installed"] = True
                if verbose:
                    print_success(f"Successfully installed NLTK (version {nltk.__version__})")
            except ImportError as e:
                result["errors"].append(f"Failed to import NLTK after installation: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Failed to install NLTK: {str(e)}")
    
    # 2. Try to download VADER lexicon
    if result["nltk_installed"]:
        try:
            import nltk.data
            
            # Check if VADER lexicon is already downloaded
            try:
                nltk.data.find('vader_lexicon')
                result["vader_downloaded"] = True
                if verbose:
                    print_info("VADER lexicon is already downloaded")
            except LookupError:
                # Try to download VADER lexicon
                try:
                    if verbose:
                        print_info("Downloading VADER lexicon...")
                    nltk.download('vader_lexicon')
                    
                    # Check if download was successful
                    try:
                        nltk.data.find('vader_lexicon')
                        result["vader_downloaded"] = True
                        if verbose:
                            print_success("Successfully downloaded VADER lexicon")
                    except LookupError as e:
                        result["errors"].append(f"Failed to find VADER lexicon after download: {str(e)}")
                except Exception as e:
                    result["errors"].append(f"Failed to download VADER lexicon: {str(e)}")
        except ImportError as e:
            result["errors"].append(f"Failed to import nltk.data: {str(e)}")
    
    # 3. Try alternative download methods if needed
    if result["nltk_installed"] and not result["vader_downloaded"]:
        # Try downloading to a custom location
        try:
            home_dir = os.path.expanduser("~")
            nltk_data_dir = os.path.join(home_dir, "nltk_data")
            
            if not os.path.exists(nltk_data_dir):
                os.makedirs(nltk_data_dir, exist_ok=True)
            
            if verbose:
                print_info(f"Trying to download VADER lexicon to {nltk_data_dir}...")
            
            nltk.download('vader_lexicon', download_dir=nltk_data_dir)
            
            # Add the custom directory to NLTK's search path
            nltk.data.path.append(nltk_data_dir)
            
            # Check if download was successful
            try:
                nltk.data.find('vader_lexicon')
                result["vader_downloaded"] = True
                if verbose:
                    print_success(f"Successfully downloaded VADER lexicon to {nltk_data_dir}")
            except LookupError as e:
                result["errors"].append(f"Failed to find VADER lexicon after custom download: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Failed to download VADER lexicon to custom location: {str(e)}")
    
    return result

def run_diagnostics(verbose: bool = False, fix: bool = False) -> Dict[str, Any]:
    """
    Run all diagnostic checks and return results.
    
    Args:
        verbose: Whether to show verbose output
        fix: Whether to attempt to fix issues
        
    Returns:
        Dictionary with all diagnostic results
    """
    results = {}
    
    # Check NLTK installation
    print_section("Checking NLTK Installation")
    nltk_installed, nltk_version, nltk_error = check_nltk_installation()
    results["nltk_installation"] = {
        "installed": nltk_installed,
        "version": nltk_version,
        "error": nltk_error
    }
    
    if nltk_installed:
        print_success(f"NLTK is installed (version {nltk_version})")
    else:
        print_error(f"NLTK is not installed: {nltk_error}")
    
    # Check NLTK data directories
    print_section("Checking NLTK Data Directories")
    nltk_data_dirs = get_nltk_data_dirs()
    results["nltk_data_dirs"] = []
    
    for directory in nltk_data_dirs:
        dir_perms = check_directory_permissions(directory)
        results["nltk_data_dirs"].append({
            "directory": directory,
            "permissions": dir_perms
        })
        
        if dir_perms["exists"]:
            if dir_perms["readable"] and dir_perms["writable"]:
                print_success(f"Directory exists with read/write permissions: {directory}")
            elif dir_perms["readable"]:
                print_warning(f"Directory exists with read-only permissions: {directory}")
            else:
                print_error(f"Directory exists but is not readable: {directory}")
        else:
            if dir_perms["can_create"]:
                print_warning(f"Directory does not exist but can be created: {directory}")
            else:
                print_error(f"Directory does not exist and cannot be created: {directory}")
    
    # Check VADER lexicon
    print_section("Checking VADER Lexicon")
    vader_result = check_vader_lexicon()
    results["vader_lexicon"] = vader_result
    
    if vader_result["available"]:
        print_success(f"VADER lexicon is available at: {vader_result['location']}")
        if verbose and vader_result["size"]:
            print_info(f"Lexicon size: {vader_result['size']} bytes")
    else:
        print_error(f"VADER lexicon is not available: {vader_result['error']}")
        if vader_result["download_attempted"]:
            if vader_result["download_success"]:
                print_success("Automatic download was successful")
            else:
                print_error("Automatic download failed")
    
    # Check SentimentIntensityAnalyzer
    print_section("Checking SentimentIntensityAnalyzer")
    sia_result = check_sentiment_analyzer()
    results["sentiment_analyzer"] = sia_result
    
    if sia_result["can_analyze"]:
        print_success("SentimentIntensityAnalyzer is working correctly")
        if verbose and sia_result["sample_result"]:
            print_info(f"Sample analysis result: {sia_result['sample_result']}")
    elif sia_result["can_initialize"]:
        print_error(f"SentimentIntensityAnalyzer initialized but analysis failed: {sia_result['error']}")
    elif sia_result["can_import"]:
        print_error(f"SentimentIntensityAnalyzer import succeeded but initialization failed: {sia_result['error']}")
    else:
        print_error(f"SentimentIntensityAnalyzer import failed: {sia_result['error']}")
    
    # Check environment variables
    print_section("Checking Environment Variables")
    env_vars = check_environment_variables()
    results["environment_variables"] = env_vars
    
    for var, value in env_vars.items():
        if value:
            print_info(f"{var} = {value}")
    
    # Check disk space
    print_section("Checking Disk Space")
    home_dir = os.path.expanduser("~")
    nltk_data_dir = os.path.join(home_dir, "nltk_data")
    disk_space = check_disk_space(nltk_data_dir)
    results["disk_space"] = disk_space
    
    if disk_space["available_space"]:
        available_mb = disk_space["available_space"] / (1024 * 1024)
        print_info(f"Available disk space: {available_mb:.2f} MB")
        if available_mb < 10:
            print_warning("Low disk space might cause download issues")
    elif disk_space["error"]:
        print_error(f"Could not check disk space: {disk_space['error']}")
    
    # Check write permissions
    print_section("Testing Write Permissions")
    write_test_results = []
    for directory in nltk_data_dirs:
        if os.path.exists(directory) or os.path.exists(os.path.dirname(directory)):
            write_test = test_write_permissions(directory)
            write_test_results.append(write_test)
            
            if write_test["can_write"]:
                print_success(f"Can write to directory: {directory}")
            else:
                print_error(f"Cannot write to directory: {directory}")
                if verbose and write_test["error"]:
                    print_info(f"Error: {write_test['error']}")
    
    results["write_tests"] = write_test_results
    
    # Check project dependencies
    print_section("Checking Project Dependencies")
    dependencies = check_project_dependencies()
    results["project_dependencies"] = dependencies
    
    if dependencies["requirements_txt"]:
        print_success("NLTK is listed in requirements.txt")
    else:
        print_warning("NLTK is not listed in requirements.txt")
    
    if dependencies["setup_py"]:
        print_success("NLTK is listed in setup.py")
    else:
        print_warning("NLTK is not listed in setup.py")
    
    if dependencies["pyproject_toml"]:
        print_success("NLTK is listed in pyproject.toml")
    else:
        print_warning("NLTK is not listed in pyproject.toml")
    
    # Check affected modules
    print_section("Checking Affected Modules")
    affected_modules = check_affected_modules()
    results["affected_modules"] = affected_modules
    
    print_info(f"Modules importing NLTK: {len(affected_modules['nltk_import'])}")
    print_info(f"Modules using VADER lexicon: {len(affected_modules['vader_lexicon'])}")
    print_info(f"Modules using SentimentIntensityAnalyzer: {len(affected_modules['sentiment_analyzer'])}")
    
    if verbose:
        if affected_modules['vader_lexicon']:
            print("\nModules using VADER lexicon:")
            for module in affected_modules['vader_lexicon']:
                print(f"  - {module}")
    
    # Attempt to fix issues if requested
    if fix:
        print_section("Attempting to Fix Issues")
        fix_results = attempt_fix_nltk_installation(verbose)
        results["fix_results"] = fix_results
        
        if fix_results["nltk_installed"] and fix_results["vader_downloaded"]:
            print_success("Successfully fixed NLTK and VADER lexicon installation")
        elif fix_results["nltk_installed"]:
            print_warning("Installed NLTK but could not download VADER lexicon")
            for error in fix_results["errors"]:
                print_error(f"Error: {error}")
        else:
            print_error("Could not fix NLTK installation")
            for error in fix_results["errors"]:
                print_error(f"Error: {error}")
    
    return results

def main():
    """Main function to run the diagnostics script."""
    parser = argparse.ArgumentParser(description="NLTK Diagnostics Script")
    parser.add_argument("--verbose", action="store_true", help="Show detailed diagnostic information")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues automatically")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, help="Save results to a file")
    args = parser.parse_args()
    
    print_header("NLTK Diagnostics")
    print_info(f"Python version: {platform.python_version()}")
    print_info(f"Platform: {platform.platform()}")
    print_info(f"System: {platform.system()} {platform.release()}")
    
    results = run_diagnostics(args.verbose, args.fix)
    
    # Print summary
    print_section("Summary")
    
    nltk_ok = results["nltk_installation"]["installed"]
    vader_ok = results["vader_lexicon"]["available"]
    sia_ok = results["sentiment_analyzer"]["can_analyze"]
    
    if nltk_ok and vader_ok and sia_ok:
        print_success("All checks passed! NLTK and VADER lexicon are working correctly.")
    elif nltk_ok and not vader_ok:
        print_warning("NLTK is installed but VADER lexicon is not available.")
        print_info("Run with --fix to attempt automatic download of VADER lexicon.")
    elif not nltk_ok:
        print_error("NLTK is not installed.")
        print_info("Run with --fix to attempt automatic installation of NLTK.")
    elif vader_ok and not sia_ok:
        print_warning("VADER lexicon is available but SentimentIntensityAnalyzer is not working.")
    
    # Output as JSON if requested
    if args.json:
        json_results = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_results)
            print_info(f"Results saved to {args.output}")
        else:
            print(json_results)
    elif args.output:
        with open(args.output, "w") as f:
            f.write("NLTK Diagnostics Results\n")
            f.write("=======================\n\n")
            
            f.write("NLTK Installation:\n")
            f.write(f"  Installed: {results['nltk_installation']['installed']}\n")
            f.write(f"  Version: {results['nltk_installation']['version']}\n")
            
            f.write("\nVADER Lexicon:\n")
            f.write(f"  Available: {results['vader_lexicon']['available']}\n")
            if results['vader_lexicon']['location']:
                f.write(f"  Location: {results['vader_lexicon']['location']}\n")
            
            f.write("\nSentimentIntensityAnalyzer:\n")
            f.write(f"  Working: {results['sentiment_analyzer']['can_analyze']}\n")
            
            f.write("\nAffected Modules:\n")
            f.write(f"  Using NLTK: {len(results['affected_modules']['nltk_import'])}\n")
            f.write(f"  Using VADER: {len(results['affected_modules']['vader_lexicon'])}\n")
            f.write(f"  Using SentimentIntensityAnalyzer: {len(results['affected_modules']['sentiment_analyzer'])}\n")
            
            print_info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()