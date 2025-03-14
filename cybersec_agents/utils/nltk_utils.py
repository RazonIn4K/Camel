"""
NLTK utilities for cybersec_agents.

This module provides utilities for initializing NLTK and downloading required data,
with a focus on robust handling of VADER lexicon for sentiment analysis.
"""

import os
import sys
import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Set up logger
logger = logging.getLogger(__name__)

class NLTKConfig:
    """Configuration for NLTK data and paths."""
    
    # Default packages needed by the project
    DEFAULT_PACKAGES: list[Any] = ["vader_lexicon", "punkt", "stopwords"]
    
    # Environment variable for custom NLTK data path
    ENV_VAR: str = "NLTK_DATA"
    
    @staticmethod
    def get_default_data_dirs() -> List[str]:
        """
        Get default NLTK data directories for the current platform.
        
        Returns:
            List of directory paths
        """
        home_dir = os.path.expanduser("~")
        
        # Common paths across platforms
        paths: list[Any] = [
            os.path.join(home_dir, "nltk_data"),
            os.path.join(sys.prefix, "nltk_data"),
            os.path.join(os.path.dirname(sys.executable), "nltk_data"),
        ]
        
        # Platform-specific paths
        if platform.system() == "Windows":
            # Windows-specific paths
            appdata = os.environ.get("APPDATA")
            if appdata:
                paths.append(os.path.join(appdata, "nltk_data"))
        elif platform.system() == "Darwin":
            # macOS-specific paths
            paths.append("/usr/local/share/nltk_data")
            paths.append("/usr/share/nltk_data")
        else:
            # Linux/Unix-specific paths
            xdg_data = os.environ.get("XDG_DATA_HOME")
            if xdg_data:
                paths.append(os.path.join(xdg_data, "nltk_data"))
            paths.append("/usr/local/share/nltk_data")
            paths.append("/usr/share/nltk_data")
        
        return paths
    
    @staticmethod
    def get_writable_data_dir() -> Optional[str]:
        """
        Find a writable directory for NLTK data.
        
        Returns:
            Path to a writable directory, or None if none found
        """
        # First, check environment variable
        env_dir = os.environ.get(NLTKConfig.ENV_VAR)
        if env_dir and os.path.isdir(env_dir) and os.access(env_dir, os.W_OK):
            return env_dir
        
        # Try to use NLTK's path if available
        try:
            import nltk.data
            for path in nltk.data.path:
                if os.path.isdir(path) and os.access(path, os.W_OK):
                    return path
        except ImportError:
            pass
        
        # Try default directories
        for directory in NLTKConfig.get_default_data_dirs():
            # Check if directory exists and is writable
            if os.path.isdir(directory) and os.access(directory, os.W_OK):
                return directory
            
            # Check if parent directory exists and is writable
            parent = os.path.dirname(directory)
            if os.path.isdir(parent) and os.access(parent, os.W_OK):
                try:
                    os.makedirs(directory, exist_ok=True)
                    return directory
                except:
                    continue
        
        # Try current directory as last resort
        try:
            cwd = os.getcwd()
            nltk_data_dir = os.path.join(cwd, "nltk_data")
            os.makedirs(nltk_data_dir, exist_ok=True)
            return nltk_data_dir
        except:
            pass
        
        return None


def initialize_nltk(packages: Optional[List[str]] = None, quiet: bool = True) -> Dict[str, bool]:
    """
    Initialize NLTK and download required data packages.
    
    Args:
        packages: List of NLTK packages to download (default: vader_lexicon, punkt, stopwords)
        quiet: Whether to suppress output
        
    Returns:
        Dictionary with status for each package
    """
    if packages is None:
        packages = NLTKConfig.DEFAULT_PACKAGES
    
    results = {pkg: False for pkg in packages}
    
    try:
        import nltk
        
        # Try to find a writable directory
        writable_dir = NLTKConfig.get_writable_data_dir()
        
        # Process each package
        for package in packages:
            # Check if package is already available
            try:
                nltk.data.find(package)
                results[package] = True
                continue
            except LookupError:
                pass
            
            # Try to download the package
            try:
                # Try standard download first
                nltk.download(package, quiet=quiet)
                
                # Verify download
                try:
                    nltk.data.find(package)
                    results[package] = True
                    continue
                except LookupError:
                    pass
                
                # If standard download failed and we have a writable directory, try that
                if writable_dir:
                    try:
                        nltk.download(package, download_dir=writable_dir, quiet=quiet)
                        
                        # Add directory to search path if not already there
                        if writable_dir not in nltk.data.path:
                            nltk.data.path.append(writable_dir)
                        
                        # Verify download
                        try:
                            nltk.data.find(package)
                            results[package] = True
                            continue
                        except LookupError:
                            pass
                    except Exception as e:
                        if not quiet:
                            logger.warning(f"Failed to download {package} to {writable_dir}: {e}")
            except Exception as e:
                if not quiet:
                    logger.warning(f"Failed to download {package}: {e}")
    
    except ImportError as e:
        if not quiet:
            logger.error(f"NLTK import error: {e}")
    
    return results


def ensure_vader_lexicon() -> bool:
    """
    Ensure VADER lexicon is available, with multiple fallback mechanisms.
    
    Returns:
        True if VADER lexicon is available, False otherwise
    """
    try:
        import nltk.data
        
        # Check if VADER lexicon is already available
        try:
            nltk.data.find('vader_lexicon')
            return True
        except LookupError:
            pass
        
        # Try to download VADER lexicon
        result: Any = initialize_nltk(['vader_lexicon'])
        if result.get('vader_lexicon', False):
            return True
        
        # If download failed, try manual installation
        writable_dir = NLTKConfig.get_writable_data_dir()
        if not writable_dir:
            return False
        
        # Create sentiment directory structure
        sentiment_dir = os.path.join(writable_dir, 'sentiment')
        vader_dir = os.path.join(sentiment_dir, 'vader_lexicon')
        os.makedirs(vader_dir, exist_ok=True)
        
        # Create a minimal VADER lexicon file
        # This is a fallback with a subset of the full lexicon
        minimal_lexicon: dict[str, Any] = {
            "good": 1.9,
            "bad": -1.9,
            "excellent": 3.2,
            "terrible": -3.2,
            "happy": 2.1,
            "sad": -1.8,
            "positive": 2.0,
            "negative": -2.0,
            "best": 3.0,
            "worst": -3.0
        }
        
        lexicon_path = os.path.join(vader_dir, 'vader_lexicon.txt')
        try:
            with open(lexicon_path, 'w') as f:
                for word, score in minimal_lexicon.items():
                    f.write(f"{word}\t{score}\n")
            
            # Add directory to search path if not already there
            if writable_dir not in nltk.data.path:
                nltk.data.path.append(writable_dir)
            
            # Verify installation
            try:
                nltk.data.find('vader_lexicon')
                logger.warning("Using minimal VADER lexicon (fallback mode)")
                return True
            except LookupError:
                return False
        except Exception:
            return False
    
    except ImportError:
        return False


def get_sentiment_analyzer():
    """
    Get a SentimentIntensityAnalyzer instance with proper initialization.
    
    Returns:
        SentimentIntensityAnalyzer instance or None if not available
    """
    try:
        # Ensure VADER lexicon is available
        if not ensure_vader_lexicon():
            logger.warning("VADER lexicon is not available")
            return None
        
        # Import and initialize SentimentIntensityAnalyzer
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception as e:
        logger.warning(f"Could not initialize SentimentIntensityAnalyzer: {e}")
        return None


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text with error handling.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores or empty dict if analysis fails
    """
    try:
        sia = get_sentiment_analyzer()
        if sia:
            return sia.polarity_scores(text)
        return {}
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return {}


def get_nltk_info() -> Dict[str, Any]:
    """
    Get information about NLTK installation and configuration.
    
    Returns:
        Dictionary with NLTK information
    """
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "data_path": [],
        "packages": {},
        "vader_lexicon": False,
        "sentiment_analyzer": False
    }
    
    try:
        import nltk
        info["installed"] = True
        info["version"] = nltk.__version__
        info["data_path"] = nltk.data.path
        
        # Check for packages
        for package in NLTKConfig.DEFAULT_PACKAGES:
            try:
                nltk.data.find(package)
                info["packages"][package] = True
            except LookupError:
                info["packages"][package] = False
        
        # Check VADER lexicon specifically
        info["vader_lexicon"] = info["packages"].get("vader_lexicon", False)
        
        # Check SentimentIntensityAnalyzer
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            sia.polarity_scores("Test")
            info["sentiment_analyzer"] = True
        except Exception:
            info["sentiment_analyzer"] = False
    
    except ImportError:
        pass
    
    return info


# Initialize NLTK when module is imported
initialize_nltk()