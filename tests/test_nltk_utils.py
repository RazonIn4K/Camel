"""
Tests for NLTK utilities.

This module contains tests for the NLTK utilities in the cybersec_agents package.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cybersec_agents.utils.nltk_utils import (
    NLTKConfig,
    initialize_nltk,
    ensure_vader_lexicon,
    get_sentiment_analyzer,
    analyze_sentiment,
    get_nltk_info
)


class TestNLTKConfig(unittest.TestCase):
    """Tests for the NLTKConfig class."""
    
    def test_get_default_data_dirs(self):
        """Test that get_default_data_dirs returns a list of directories."""
        dirs = NLTKConfig.get_default_data_dirs()
        self.assertIsInstance(dirs, list)
        self.assertTrue(len(dirs) > 0)
        
        # Check that home directory is included
        home_dir = os.path.expanduser("~")
        home_nltk_dir = os.path.join(home_dir, "nltk_data")
        self.assertIn(home_nltk_dir, dirs)
    
    def test_get_writable_data_dir(self):
        """Test that get_writable_data_dir returns a writable directory."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock environment variable
            with patch.dict(os.environ, {"NLTK_DATA": temp_dir}):
                writable_dir = NLTKConfig.get_writable_data_dir()
                self.assertEqual(writable_dir, temp_dir)
        
        # Test with no environment variable
        with patch.dict(os.environ, {}, clear=True):
            writable_dir = NLTKConfig.get_writable_data_dir()
            self.assertIsInstance(writable_dir, str)
            
            # Check that the directory exists and is writable
            self.assertTrue(os.path.exists(writable_dir))
            self.assertTrue(os.access(writable_dir, os.W_OK))


class TestNLTKInitialization(unittest.TestCase):
    """Tests for NLTK initialization functions."""
    
    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_initialize_nltk(self, mock_find, mock_download):
        """Test initialize_nltk function."""
        # Mock nltk.data.find to simulate package not found, then found after download
        mock_find.side_effect = [
            LookupError("Package not found"),  # First call (vader_lexicon)
            None,                              # After download (vader_lexicon)
            LookupError("Package not found"),  # First call (punkt)
            None,                              # After download (punkt)
            LookupError("Package not found"),  # First call (stopwords)
            None                               # After download (stopwords)
        ]
        
        # Call initialize_nltk
        results = initialize_nltk(["vader_lexicon", "punkt", "stopwords"])
        
        # Check that download was called for each package
        self.assertEqual(mock_download.call_count, 3)
        
        # Check results
        self.assertEqual(results, {
            "vader_lexicon": True,
            "punkt": True,
            "stopwords": True
        })
    
    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_initialize_nltk_already_downloaded(self, mock_find, mock_download):
        """Test initialize_nltk when packages are already downloaded."""
        # Mock nltk.data.find to simulate packages already found
        mock_find.return_value = None
        
        # Call initialize_nltk
        results = initialize_nltk(["vader_lexicon", "punkt"])
        
        # Check that download was not called
        mock_download.assert_not_called()
        
        # Check results
        self.assertEqual(results, {
            "vader_lexicon": True,
            "punkt": True
        })
    
    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_initialize_nltk_download_failure(self, mock_find, mock_download):
        """Test initialize_nltk when download fails."""
        # Mock nltk.data.find to simulate package not found
        mock_find.side_effect = LookupError("Package not found")
        
        # Mock nltk.download to simulate download failure
        mock_download.side_effect = Exception("Download failed")
        
        # Call initialize_nltk
        results = initialize_nltk(["vader_lexicon"])
        
        # Check results
        self.assertEqual(results, {
            "vader_lexicon": False
        })
    
    @patch('cybersec_agents.utils.nltk_utils.initialize_nltk')
    @patch('nltk.data.find')
    def test_ensure_vader_lexicon_already_available(self, mock_find, mock_initialize):
        """Test ensure_vader_lexicon when VADER lexicon is already available."""
        # Mock nltk.data.find to simulate VADER lexicon already found
        mock_find.return_value = None
        
        # Call ensure_vader_lexicon
        result = ensure_vader_lexicon()
        
        # Check that initialize_nltk was not called
        mock_initialize.assert_not_called()
        
        # Check result
        self.assertTrue(result)
    
    @patch('cybersec_agents.utils.nltk_utils.initialize_nltk')
    @patch('nltk.data.find')
    def test_ensure_vader_lexicon_download_success(self, mock_find, mock_initialize):
        """Test ensure_vader_lexicon when download succeeds."""
        # Mock nltk.data.find to simulate VADER lexicon not found, then found after download
        mock_find.side_effect = [
            LookupError("Package not found"),  # First call
            None                               # After download
        ]
        
        # Mock initialize_nltk to simulate successful download
        mock_initialize.return_value = {'vader_lexicon': True}
        
        # Call ensure_vader_lexicon
        result = ensure_vader_lexicon()
        
        # Check that initialize_nltk was called
        mock_initialize.assert_called_once_with(['vader_lexicon'])
        
        # Check result
        self.assertTrue(result)
    
    @patch('cybersec_agents.utils.nltk_utils.initialize_nltk')
    @patch('cybersec_agents.utils.nltk_utils.NLTKConfig.get_writable_data_dir')
    @patch('nltk.data.find')
    @patch('builtins.open')
    def test_ensure_vader_lexicon_fallback(self, mock_open, mock_find, mock_get_dir, mock_initialize):
        """Test ensure_vader_lexicon fallback mechanism."""
        # Mock nltk.data.find to simulate VADER lexicon not found, then found after fallback
        mock_find.side_effect = [
            LookupError("Package not found"),  # First call
            LookupError("Package not found"),  # After download attempt
            None                               # After fallback
        ]
        
        # Mock initialize_nltk to simulate download failure
        mock_initialize.return_value = {'vader_lexicon': False}
        
        # Mock get_writable_data_dir to return a directory
        mock_get_dir.return_value = "/tmp/nltk_data"
        
        # Mock os.makedirs to avoid creating directories
        with patch('os.makedirs'):
            # Call ensure_vader_lexicon
            result = ensure_vader_lexicon()
        
        # Check that initialize_nltk was called
        mock_initialize.assert_called_once_with(['vader_lexicon'])
        
        # Check that open was called (to create minimal lexicon)
        mock_open.assert_called_once()
        
        # Check result
        self.assertTrue(result)


class TestSentimentAnalysis(unittest.TestCase):
    """Tests for sentiment analysis functions."""
    
    @patch('cybersec_agents.utils.nltk_utils.ensure_vader_lexicon')
    @patch('nltk.sentiment.SentimentIntensityAnalyzer')
    def test_get_sentiment_analyzer(self, mock_sia, mock_ensure):
        """Test get_sentiment_analyzer function."""
        # Mock ensure_vader_lexicon to return True
        mock_ensure.return_value = True
        
        # Mock SentimentIntensityAnalyzer
        mock_sia_instance = MagicMock()
        mock_sia.return_value = mock_sia_instance
        
        # Call get_sentiment_analyzer
        result = get_sentiment_analyzer()
        
        # Check that ensure_vader_lexicon was called
        mock_ensure.assert_called_once()
        
        # Check that SentimentIntensityAnalyzer was instantiated
        mock_sia.assert_called_once()
        
        # Check result
        self.assertEqual(result, mock_sia_instance)
    
    @patch('cybersec_agents.utils.nltk_utils.ensure_vader_lexicon')
    def test_get_sentiment_analyzer_failure(self, mock_ensure):
        """Test get_sentiment_analyzer when VADER lexicon is not available."""
        # Mock ensure_vader_lexicon to return False
        mock_ensure.return_value = False
        
        # Call get_sentiment_analyzer
        result = get_sentiment_analyzer()
        
        # Check that ensure_vader_lexicon was called
        mock_ensure.assert_called_once()
        
        # Check result
        self.assertIsNone(result)
    
    @patch('cybersec_agents.utils.nltk_utils.get_sentiment_analyzer')
    def test_analyze_sentiment(self, mock_get_analyzer):
        """Test analyze_sentiment function."""
        # Mock get_sentiment_analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.return_value = {
            'neg': 0.0,
            'neu': 0.5,
            'pos': 0.5,
            'compound': 0.5
        }
        mock_get_analyzer.return_value = mock_analyzer
        
        # Call analyze_sentiment
        result = analyze_sentiment("This is a test.")
        
        # Check that get_sentiment_analyzer was called
        mock_get_analyzer.assert_called_once()
        
        # Check that polarity_scores was called
        mock_analyzer.polarity_scores.assert_called_once_with("This is a test.")
        
        # Check result
        self.assertEqual(result, {
            'neg': 0.0,
            'neu': 0.5,
            'pos': 0.5,
            'compound': 0.5
        })
    
    @patch('cybersec_agents.utils.nltk_utils.get_sentiment_analyzer')
    def test_analyze_sentiment_no_analyzer(self, mock_get_analyzer):
        """Test analyze_sentiment when analyzer is not available."""
        # Mock get_sentiment_analyzer to return None
        mock_get_analyzer.return_value = None
        
        # Call analyze_sentiment
        result = analyze_sentiment("This is a test.")
        
        # Check that get_sentiment_analyzer was called
        mock_get_analyzer.assert_called_once()
        
        # Check result
        self.assertEqual(result, {})
    
    @patch('cybersec_agents.utils.nltk_utils.get_sentiment_analyzer')
    def test_analyze_sentiment_exception(self, mock_get_analyzer):
        """Test analyze_sentiment when an exception occurs."""
        # Mock get_sentiment_analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.side_effect = Exception("Test exception")
        mock_get_analyzer.return_value = mock_analyzer
        
        # Call analyze_sentiment
        result = analyze_sentiment("This is a test.")
        
        # Check that get_sentiment_analyzer was called
        mock_get_analyzer.assert_called_once()
        
        # Check that polarity_scores was called
        mock_analyzer.polarity_scores.assert_called_once_with("This is a test.")
        
        # Check result
        self.assertEqual(result, {})


class TestNLTKInfo(unittest.TestCase):
    """Tests for get_nltk_info function."""
    
    @patch('nltk.data.find')
    @patch('nltk.sentiment.SentimentIntensityAnalyzer')
    def test_get_nltk_info(self, mock_sia, mock_find):
        """Test get_nltk_info function."""
        # Mock nltk.data.find to simulate packages found
        mock_find.return_value = None
        
        # Mock SentimentIntensityAnalyzer
        mock_sia_instance = MagicMock()
        mock_sia_instance.polarity_scores.return_value = {}
        mock_sia.return_value = mock_sia_instance
        
        # Mock nltk module
        with patch.dict('sys.modules', {
            'nltk': MagicMock(
                __version__='3.8.1',
                data=MagicMock(
                    path=['/usr/share/nltk_data', '/home/user/nltk_data']
                )
            )
        }):
            # Call get_nltk_info
            result = get_nltk_info()
        
        # Check result
        self.assertTrue(result['installed'])
        self.assertEqual(result['version'], '3.8.1')
        self.assertEqual(result['data_path'], ['/usr/share/nltk_data', '/home/user/nltk_data'])
        self.assertTrue(result['packages']['vader_lexicon'])
        self.assertTrue(result['vader_lexicon'])
        self.assertTrue(result['sentiment_analyzer'])
    
    def test_get_nltk_info_not_installed(self):
        """Test get_nltk_info when NLTK is not installed."""
        # Mock ImportError when importing nltk
        with patch.dict('sys.modules', {'nltk': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                # Call get_nltk_info
                result = get_nltk_info()
        
        # Check result
        self.assertFalse(result['installed'])
        self.assertIsNone(result['version'])
        self.assertEqual(result['data_path'], [])
        self.assertEqual(result['packages'], {})
        self.assertFalse(result['vader_lexicon'])
        self.assertFalse(result['sentiment_analyzer'])


if __name__ == '__main__':
    unittest.main()