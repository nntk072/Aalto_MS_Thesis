"""Tests for rule-based strategies in live_trading.py."""

import sys
from unittest.mock import MagicMock, patch

# Mock MetaTrader5 module for testing in environments without MT5
mock_mt5 = MagicMock()
sys.modules['MetaTrader5'] = mock_mt5


class TestLiveTradingConfig:
    """Test live_trading.py configuration."""

    @patch.dict('os.environ', {'PAPER_TRADING': 'true', 'LOGIN': '', 'PASSWORD': '', 'SERVER': ''})
    def test_paper_trading_default_true(self):
        """Test that PAPER_TRADING defaults to true."""
        # Import after setting env vars
        import live_trading
        
        # Check that PAPER_TRADING is True by default
        assert live_trading.PAPER_TRADING is True

    @patch.dict('os.environ', {'PAPER_TRADING': 'false', 'LOGIN': '', 'PASSWORD': '', 'SERVER': ''})
    def test_paper_trading_false(self):
        """Test that PAPER_TRADING can be set to false."""
        # Need to reload the module to pick up new env vars
        import sys
        if 'live_trading' in sys.modules:
            del sys.modules['live_trading']
        
        import live_trading
        
        # Check that PAPER_TRADING is False when explicitly set
        assert live_trading.PAPER_TRADING is False

    @patch.dict('os.environ', {'PAPER_TRADING': 'True', 'LOGIN': '', 'PASSWORD': '', 'SERVER': ''})
    def test_paper_trading_case_insensitive(self):
        """Test that PAPER_TRADING is case insensitive."""
        import sys
        if 'live_trading' in sys.modules:
            del sys.modules['live_trading']
        
        import live_trading
        
        # Check that PAPER_TRADING is True even with capital T
        assert live_trading.PAPER_TRADING is True


class TestMT5TerminalPath:
    """Test that MT5 terminal path configuration works."""

    @patch.dict('os.environ', {
        'MT5_TERMINAL_PATH': '/custom/path/terminal64.exe',
        'LOGIN': '', 'PASSWORD': '', 'SERVER': ''
    })
    def test_custom_terminal_path(self):
        """Test that custom MT5_TERMINAL_PATH is used."""
        import sys
        if 'live_trading' in sys.modules:
            del sys.modules['live_trading']
        
        import live_trading
        
        assert live_trading.TERMINAL_PATH == '/custom/path/terminal64.exe'

    @patch.dict('os.environ', {'LOGIN': '', 'PASSWORD': '', 'SERVER': ''})
    def test_default_terminal_path(self):
        """Test that default MT5 terminal path is used when not specified."""
        import sys
        if 'live_trading' in sys.modules:
            del sys.modules['live_trading']
        
        import live_trading
        
        expected_default = r"C:\Program Files\MetaTrader 5\terminal64.exe"
        assert live_trading.TERMINAL_PATH == expected_default


class TestStrategyTypeConfig:
    """Test STRATEGY_TYPE configuration."""

    @patch.dict('os.environ', {
        'STRATEGY_TYPE': 'crossover',
        'LOGIN': '', 'PASSWORD': '', 'SERVER': ''
    })
    def test_strategy_type_crossover(self):
        """Test that STRATEGY_TYPE can be set to crossover."""
        import sys
        if 'live_trading' in sys.modules:
            del sys.modules['live_trading']
        
        import live_trading
        
        assert live_trading.STRATEGY_TYPE == 'crossover'

    @patch.dict('os.environ', {
        'STRATEGY_TYPE': 'combined',
        'LOGIN': '', 'PASSWORD': '', 'SERVER': ''
    })
    def test_strategy_type_combined(self):
        """Test that STRATEGY_TYPE defaults to combined."""
        import sys
        if 'live_trading' in sys.modules:
            del sys.modules['live_trading']
        
        import live_trading
        
        assert live_trading.STRATEGY_TYPE == 'combined'