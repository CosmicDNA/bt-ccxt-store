import time
import unittest
import os
from datetime import datetime, timezone
from unittest.mock import patch

from backtrader import Strategy, Cerebro, TimeFrame
from ccxt.base.errors import ExchangeError

from ccxtbt import CCXTFeed, CCXTStore


class TestFeedInitialFetchBalance(unittest.TestCase):
    """
    At least at Binance and probably on other exchanges too fetching ohlcv data doesn't need authentication
    while obviously fetching the balance of ones account does need authentication.
    Usually the CCXTStore fetches the balance when it is initialized which is not a problem during live trading
    operation.
    But the store is also initialized when the CCXTFeed is created and used during unit testing and backtesting.
    For this case it is beneficial to turn off the initial fetching of the balance as it is not really needed and
    it avoids needing to have api keys.
    This makes it possible for users that don't have a Binance api key to run backtesting and unit tests with real
    ohlcv data to try out this lib.
    """

    def mock_binance_markets(self):
        # A minimal mock for markets needed by the tests
        return {
            'BNB/USDT': {
                'id': 'BNBUSDT', 'symbol': 'BNB/USDT', 'base': 'BNB', 'quote': 'USDT',
                'baseId': 'BNB', 'quoteId': 'USDT', 'active': True, 'type': 'spot',
                'linear': None, 'inverse': None, 'spot': True, 'swap': False, 'future': False,
                'option': False, 'margin': True, 'contract': False, 'contractSize': None,
                'expiry': None, 'expiryDatetime': None, 'optionType': None, 'strike': None,
                'settle': None, 'settleId': None, 'precision': {}, 'limits': {}, 'info': {},
            }
        }

    def setUp(self):
        """
        The initial balance is fetched in the context of the initialization of the CCXTStore.
        But as the CCXTStore is a singleton it's normally initialized only once and the instance is reused
        causing side effects.
        If the  first test run initializes the store without fetching the balance a subsequent test run
        would not try to fetch the balance again as the initialization won't happen again.
        Resetting the singleton to None here causes the initialization of the store to happen in every test method.
        """
        CCXTStore._singleton = None

    @patch("ccxt.binance.binance.fetch_balance")
    def test_fetch_balance_throws_error(self, fetch_balance_mock):
        """
        If API keys are provided the store is expected to fetch the balance.
        If fetch_balance fails (as simulated by the mock), an error should be raised.
        """

        config = {
            "apiKey": "an-api-key",  # Changed to camelCase 'apiKey'
            "secret": "an-api-secret",
            "enableRateLimit": True,
            "nonce": lambda: str(int(time.time() * 1000)),
        }

        fetch_markets_patcher = None
        # Only mock fetch_markets in CI environment to avoid geo-blocking issues
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            fetch_markets_patcher = patch("ccxt.binance.binance.fetch_markets")
            mock_markets = fetch_markets_patcher.start()
            mock_markets.return_value = self.mock_binance_markets()
            self.addCleanup(fetch_markets_patcher.stop)

        # Simulate an error during fetch_balance
        fetch_balance_mock.side_effect = ExchangeError("API call to fetch_balance failed")

        with self.assertRaises(ExchangeError):
            backtesting(config)

        fetch_balance_mock.assert_called_once()

    def test_default_fetch_balance_param(self):
        """
        If API keys are NOT provided the store is expected to
        not fetch the balance and load the ohlcv data without them.
        """
        config = {
            "enableRateLimit": True,
            "nonce": lambda: str(int(time.time() * 1000)),
        }
        fetch_markets_patcher = None
        fetch_ohlcv_patcher = None

        # Only mock network calls in CI environment
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            fetch_markets_patcher = patch("ccxt.binance.binance.fetch_markets")
            mock_markets = fetch_markets_patcher.start()
            mock_markets.return_value = self.mock_binance_markets()
            self.addCleanup(fetch_markets_patcher.stop)

            fetch_ohlcv_patcher = patch("ccxt.binance.binance.fetch_ohlcv")
            mock_ohlcv = fetch_ohlcv_patcher.start()
            # Mock a couple of kline entries. The structure is:
            # [timestamp, open, high, low, close, volume]
            # Timestamps should align with what backtesting() function expects.
            mock_ohlcv.return_value = [
                [1546300800000, 10, 12, 9, 11, 100],  # 2019-01-01 00:00:00 UTC
                [1546300860000, 11, 13, 10, 12, 150], # 2019-01-01 00:01:00 UTC
                # Add a third one to satisfy the next_runs = 3 assertion
                [1546300920000, 12, 14, 11, 13, 200], # 2019-01-01 00:02:00 UTC
            ]
            self.addCleanup(fetch_ohlcv_patcher.stop)

        finished_strategies = backtesting(config)
        self.assertEqual(len(finished_strategies), 1)
        self.assertEqual(finished_strategies[0].next_runs, 3)


class _TestStrategy(Strategy):
    def __init__(self):
        self.next_runs = 0

    def next(self, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print("%s closing price: %s" % (dt.isoformat(), self.datas[0].close[0]))
        self.next_runs += 1


def backtesting(config):
    cerebro = Cerebro()

    cerebro.addstrategy(_TestStrategy)

    cerebro.adddata(
        CCXTFeed(
            exchange="binance",
            dataname="BNB/USDT",
            timeframe=TimeFrame.Minutes,
            fromdate=datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            todate=datetime(2019, 1, 1, 0, 2, tzinfo=timezone.utc),
            compression=1,
            ohlcv_limit=2,
            currency="BNB",
            config=config,
            retries=5,
        )
    )

    finished_strategies = cerebro.run()
    return finished_strategies


if __name__ == "__main__":
    unittest.main()
