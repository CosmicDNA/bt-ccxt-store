import time
from datetime import datetime, timezone
import logging

import backtrader as bt

from ccxtbt import CCXTFeed


def main():
    # Set a general level (e.g., INFO) for other loggers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Specifically set the CCXTFeed logger to DEBUG level
    logging.getLogger("CCXTFeed").setLevel(logging.DEBUG)

    class TestStrategy(bt.Strategy):
        def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.next_runs = 0

        def next(self, dt=None):
            dt = dt or self.datas[0].datetime.datetime(0)
            self.logger.info(
                "%s closing price: %s" % (dt.isoformat(), self.datas[0].close[0])
            )
            self.next_runs += 1

    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    # Add the feed
    cerebro.adddata(
        CCXTFeed(
            exchange="binance",
            dataname="BNB/USDT",
            timeframe=bt.TimeFrame.Minutes,
            fromdate=datetime(2019, 1, 1, 0, 0, tzinfo=timezone.utc),
            todate=datetime(2019, 1, 1, 0, 2, tzinfo=timezone.utc),
            compression=1,
            ohlcv_limit=2,
            currency="BNB",
            retries=5,
            # 'apiKey' and 'secret' are skipped
            config={
                "enableRateLimit": True,
                "nonce": lambda: str(int(time.time() * 1000)),
            },
        )
    )

    # Run the strategy
    cerebro.run()


if __name__ == "__main__":
    main()
