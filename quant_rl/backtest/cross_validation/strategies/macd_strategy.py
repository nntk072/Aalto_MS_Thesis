"""MACD strategy for backtrader to match the quant_rl baselines."""

import backtrader as bt


class MACDStrategy(bt.Strategy):
    """Backtrader implementation of the MACD baseline strategy.

    Matches the logic in quant_rl.baselines.rule_based.macd_baseline:
    - Long when MACD histogram > 0
    - Short when MACD histogram < 0
    - Flat when MACD histogram == 0
    """

    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal_period", 9),
        ("printlog", False),
    )

    def __init__(self):
        # Keep reference to close
        self.dataclose = self.datas[0].close

        # MACD parameters
        self.fast = self.p.fast  # type: ignore[attr-defined]
        self.slow = self.p.slow  # type: ignore[attr-defined]
        self.signal_period = self.p.signal_period  # type: ignore[attr-defined]

        # MACD indicator
        self.macd = bt.indicators.MACD(
            self.dataclose,
            period_me1=self.fast,
            period_me2=self.slow,
            period_signal=self.signal_period,
        )

        # Track orders and position
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log(self, txt: str, dt: object = None, doprint: bool = False) -> None:
        """Logging function."""
        if self.p.printlog or doprint:  # type: ignore[attr-defined]
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f"OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}")

    def next(self):
        # Log close price
        self.log(f"Close: {self.dataclose[0]:.2f}")

        # Check if order is pending
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in market, look for buy signal (MACD histogram > 0)
            macd_hist = self.macd.macd - self.macd.signal
            if macd_hist[0] > 0:
                self.log(f"BUY CREATE, {self.dataclose[0]:.2f}")
                self.order = self.buy()
        else:
            # In market, look for sell signal (MACD histogram < 0)
            macd_hist = self.macd.macd - self.macd.signal
            if macd_hist[0] < 0:
                self.log(f"SELL CREATE, {self.dataclose[0]:.2f}")
                self.order = self.sell()

    def stop(self):
        self.log(
            f"MACD Fast: {self.p.fast}, Slow: {self.p.slow}, Signal: {self.p.signal_period}",  # type: ignore[attr-defined]
            doprint=True,
        )
        self.log(f"(MACD Strategy) Ending Value: {self.broker.getvalue():.2f}", doprint=True)


class EMAMACDStrategy(bt.Strategy):
    """Backtrader implementation of the MACD + EMA50 baseline strategy.

    Matches the logic in quant_rl.baselines.rule_based.macd_ema50_baseline:
    - Long entry: close > EMA50 AND bullish MACD cross
    - Long exit: bearish MACD cross
    - Short entry: close < EMA50 AND bearish MACD cross
    - Short exit: bullish MACD cross
    - Cooldown: wait >= 5 bars after any exit before next entry
    """

    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal_period", 9),
        ("ema50_period", 50),
        ("cooldown_bars", 5),
        ("printlog", False),
    )

    def __init__(self):
        # Keep reference to close
        self.dataclose = self.datas[0].close

        # Parameters
        self.fast = self.p.fast  # type: ignore[attr-defined]
        self.slow = self.p.slow  # type: ignore[attr-defined]
        self.signal_period = self.p.signal_period  # type: ignore[attr-defined]
        self.ema50_period = self.p.ema50_period  # type: ignore[attr-defined]
        self.cooldown_bars = self.p.cooldown_bars  # type: ignore[attr-defined]

        # Indicators
        self.fast_ema = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.fast)
        self.slow_ema = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.slow)
        self.ema50 = bt.indicators.ExponentialMovingAverage(
            self.dataclose, period=self.ema50_period
        )

        # MACD line and signal line
        self.macd_line = self.fast_ema - self.slow_ema
        self.signal_line = bt.indicators.SimpleMovingAverage(
            self.macd_line, period=self.signal_period
        )

        # Track state
        self.order = None
        self.position = 0  # 0=flat, 1=long, -1=short
        self.cooldown_counter = 0

    def log(self, txt: str, dt: object = None, doprint: bool = False) -> None:
        """Logging function."""
        if self.p.printlog or doprint:  # type: ignore[attr-defined]
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                self.position = 1
            else:  # Sell
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                self.position = 0

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def next(self):
        # Decrement cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Check if order is pending
        if self.order:
            return

        # Detect crosses
        # Bullish cross: MACD line crosses above signal line
        bullish_cross = (
            self.macd_line[-1] <= self.signal_line[-1] and self.macd_line[0] > self.signal_line[0]
        )
        # Bearish cross: MACD line crosses below signal line
        bearish_cross = (
            self.macd_line[-1] >= self.signal_line[-1] and self.macd_line[0] < self.signal_line[0]
        )

        if self.position == 0:
            # Flat: check for entry
            if self.cooldown_counter == 0:
                if self.dataclose[0] > self.ema50[0] and bullish_cross:
                    # Long entry
                    self.log(f"BUY CREATE, {self.dataclose[0]:.2f}")
                    self.order = self.buy()
                    self.position = 1
                elif self.dataclose[0] < self.ema50[0] and bearish_cross:
                    # Short entry
                    self.log(f"SELL CREATE, {self.dataclose[0]:.2f}")
                    self.order = self.sell()
                    self.position = -1

        elif self.position == 1:
            # In long: check for exit
            if bearish_cross:
                # Exit signal
                self.log(f"EXIT LONG CREATE, {self.dataclose[0]:.2f}")
                self.order = self.close()
                self.position = 0
                self.cooldown_counter = self.cooldown_bars

        elif self.position == -1:
            # In short: check for exit
            if bullish_cross:
                # Exit signal
                self.log(f"EXIT SHORT CREATE, {self.dataclose[0]:.2f}")
                self.order = self.close()
                self.position = 0
                self.cooldown_counter = self.cooldown_bars
