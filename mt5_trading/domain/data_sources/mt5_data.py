import MetaTrader5 as mt5
import pandas as pd

from mt5_trading.adapters import TradingData


class MT5Data(TradingData):
    def __init__(self, symbol: str, time_frame: int) -> None:
        self.symbol = symbol
        self.time_frame = time_frame

    def get_data(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, self.time_frame, 0, 1000)
        rates_frame = pd.DataFrame(rates)
        if not rates_frame.empty:
            rates_frame["time"] = pd.to_datetime(rates_frame["time"], unit="s")
        return rates_frame

    def get_symbol(self) -> str:
        return self.symbol
