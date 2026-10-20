from abc import ABC, abstractmethod

import pandas as pd


class TradingData(ABC):
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_symbol(self) -> str:
        raise NotImplementedError
