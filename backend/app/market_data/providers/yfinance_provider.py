from datetime import date

import pandas as pd
import yfinance as yf

from app.market_data.enums import MarketDataSource
from app.market_data.providers.base import MarketDataProvider
from app.market_data.schemas import MarketDataSnapshotResponse


class YFinanceMarketDataProvider(MarketDataProvider):
    """Market data provider for ETF and stock historical prices via yfinance."""

    def _normalize_dataframe(self, history_df):
        """Normalize yfinance dataframe columns for single-symbol usage."""
        if history_df.empty:
            return history_df

        if isinstance(history_df.columns, pd.MultiIndex):
            history_df.columns = history_df.columns.get_level_values(0)

        return history_df

    def _to_float(self, value) -> float | None:
        """Convert yfinance values safely into float."""
        if value is None:
            return None

        if isinstance(value, pd.Series):
            if value.empty:
                return None
            value = value.iloc[0]

        if pd.isna(value):
            return None

        return round(float(value), 4)

    def _build_snapshot(
        self,
        symbol: str,
        data_date: date,
        row,
    ) -> MarketDataSnapshotResponse:
        return MarketDataSnapshotResponse(
            snapshot_id=f"yfinance-{symbol}-{data_date}",
            instrument_id=symbol,
            data_date=data_date,
            open_price=self._to_float(row.get("Open")),
            high_price=self._to_float(row.get("High")),
            low_price=self._to_float(row.get("Low")),
            close_price=self._to_float(row.get("Close")),
            nav=None,
            volume=self._to_float(row.get("Volume")),
            source=MarketDataSource.YFINANCE,
        )

    def get_history(
        self,
        instrument_id: str,
    ) -> list[MarketDataSnapshotResponse]:
        symbol = instrument_id

        history_df = yf.download(
            symbol,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )

        history_df = self._normalize_dataframe(history_df)

        if history_df.empty:
            return []

        snapshots: list[MarketDataSnapshotResponse] = []

        for index, row in history_df.iterrows():
            data_date = index.date()

            snapshot = self._build_snapshot(
                symbol=symbol,
                data_date=data_date,
                row=row,
            )

            if snapshot.close_price is not None:
                snapshots.append(snapshot)

        return snapshots

    def get_latest(
        self,
        instrument_id: str,
    ) -> MarketDataSnapshotResponse | None:
        history = self.get_history(instrument_id)

        if not history:
            return None

        return history[-1]