import pandas as pd
import pandas_ta as ta

def perform_technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates key technical indicators and adds them to the price history DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with historical price data. 
                           Must contain 'Close' prices.

    Returns:
        pd.DataFrame: The DataFrame with appended technical indicator columns.
    """
    print("Performing technical analysis...")
    df_ta = df.copy()

    try:
        # Calculate Simple Moving Averages
        df_ta.ta.sma(length=50, append=True)
        df_ta.ta.sma(length=200, append=True)

        # Calculate Relative Strength Index (RSI)
        df_ta.ta.rsi(length=14, append=True)

        # Calculate Moving Average Convergence Divergence (MACD)
        df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)

        # Clean up the default column names for better readability
        df_ta.rename(columns={
            "SMA_50": "50_day_sma",
            "SMA_200": "200_day_sma",
            "RSI_14": "rsi",
            "MACD_12_26_9": "macd",
            "MACDh_12_26_9": "macd_histogram",
            "MACDs_12_26_9": "macd_signal"
        }, inplace=True)
        
        print("Technical analysis complete.")
        return df_ta

    except Exception as e:
        print(f"An error occurred during technical analysis: {e}")
        # Return the original dataframe if analysis fails
        return df

# This block allows for direct testing of this module
if __name__ == '__main__':
    print("--- Testing analysis.py module ---")
    
    # We need some data to test with. Let's import our data_sourcing module.
    # This demonstrates how modules can be used by each other.
    from data_sourcing import get_stock_data
    
    ticker = "GOOGL"
    price_data, _ = get_stock_data(ticker)
    
    if price_data is not None:
        # Run the analysis
        analyzed_data = perform_technical_analysis(price_data)
        
        print(f"\n--- Test for {ticker} successful ---")
        print("Columns after analysis:")
        print(analyzed_data.columns.tolist())
        print("\nAnalyzed Data Sample (last 2 rows):")
        # Display the last two rows to show the new indicator values
        print(analyzed_data.tail(2))