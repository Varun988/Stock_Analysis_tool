
import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Import our configuration settings
import config

# 1. Define the desired structured output for our recommendation
class StockRecommendation(BaseModel):
    """A structured recommendation for a stock."""
    recommendation: str = Field(description="The final recommendation, must be one of: 'Buy', 'Hold', or 'Sell'.")
    confidence_score: float = Field(description="A confidence score from 0.0 to 1.0 for the recommendation.")
    explanation: str = Field(description="A detailed, multi-paragraph explanation supporting the recommendation.")

# 2. Initialize the LLM client using settings from the config file
try:
    llm = ChatGroq(
        temperature=config.LLM_TEMPERATURE,
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL_NAME
    )
    print("LLM client initialized successfully.")
except Exception as e:
    print(f"Error initializing LLM client: {e}")
    llm = None

# 3. Create the main agent function
def generate_structured_recommendation(
    ticker_symbol: str, 
    fundamentals: dict, 
    prices_with_ta: pd.DataFrame
) -> StockRecommendation | None:
    """
    Generates a structured recommendation using the LLM.

    Args:
        ticker_symbol (str): The stock ticker.
        fundamentals (dict): The dictionary of fundamental data.
        prices_with_ta (pd.DataFrame): The DataFrame with technical analysis.

    Returns:
        StockRecommendation: A Pydantic object with the structured recommendation, or None if it fails.
    """
    if not llm:
        print("LLM client is not available. Cannot generate recommendation.")
        return None

    print("Generating structured recommendation with Groq LLM...")
    
    structured_llm = llm.with_structured_output(StockRecommendation)
    latest_indicators = prices_with_ta.iloc[-1]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior investment analyst. Your task is to provide a clear, data-driven investment recommendation (Buy, Hold, or Sell) with a confidence score. You must base your entire analysis and explanation *only* on the data provided. Do not use any external knowledge."),
        ("human", """
            Analyze the stock: {ticker}

            **Fundamental Data:**
            {fundamentals}

            **Technical Analysis Data (Latest):**
            - Current Price: {current_price:.2f}
            - 50-Day SMA: {fifty_day_sma:.2f}
            - 200-Day SMA: {two_hundred_day_sma:.2f}
            - RSI: {rsi:.2f}
            - MACD Line: {macd:.2f}
            - MACD Signal Line: {macd_signal:.2f}

            **Your Task:**
            Based on a holistic analysis of the provided fundamental and technical data, generate a structured recommendation.
            - **For Fundamentals:** Consider if the P/E ratio suggests the stock is over or undervalued. Look at debt, profitability (Return on Equity), and dividend yield for stability.
            - **For Technicals:** Is the current price above or below key moving averages? Is the RSI indicating overbought (>70) or oversold (<30)? Is the MACD line above its signal line (bullish) or below (bearish)?
            - **Synthesize:** Combine these insights into a final recommendation, a confidence score, and a detailed explanation.
        """)
    ])

    chain = prompt | structured_llm

    try:
        recommendation = chain.invoke({
            "ticker": ticker_symbol,
            "fundamentals": str(fundamentals),
            "current_price": latest_indicators.get('Close', 0),
            "fifty_day_sma": latest_indicators.get('50_day_sma', 0),
            "two_hundred_day_sma": latest_indicators.get('200_day_sma', 0),
            "rsi": latest_indicators.get('rsi', 0),
            "macd": latest_indicators.get('macd', 0),
            "macd_signal": latest_indicators.get('macd_signal', 0)
        })
        print("Structured recommendation generated successfully.")
        return recommendation
    except Exception as e:
        print(f"An error occurred while invoking the LLM chain: {e}")
        return None

# This block allows for direct testing of this module
if __name__ == '__main__':
    print("--- Testing financial_agent.py module ---")
    
    # To test this, we need data from our other modules
    from data_sourcing import get_stock_data
    from analysis import perform_technical_analysis
    
    ticker = "TSLA"
    price_data, fundamental_data = get_stock_data(ticker)
    
    if price_data is not None and fundamental_data is not None:
        analyzed_data = perform_technical_analysis(price_data)
        
        # Now, call our agent function
        recommendation = generate_structured_recommendation(ticker, fundamental_data, analyzed_data)
        
        if recommendation:
            print(f"\n--- Test for {ticker} successful ---")
            print(f"Recommendation: {recommendation.recommendation}")
            print(f"Confidence Score: {recommendation.confidence_score}")
            print("\nExplanation:")
            print(recommendation.explanation)