
import streamlit as st

# Import the functions from our other modules
from data_sourcing import get_stock_data
from analysis import perform_technical_analysis
from financial_agent import generate_structured_recommendation
import config

# --- Streamlit User Interface ---

st.set_page_config(page_title="AI Financial Analyst", layout="wide")
st.title("AI-Powered Financial Analyst")
st.markdown("Enter a stock ticker to get a comprehensive analysis and recommendation powered by Groq and Llama 3.")

# Check for API key configuration
if config.GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not config.GROQ_API_KEY:
    st.error("Groq API Key is not configured. Please set it in the config.py file.")
else:
    # Input field for the stock ticker
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, NVDA):", "AAPL").upper()

    if st.button("Analyze Stock"):
        if not ticker_input:
            st.warning("Please enter a stock ticker.")
        else:
            with st.spinner(f"Running full analysis for {ticker_input}... This may take a moment."):
                
                # --- Orchestration ---
                # 1. Fetch Data
                price_data, fundamental_data = get_stock_data(ticker_input)

                if price_data is not None and fundamental_data is not None:
                    # 2. Perform Technical Analysis
                    analyzed_data = perform_technical_analysis(price_data)

                    # 3. Get Recommendation from the Financial Agent
                    recommendation = generate_structured_recommendation(
                        ticker_symbol=ticker_input,
                        fundamentals=fundamental_data,
                        prices_with_ta=analyzed_data
                    )

                    # --- Display Results ---
                    st.header(f"Analysis for {fundamental_data.get('company_name', ticker_input)}")

                    if recommendation:
                        rec_color = "green" if recommendation.recommendation == "Buy" else "orange" if recommendation.recommendation == "Hold" else "red"
                        st.markdown(f"### Recommendation: <span style='color:{rec_color};'>{recommendation.recommendation}</span>", unsafe_allow_html=True)
                        st.progress(recommendation.confidence_score)
                        st.markdown(f"**Confidence Score:** {recommendation.confidence_score:.2f}")

                        st.subheader("Analyst's Explanation")
                        st.write(recommendation.explanation)
                    else:
                        st.error("Could not generate a recommendation. Please check the logs.")

                    # Display charts and raw data
                    st.subheader("Price Chart with Moving Averages")
                    st.line_chart(analyzed_data[['Close', '50_day_sma', '200_day_sma']])
                    
                    with st.expander("View Raw Data"):
                        st.subheader("Key Fundamentals")
                        st.json(fundamental_data)
                        st.subheader("Latest Technical Indicators")
                        st.dataframe(analyzed_data.tail(1))
                else:
                    st.error(f"Could not retrieve data for {ticker_input}. Please ensure the ticker is correct.")

# Financial Disclaimer
st.markdown("---")
st.warning("**Disclaimer:** This is an AI-generated analysis for informational purposes only. It does not constitute financial advice. Always conduct your own research.", icon="⚠️")