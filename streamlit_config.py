"""
Streamlit configuration for deployment
"""
import streamlit as st
import os

def configure_streamlit():
    """Configure Streamlit settings for deployment"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Portfolio Manager",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def check_api_keys():
    """Check if required API keys are configured"""
    required_keys = {
        "ANTHROPIC_API_KEY": "Anthropic API key for AI analysis",
        "ALPHA_VANTAGE_API_KEY": "Alpha Vantage API key for market data"
    }
    
    optional_keys = {
        "COINGECKO_API_KEY": "CoinGecko API key for crypto data (optional - free tier available)",
        "FINNHUB_API_KEY": "Finnhub API key for stock data backup (optional - free tier available)"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"- {key}: {description}")
    
    if missing_keys:
        st.error("Missing required API keys:")
        for key in missing_keys:
            st.write(key)
        
        st.markdown("""
        ### How to set up API keys:
        
        **For local development:**
        1. Copy `.env.example` to `.env`
        2. Add your API keys to the `.env` file
        
        **For Streamlit Cloud deployment:**
        1. Go to your app settings in Streamlit Cloud
        2. Add the API keys in the secrets section
        
        **Getting API keys:**
        - Anthropic: https://console.anthropic.com/
        - Alpha Vantage: https://www.alphavantage.co/support/#api-key
        - CoinGecko (optional): https://www.coingecko.com/en/api/pricing
        - Finnhub (optional): https://finnhub.io/register
        """)
        
        return False
    
    # Check optional keys and show info
    missing_optional = []
    for key, description in optional_keys.items():
        if not os.getenv(key):
            missing_optional.append(f"- {key}: {description}")
    
    if missing_optional:
        st.info("Optional configurations not set (app will use alternatives):")
        for key in missing_optional:
            st.write(key)
        
        # Check database status
        if not os.getenv("DATABASE_URL"):
            st.write("- DATABASE_URL: PostgreSQL database for data caching (optional - app works without it)")
    
    return True