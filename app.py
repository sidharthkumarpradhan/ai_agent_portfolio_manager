import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import importlib.util
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('portfolio_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_fetcher import DataFetcher
from utils.ai_analyzer import AIAnalyzer
from utils.database_manager import DatabaseManager

def init_components():
    """Initialize data fetcher and AI analyzer with logging"""
    logger.info("Initializing application components...")
    
    if 'data_fetcher' not in st.session_state:
        logger.info("Creating new DataFetcher instance")
        st.session_state.data_fetcher = DataFetcher()
    
    if 'ai_analyzer' not in st.session_state:
        logger.info("Creating new AIAnalyzer instance")
        st.session_state.ai_analyzer = AIAnalyzer()
    
    logger.info("Component initialization complete")

def main():
    logger.info("Starting main application")
    
    try:
        st.set_page_config(
            page_title="AI Portfolio Management",
            page_icon="💼",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        logger.info("Page configuration set successfully")
    except Exception as e:
        logger.error(f"Error setting page config: {str(e)}")

    # Custom navigation - this replaces the default Streamlit page navigation
    st.sidebar.title("🤖 AI Portfolio Manager")
    st.sidebar.markdown("### Navigation")
    logger.info("Sidebar navigation initialized")

    # Three main navigation options
    page_selection = st.sidebar.selectbox(
        "Choose Section:",
        ["📊 Market Analysis", "🧠 AI Agent", "💾 Data Management"],
        index=0
    )
    logger.info(f"User selected page: {page_selection}")

    # Route to appropriate page based on selection
    try:
        if page_selection == "📊 Market Analysis":
            logger.info("Loading Market Analysis component")
            spec = importlib.util.spec_from_file_location("market_analysis", "components/Market_Analysis.py")
            market_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(market_module)
            market_module.main()
            return
        elif page_selection == "🧠 AI Agent":
            logger.info("Loading AI Agent component")
            spec = importlib.util.spec_from_file_location("ai_agent", "components/AI_Agent.py")
            ai_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ai_module)
            ai_module.main()
            return
        elif page_selection == "💾 Data Management":
            logger.info("Loading Data Management component")
            spec = importlib.util.spec_from_file_location("data_management", "components/Data_Management.py")
            data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_module)
            data_module.main()
            return
    except Exception as e:
        logger.error(f"Error loading component {page_selection}: {str(e)}")
        st.error(f"Error loading {page_selection}. Check logs for details.")

    # This should never be reached due to the returns above, but kept as fallback
    logger.warning("Reached fallback section - this should not happen")
    st.title("💼 AI Portfolio Management System")
    st.markdown("### Please select a section from the navigation dropdown above")

if __name__ == "__main__":
    main()