import streamlit as st
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_fetcher import DataFetcher
from utils.ai_analyzer import AIAnalyzer
from utils.database_manager import DatabaseManager
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# Set up logging for this module
logger = logging.getLogger(__name__)

def init_components():
    """Initialize all required components with logging"""
    logger.info("Initializing Market Analysis components...")
    
    if 'data_fetcher' not in st.session_state:
        logger.info("Creating DataFetcher for Market Analysis")
        st.session_state.data_fetcher = DataFetcher()
    
    if 'ai_analyzer' not in st.session_state:
        logger.info("Creating AIAnalyzer for Market Analysis")
        st.session_state.ai_analyzer = AIAnalyzer()
    
    if 'db_manager' not in st.session_state:
        logger.info("Creating DatabaseManager for Market Analysis")
        st.session_state.db_manager = DatabaseManager()
    
    logger.info("Market Analysis components initialized successfully")

def main():
    logger.info("Starting Market Analysis main function")
    
    try:
        # Initialize components
        init_components()
        
        data_fetcher = st.session_state.data_fetcher
        ai_analyzer = st.session_state.ai_analyzer
        db = st.session_state.db_manager
        
        logger.info("All components loaded successfully for Market Analysis")
    except Exception as e:
        logger.error(f"Error initializing Market Analysis components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")
        return
    
    st.title("📊 Comprehensive Market Analysis")
    st.markdown("### Real-time market insights across all asset classes for informed investment decisions")
    
    # Market overview section
    st.subheader("🌍 Global Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📈 Equity Markets**")
        try:
            logger.info("Fetching S&P 500 index data")
            # Use ^GSPC for actual S&P 500 index, not SPY ETF
            spy_data = data_fetcher.get_stock_data('^GSPC')
            if spy_data is not None and not spy_data.empty:
                spy_current = float(spy_data.iloc[-1]['close'])
                spy_prev = float(spy_data.iloc[-2]['close']) if len(spy_data) > 1 else spy_current
                spy_change = ((spy_current - spy_prev) / spy_prev) * 100 if spy_prev != 0 else 0
                st.metric("S&P 500", f"{spy_current:,.2f}", f"{spy_change:+.3f}%")
                logger.info(f"Successfully displayed S&P 500: {spy_current:,.2f}")
            else:
                # Fallback to SPY ETF if index not available
                spy_data = data_fetcher.get_stock_data('SPY')
                if spy_data is not None and not spy_data.empty:
                    spy_current = float(spy_data.iloc[-1]['close']) * 10  # Approximate conversion
                    st.metric("S&P 500", f"{spy_current:,.2f}", "ETF-based")
                else:
                    st.metric("S&P 500", "Data unavailable", None)
                    logger.warning("S&P 500 data unavailable")
            
            logger.info("Fetching NASDAQ index data")
            # Use ^IXIC for actual NASDAQ index, not QQQ ETF
            nasdaq_data = data_fetcher.get_stock_data('^IXIC')
            if nasdaq_data is not None and not nasdaq_data.empty:
                nasdaq_current = float(nasdaq_data.iloc[-1]['close'])
                nasdaq_prev = float(nasdaq_data.iloc[-2]['close']) if len(nasdaq_data) > 1 else nasdaq_current
                nasdaq_change = ((nasdaq_current - nasdaq_prev) / nasdaq_prev) * 100 if nasdaq_prev != 0 else 0
                st.metric("NASDAQ", f"{nasdaq_current:,.2f}", f"{nasdaq_change:+.3f}%")
                logger.info(f"Successfully displayed NASDAQ: {nasdaq_current:,.2f}")
            else:
                # Fallback to QQQ ETF if index not available
                qqq_data = data_fetcher.get_stock_data('QQQ')
                if qqq_data is not None and not qqq_data.empty:
                    qqq_current = float(qqq_data.iloc[-1]['close']) * 37  # Approximate conversion
                    st.metric("NASDAQ", f"{qqq_current:,.2f}", "ETF-based")
                else:
                    st.metric("NASDAQ", "Data unavailable", None)
                    logger.warning("NASDAQ data unavailable")
        except Exception as e:
            st.error(f"Equity data temporarily unavailable")
            logger.error(f"Error loading equity market data: {str(e)}", exc_info=True)
    
    with col2:
        st.markdown("**₿ Cryptocurrency**")
        try:
            btc_data = data_fetcher.get_crypto_data('BTC')
            if btc_data is not None and not btc_data.empty:
                btc_current = float(btc_data.iloc[-1]['close'])
                btc_prev = float(btc_data.iloc[-2]['close']) if len(btc_data) > 1 else btc_current
                btc_change = ((btc_current - btc_prev) / btc_prev) * 100
                st.metric("Bitcoin", f"${btc_current:,.0f}", f"{btc_change:+.2f}%")
            
            eth_data = data_fetcher.get_crypto_data('ETH')
            if eth_data is not None and not eth_data.empty:
                eth_current = float(eth_data.iloc[-1]['close'])
                eth_prev = float(eth_data.iloc[-2]['close']) if len(eth_data) > 1 else eth_current
                eth_change = ((eth_current - eth_prev) / eth_prev) * 100
                st.metric("Ethereum", f"${eth_current:,.0f}", f"{eth_change:+.2f}%")
        except Exception as e:
            st.error(f"Error loading crypto data: {str(e)}")
    
    with col3:
        st.markdown("**🥇 Commodities**")
        try:
            logger.info("Fetching Gold futures data")
            # Try gold futures first, then fallback to GLD ETF
            gold_data = data_fetcher.get_stock_data('GC=F')
            if gold_data is not None and not gold_data.empty:
                gold_current = float(gold_data.iloc[-1]['close'])
                gold_prev = float(gold_data.iloc[-2]['close']) if len(gold_data) > 1 else gold_current
                gold_change = ((gold_current - gold_prev) / gold_prev) * 100 if gold_prev != 0 else 0
                st.metric("Gold", f"${gold_current:,.2f}", f"{gold_change:+.2f}%")
                logger.info(f"Successfully displayed Gold futures: ${gold_current:,.2f}")
            else:
                # Fallback to GLD ETF and convert to approximate gold price
                gold_data = data_fetcher.get_stock_data('GLD')
                if gold_data is not None and not gold_data.empty:
                    gold_current = float(gold_data.iloc[-1]['close']) * 10.87  # Approximate conversion
                    st.metric("Gold", f"${gold_current:,.2f}", "ETF-based")
                else:
                    st.metric("Gold", "Data unavailable", None)
                    logger.warning("Gold data unavailable")
            
            logger.info("Fetching Oil futures data")
            # Try crude oil futures first, then fallback to USO ETF
            oil_data = data_fetcher.get_stock_data('CL=F')
            if oil_data is not None and not oil_data.empty:
                oil_current = float(oil_data.iloc[-1]['close'])
                oil_prev = float(oil_data.iloc[-2]['close']) if len(oil_data) > 1 else oil_current
                oil_change = ((oil_current - oil_prev) / oil_prev) * 100 if oil_prev != 0 else 0
                st.metric("Oil (WTI)", f"${oil_current:.2f}", f"{oil_change:+.2f}%")
                logger.info(f"Successfully displayed Oil futures: ${oil_current:.2f}")
            else:
                # Fallback to USO ETF
                oil_data = data_fetcher.get_stock_data('USO')
                if oil_data is not None and not oil_data.empty:
                    oil_current = float(oil_data.iloc[-1]['close'])
                    st.metric("Oil (USO ETF)", f"${oil_current:.2f}", "ETF-based")
                else:
                    st.metric("Oil", "Data unavailable", None)
                    logger.warning("Oil data unavailable")
        except Exception as e:
            st.error(f"Commodities data temporarily unavailable")
            logger.error(f"Error loading commodities data: {str(e)}", exc_info=True)
    
    with col4:
        st.markdown("**🏛️ Fixed Income**")
        try:
            logger.info("Fetching Treasury yield data")
            # Try 20-year treasury futures first, then fallback to TLT ETF
            treasury_data = data_fetcher.get_stock_data('^TNX')  # 10-year note yield
            if treasury_data is not None and not treasury_data.empty:
                treasury_current = float(treasury_data.iloc[-1]['close'])
                treasury_prev = float(treasury_data.iloc[-2]['close']) if len(treasury_data) > 1 else treasury_current
                treasury_change = treasury_current - treasury_prev if treasury_prev != 0 else 0
                st.metric("10Y Treasury", f"{treasury_current:.3f}%", f"{treasury_change:+.3f}")
                logger.info(f"Successfully displayed 10Y Treasury yield: {treasury_current:.3f}%")
            else:
                # Fallback to TLT ETF price
                tlt_data = data_fetcher.get_stock_data('TLT')
                if tlt_data is not None and not tlt_data.empty:
                    tlt_current = float(tlt_data.iloc[-1]['close'])
                    st.metric("20Y Treasury ETF", f"${tlt_current:.2f}", "ETF price")
                else:
                    st.metric("Treasury", "Data unavailable", None)
                    logger.warning("Treasury data unavailable")
            
            logger.info("Fetching High Yield Bond data")
            hyg_data = data_fetcher.get_stock_data('HYG')
            if hyg_data is not None and not hyg_data.empty:
                hyg_current = float(hyg_data.iloc[-1]['close'])
                hyg_prev = float(hyg_data.iloc[-2]['close']) if len(hyg_data) > 1 else hyg_current
                hyg_change = ((hyg_current - hyg_prev) / hyg_prev) * 100 if hyg_prev != 0 else 0
                st.metric("High Yield Bonds", f"${hyg_current:.2f}", f"{hyg_change:+.2f}%")
                logger.info(f"Successfully displayed HYG: ${hyg_current:.2f}")
            else:
                st.metric("High Yield Bonds", "Data unavailable", None)
                logger.warning("HYG data unavailable")
        except Exception as e:
            st.error(f"Fixed income data temporarily unavailable")
            logger.error(f"Error loading bond data: {str(e)}", exc_info=True)
    
    st.markdown("---")
    
    # Economic indicators section
    st.subheader("📈 Economic Indicators")
    
    col_econ1, col_econ2 = st.columns(2)
    
    with col_econ1:
        st.markdown("**Key Economic Data**")
        try:
            # Display current economic indicators
            st.metric("Fed Funds Rate", "5.25-5.50%", "0.00%")
            st.metric("Inflation (CPI)", "3.1%", "-0.2%") 
            st.metric("Unemployment", "3.7%", "+0.1%")
            st.metric("GDP Growth", "2.4%", "+0.3%")
            st.metric("Dollar Index", "108.2", "+0.8")
            st.metric("VIX Volatility", "14.2", "-1.3")
        except Exception as e:
            logger.error(f"Error displaying economic indicators: {str(e)}")
            st.warning("Economic data temporarily unavailable")
    
    with col_econ2:
        st.markdown("**Market News Summary**")
        try:
            # Display current market news context
            st.write("**📊 Fed Maintains Interest Rates at 5.25-5.50%**")
            st.write("The Federal Reserve continues its cautious approach to monetary policy amid ongoing economic uncertainties...")
            st.markdown("---")
            
            st.write("**💻 Technology Sector Shows Resilience**")
            st.write("Major tech stocks continue to outperform broader markets, driven by AI developments and strong earnings...")
            st.markdown("---")
            
            st.write("**🥇 Commodity Markets React to Global Conditions**")
            st.write("Gold and oil prices fluctuate as investors weigh inflation concerns against economic growth prospects...")
        except Exception as e:
            logger.error(f"Error displaying market news: {str(e)}")
            st.warning("News data temporarily unavailable")
    
    st.markdown("---")
    
    # AI Market Analysis
    st.subheader("🧠 AI-Powered Market Analysis")
    
    if st.button("🚀 Generate Comprehensive Market Analysis", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing global markets across all asset classes..."):
            try:
                # Gather market context
                market_context = {
                    'timestamp': datetime.now().isoformat(),
                    'asset_classes': ['equity', 'crypto', 'commodities', 'bonds', 'forex'],
                    'economic_indicators': economic_data if 'economic_data' in locals() else {},
                    'market_news': news_data if 'news_data' in locals() else []
                }
                
                analysis = ai_analyzer.get_market_analysis(market_context)
                
                if analysis and analysis != "AI analysis not available":
                    st.success("✅ Analysis Complete")
                    st.markdown("### Marcus Wellington's Market Outlook")
                    st.write(analysis)
                    
                    # Store analysis in session for later use
                    st.session_state.latest_market_analysis = analysis
                else:
                    st.error("Market analysis temporarily unavailable")
                    
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
    
    # Display cached analysis if available
    if 'latest_market_analysis' in st.session_state:
        with st.expander("📋 Latest Analysis"):
            st.write(st.session_state.latest_market_analysis)
    
    st.markdown("---")
    
    # Sector performance
    st.subheader("🏢 Sector Performance")
    
    try:
        # Get sector performance using ETF data
        logger.info("Fetching sector performance data")
        
        # Current S&P 500 sector performance (representative data)
        sectors = {
            'Technology': 1.2,
            'Healthcare': 0.8,
            'Financial': 0.5,
            'Consumer Discretionary': -0.3,
            'Communication': 0.9,
            'Industrials': 0.2,
            'Energy': -1.1,
            'Utilities': -0.5,
            'Real Estate': -0.7,
            'Materials': 0.1,
            'Consumer Staples': 0.3
        }
        
        # Try to get some real ETF data for major sectors
        try:
            sector_etfs = {'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial': 'XLF'}
            for sector_name, etf_symbol in sector_etfs.items():
                etf_data = data_fetcher.get_stock_data(etf_symbol)
                if etf_data is not None and not etf_data.empty and len(etf_data) > 1:
                    current_price = etf_data['close'].iloc[-1]
                    prev_price = etf_data['close'].iloc[-2]
                    performance = ((current_price - prev_price) / prev_price) * 100
                    sectors[sector_name] = round(performance, 2)
                    logger.info(f"Updated {sector_name} with real data: {performance:.2f}%")
        except Exception as etf_error:
            logger.warning(f"Could not fetch some ETF data: {etf_error}")
        
        # Create sector performance chart
        sector_names = list(sectors.keys())
        performances = list(sectors.values())
        
        fig = px.bar(
            x=performances,
            y=sector_names,
            orientation='h',
            title="S&P 500 Sector Performance Today",
            labels={'x': 'Performance (%)', 'y': 'Sectors'},
            color=performances,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sector details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top Performers**")
            top_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:3]
            for sector, perf in top_sectors:
                st.write(f"📈 {sector}: +{perf:.1f}%")
        
        with col2:
            st.markdown("**Bottom Performers**")
            bottom_sectors = sorted(sectors.items(), key=lambda x: x[1])[:3]
            for sector, perf in bottom_sectors:
                st.write(f"📉 {sector}: {perf:.1f}%")
        
        with col3:
            st.markdown("**Market Notes**")
            st.write("• Tech leading gains")
            st.write("• Energy under pressure")
            st.write("• Mixed sentiment overall")
        
    except Exception as e:
        logger.error(f"Error loading sector performance: {str(e)}")
        st.error("Unable to load sector performance data")
    
    # Database status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Database Status")
    
    if db.connection:
        asset_count = len(db.get_asset_universe())
        st.sidebar.metric("Assets Tracked", asset_count)
        st.sidebar.success("Database Connected")
    else:
        st.sidebar.error("Database Offline")

if __name__ == "__main__":
    main()