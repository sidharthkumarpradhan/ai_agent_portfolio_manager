import streamlit as st
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.database_manager import DatabaseManager
from utils.data_fetcher import DataFetcher
import pandas as pd
import json
from datetime import datetime, timedelta
import io

# Set up logging for this module
logger = logging.getLogger(__name__)

def init_components():
    """Initialize all required components with logging"""
    logger.info("Initializing Data Management components...")
    
    if 'db_manager' not in st.session_state:
        logger.info("Creating DatabaseManager for Data Management")
        st.session_state.db_manager = DatabaseManager()
    
    if 'data_fetcher' not in st.session_state:
        logger.info("Creating DataFetcher for Data Management")
        st.session_state.data_fetcher = DataFetcher()
    
    logger.info("Data Management components initialized successfully")

def main():
    logger.info("Starting Data Management main function")
    
    try:
        # Initialize components
        init_components()
        
        db = st.session_state.db_manager
        data_fetcher = st.session_state.data_fetcher
        
        logger.info("All components loaded successfully for Data Management")
    except Exception as e:
        logger.error(f"Error initializing Data Management components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")
        return
    
    st.title("💾 Data Management & Analytics")
    st.markdown("### Manage your investment data, query databases, and export insights")
    
    # Database schema section
    st.subheader("🗄️ Database Schema")
    
    col_schema1, col_schema2 = st.columns(2)
    
    with col_schema1:
        st.markdown("**Current Database Tables:**")
        
        if db.connection:
            try:
                # Get table information
                db.cursor.execute("""
                    SELECT table_name, 
                           (SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                
                tables = db.cursor.fetchall()
                
                for table in tables:
                    st.write(f"📋 **{table['table_name']}** ({table['column_count']} columns)")
                
            except Exception as e:
                st.error(f"Error fetching schema: {str(e)}")
        else:
            st.error("Database connection not available")
    
    with col_schema2:
        st.markdown("**Database Statistics:**")
        
        if db.connection:
            try:
                # Get data counts
                asset_universe = db.get_asset_universe()
                st.metric("Total Assets Tracked", len(asset_universe))
                
                # Market data count
                db.cursor.execute("SELECT COUNT(*) as count FROM market_data")
                market_count = db.cursor.fetchone()['count']
                st.metric("Market Data Records", market_count)
                
                # Portfolio recommendations count
                db.cursor.execute("SELECT COUNT(*) as count FROM portfolio_recommendations")
                portfolio_count = db.cursor.fetchone()['count']
                st.metric("Portfolio Recommendations", portfolio_count)
                
            except Exception as e:
                st.warning(f"Error fetching statistics: {str(e)}")
    
    st.markdown("---")
    
    # Data query section
    st.subheader("🔍 Query Data Sources")
    
    query_option = st.selectbox(
        "Select Data Source to Query:",
        ["Market Data", "Economic Indicators", "Portfolio Recommendations", "Trading Signals", "Custom SQL Query"]
    )
    
    if query_option == "Market Data":
        st.markdown("**Query Market Data**")
        
        col_query1, col_query2, col_query3 = st.columns(3)
        
        with col_query1:
            symbol_filter = st.text_input("Symbol (optional)", placeholder="e.g., AAPL")
        
        with col_query2:
            asset_type_filter = st.selectbox("Asset Type", ["All", "stock", "crypto", "commodity"])
        
        with col_query3:
            days_back = st.number_input("Days Back", min_value=1, max_value=365, value=30)
        
        if st.button("📊 Query Market Data", type="primary"):
            if db.connection:
                try:
                    query = "SELECT * FROM market_data WHERE timestamp >= %s"
                    params = [datetime.now() - timedelta(days=days_back)]
                    
                    if symbol_filter:
                        query += " AND symbol = %s"
                        params.append(symbol_filter.upper())
                    
                    if asset_type_filter != "All":
                        query += " AND asset_type = %s"
                        params.append(asset_type_filter)
                    
                    query += " ORDER BY timestamp DESC LIMIT 1000"
                    
                    db.cursor.execute(query, params)
                    results = db.cursor.fetchall()
                    
                    if results:
                        df = pd.DataFrame([dict(row) for row in results])
                        st.success(f"Found {len(df)} records")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No data found for the specified criteria")
                        
                except Exception as e:
                    st.error(f"Query error: {str(e)}")
            else:
                st.error("Database connection not available")
    
    elif query_option == "Portfolio Recommendations":
        st.markdown("**Query Portfolio Recommendations**")
        
        if st.button("📈 Get All Recommendations", type="primary"):
            if db.connection:
                try:
                    db.cursor.execute("""
                        SELECT user_session, investment_amount, risk_profile, 
                               created_at, 
                               LEFT(ai_analysis, 200) as analysis_preview
                        FROM portfolio_recommendations 
                        ORDER BY created_at DESC
                    """)
                    
                    results = db.cursor.fetchall()
                    
                    if results:
                        df = pd.DataFrame([dict(row) for row in results])
                        st.success(f"Found {len(df)} recommendations")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Recommendations",
                            data=csv,
                            file_name=f"portfolio_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No portfolio recommendations found")
                        
                except Exception as e:
                    st.error(f"Query error: {str(e)}")
            else:
                st.error("Database connection not available")
    
    elif query_option == "Custom SQL Query":
        st.markdown("**Execute Custom SQL Query**")
        st.warning("⚠️ Use with caution. Only SELECT queries are recommended.")
        
        sql_query = st.text_area(
            "Enter your SQL query:",
            placeholder="SELECT * FROM market_data WHERE symbol = 'AAPL' LIMIT 10;",
            height=100
        )
        
        if st.button("🚀 Execute Query", type="primary"):
            if db.connection and sql_query.strip():
                try:
                    # Basic safety check
                    if not sql_query.upper().strip().startswith('SELECT'):
                        st.error("Only SELECT queries are allowed for safety")
                    else:
                        db.cursor.execute(sql_query)
                        results = db.cursor.fetchall()
                        
                        if results:
                            df = pd.DataFrame([dict(row) for row in results])
                            st.success(f"Query returned {len(df)} records")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("Query executed successfully but returned no results")
                            
                except Exception as e:
                    st.error(f"Query error: {str(e)}")
            else:
                st.error("Database connection not available or empty query")
    
    st.markdown("---")
    
    # Data export section
    st.subheader("📤 Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.markdown("**Quick Exports:**")
        
        if st.button("📊 Export All Market Data"):
            if db.connection:
                try:
                    db.cursor.execute("SELECT * FROM market_data ORDER BY timestamp DESC")
                    results = db.cursor.fetchall()
                    
                    if results:
                        df = pd.DataFrame([dict(row) for row in results])
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Market Data",
                            data=csv,
                            file_name=f"all_market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        st.success(f"Prepared {len(df)} market data records for download")
                    else:
                        st.info("No market data available for export")
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
        
        if st.button("💼 Export Portfolio Data"):
            if 'portfolio_data' in st.session_state:
                portfolio_df = pd.DataFrame.from_dict(st.session_state.portfolio_data, orient='index')
                csv = portfolio_df.to_csv()
                
                st.download_button(
                    label="📥 Download Portfolio",
                    data=csv,
                    file_name=f"my_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.success("Portfolio data prepared for download")
            else:
                st.info("No portfolio data available")
    
    with export_col2:
        st.markdown("**Data Import:**")
        
        uploaded_file = st.file_uploader("Upload CSV data", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully: {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("💾 Store in Database"):
                    st.info("Data import functionality will be implemented with proper validation")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")
    
    # Database maintenance
    st.subheader("🔧 Database Maintenance")
    
    maint_col1, maint_col2 = st.columns(2)
    
    with maint_col1:
        st.markdown("**Data Refresh:**")
        
        if st.button("🔄 Refresh Market Data"):
            with st.spinner("Refreshing market data from APIs..."):
                try:
                    # Force refresh key stocks and crypto
                    symbols_to_refresh = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
                    crypto_to_refresh = ['BTC', 'ETH']
                    
                    refreshed_count = 0
                    for symbol in symbols_to_refresh:
                        data = data_fetcher.get_stock_data(symbol)
                        if data is not None and not data.empty:
                            refreshed_count += 1
                    
                    for symbol in crypto_to_refresh:
                        data = data_fetcher.get_crypto_data(symbol)
                        if data is not None and not data.empty:
                            refreshed_count += 1
                    
                    st.success(f"Refreshed data for {refreshed_count} assets")
                    
                except Exception as e:
                    st.error(f"Refresh error: {str(e)}")
    
    with maint_col2:
        st.markdown("**Database Info:**")
        
        if db.connection:
            st.success("✅ Database Connected")
            
            # Connection info
            try:
                db.cursor.execute("SELECT version();")
                version = db.cursor.fetchone()
                st.info(f"PostgreSQL Version: {version[0]}")
            except:
                st.info("Database version unavailable")
        else:
            st.error("❌ Database Offline")

if __name__ == "__main__":
    main()