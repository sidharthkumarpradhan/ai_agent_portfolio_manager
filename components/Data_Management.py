import streamlit as st
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.database_manager import DatabaseManager
import pandas as pd
from datetime import datetime

# Set up logging for this module
logger = logging.getLogger(__name__)

def main():
    """Data Management interface for querying collected data"""
    logger.info("Loading Data Management component")
    
    st.title("💾 Data Management & Analytics")
    st.markdown("### Query your collected market data and export insights")
    
    # Initialize database connection
    try:
        db_manager = DatabaseManager()
        if not db_manager.is_available():
            st.warning("**Database Not Available**")
            st.info("Database initialization failed. The app can still work in real-time mode.")
            st.markdown("**Available features:**")
            st.markdown("- View real-time market data in Market Analysis")
            st.markdown("- Use AI Agent for portfolio analysis")
            st.markdown("- All core features work without database")
            return
        else:
            db_info = db_manager.get_database_info()
            if db_info["type"] == "sqlite":
                st.success("**SQLite Database Active** - Lightweight data caching enabled")
                st.info("💡 Upgrade tip: Set DATABASE_URL for PostgreSQL in production")
            elif db_info["type"] == "postgresql":
                st.success("**PostgreSQL Database Active** - Production-grade data storage")
            else:
                st.success("**Database Connected** - You can query collected data")
            
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return
    
    # Show database statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Database Overview")
        try:
            db_info = db_manager.get_database_info()
            
            if db_info["type"] == "sqlite":
                st.success(f"✅ SQLite Database Connected")
                st.info(f"📁 File: {db_info['file_path']}")
            elif db_info["type"] == "postgresql":
                st.success(f"✅ PostgreSQL Database Connected")
                st.info(f"🔗 Production Database")
            
            st.metric("Market Data Records", db_info.get('market_records', 0))
            st.metric("Portfolio Records", db_info.get('portfolio_records', 0))
            
            assets = db_manager.get_asset_universe()
            st.metric("Assets Tracked", len(assets))
            
            if assets:
                st.markdown("**Recent Assets:**")
                for asset in assets[:8]:  # Show first 8
                    st.write(f"📈 {asset['symbol']} ({asset['asset_type']}) - {asset['data_points']} records")
                    
        except Exception as e:
            st.error(f"Error loading database info: {str(e)}")
    
    with col2:
        st.subheader("🔍 Quick Data Query")
        
        # Simple query interface
        if st.button("📈 View Recent Market Data", type="primary"):
            try:
                cursor = db_manager.conn.cursor()
                
                if db_manager.db_type == "sqlite":
                    cursor.execute("""
                        SELECT symbol, asset_type, timestamp, close_price, volume
                        FROM market_data 
                        ORDER BY timestamp DESC 
                        LIMIT 100
                    """)
                else:  # PostgreSQL
                    cursor.execute("""
                        SELECT symbol, asset_type, timestamp, close_price, volume
                        FROM market_data 
                        ORDER BY timestamp DESC 
                        LIMIT 100
                    """)
                
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    # Convert to DataFrame for display
                    df = pd.DataFrame(results, columns=['Symbol', 'Type', 'Timestamp', 'Price', 'Volume'])
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
                    st.info("No market data found in database")
                    
            except Exception as e:
                st.error(f"Query error: {str(e)}")
    
    # Advanced query section
    st.markdown("---")
    st.subheader("🛠️ Advanced Queries")
    
    query_type = st.selectbox(
        "Select Query Type:",
        ["Stock Data by Symbol", "Crypto Data", "All Data Export", "Custom SQL"]
    )
    
    if query_type == "Stock Data by Symbol":
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
        
        if st.button("📊 Get Stock Data"):
            try:
                cursor = db_manager.conn.cursor()
                
                if db_manager.db_type == "sqlite":
                    cursor.execute("""
                        SELECT timestamp, open_price, high_price, low_price, close_price, volume
                        FROM market_data 
                        WHERE symbol = ? AND asset_type = 'stock'
                        ORDER BY timestamp DESC 
                        LIMIT 30
                    """, (symbol.upper(),))
                else:  # PostgreSQL
                    cursor.execute("""
                        SELECT timestamp, open_price, high_price, low_price, close_price, volume
                        FROM market_data 
                        WHERE symbol = %s AND asset_type = 'stock'
                        ORDER BY timestamp DESC 
                        LIMIT 30
                    """, (symbol.upper(),))
                
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    df = pd.DataFrame(results, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    st.success(f"Found {len(df)} records for {symbol}")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {symbol} Data",
                        data=csv,
                        file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                st.error(f"Query error: {str(e)}")
    
    elif query_type == "Custom SQL":
        st.markdown("**Execute Custom SQL Query**")
        st.warning("⚠️ Only SELECT queries allowed for security")
        
        sql_query = st.text_area(
            "Enter SQL Query:",
            placeholder="SELECT * FROM market_data WHERE symbol = 'AAPL' LIMIT 10;",
            height=100
        )
        
        if st.button("🚀 Execute Query"):
            if sql_query.strip():
                try:
                    # Safety check
                    if not sql_query.upper().strip().startswith('SELECT'):
                        st.error("Only SELECT queries are allowed")
                    else:
                        cursor = db_manager.conn.cursor()
                        cursor.execute(sql_query)
                        results = cursor.fetchall()
                        
                        if results:
                            # Get column names from cursor description
                            colnames = [desc[0] for desc in cursor.description]
                            cursor.close()
                            
                            df = pd.DataFrame(results, columns=colnames)
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
    
    # Show recent activity
    st.markdown("---")
    st.subheader("📈 Recent Database Activity")
    
    try:
        cursor = db_manager.conn.cursor()
        
        if db_manager.db_type == "sqlite":
            cursor.execute("""
                SELECT DISTINCT symbol, asset_type, 
                       COUNT(*) as records,
                       MAX(created_at) as last_updated
                FROM market_data 
                GROUP BY symbol, asset_type
                ORDER BY last_updated DESC
                LIMIT 10
            """)
        else:  # PostgreSQL
            cursor.execute("""
                SELECT DISTINCT symbol, asset_type, 
                       COUNT(*) as records,
                       MAX(created_at) as last_updated
                FROM market_data 
                GROUP BY symbol, asset_type
                ORDER BY last_updated DESC
                LIMIT 10
            """)
        
        results = cursor.fetchall()
        cursor.close()
        
        if results:
            activity_df = pd.DataFrame(results, columns=['Symbol', 'Type', 'Records', 'Last Updated'])
            st.dataframe(activity_df, use_container_width=True)
        else:
            st.info("No recent activity found")
            
    except Exception as e:
        st.error(f"Error loading activity: {str(e)}")

if __name__ == "__main__":
    main()