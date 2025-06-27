import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime, timedelta
import json
import streamlit as st

class DatabaseManager:
    def __init__(self):
        """Initialize database connection using environment variables"""
        try:
            self.database_url = os.getenv('DATABASE_URL')
            if not self.database_url:
                st.error("DATABASE_URL environment variable not found")
                self.connection = None
                return
            
            self.connection = psycopg2.connect(self.database_url)
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            self.init_tables()
            
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            self.connection = None
    
    def init_tables(self):
        """Initialize database tables for storing market data"""
        try:
            # Market data table for stocks, crypto, etc.
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    asset_type VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price DECIMAL(15,8),
                    high_price DECIMAL(15,8),
                    low_price DECIMAL(15,8),
                    close_price DECIMAL(15,8),
                    volume BIGINT,
                    market_cap DECIMAL(20,2),
                    additional_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, asset_type, timestamp)
                )
            """)
            
            # Economic indicators table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id SERIAL PRIMARY KEY,
                    indicator_name VARCHAR(100) NOT NULL,
                    value DECIMAL(15,8),
                    timestamp TIMESTAMP NOT NULL,
                    country VARCHAR(10),
                    additional_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, timestamp, country)
                )
            """)
            
            # News sentiment table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    sentiment_score DECIMAL(5,4),
                    sentiment_label VARCHAR(20),
                    symbols TEXT[],
                    source VARCHAR(100),
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio recommendations table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                    id SERIAL PRIMARY KEY,
                    user_session VARCHAR(100),
                    investment_amount DECIMAL(15,2),
                    risk_profile VARCHAR(20),
                    recommended_allocation JSONB,
                    ai_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trading signals table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(20), -- BUY, SELL, HOLD
                    confidence DECIMAL(5,4),
                    reasoning TEXT,
                    price_target DECIMAL(15,8),
                    stop_loss DECIMAL(15,8),
                    time_horizon VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            
        except Exception as e:
            st.error(f"Error initializing database tables: {str(e)}")
            self.connection.rollback()
    
    def store_market_data(self, symbol, asset_type, data_df):
        """Store market data in database"""
        if not self.connection:
            return False
            
        try:
            for _, row in data_df.iterrows():
                self.cursor.execute("""
                    INSERT INTO market_data 
                    (symbol, asset_type, timestamp, open_price, high_price, low_price, close_price, volume, additional_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, asset_type, timestamp) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    additional_data = EXCLUDED.additional_data
                """, (
                    symbol, 
                    asset_type, 
                    str(row.get('timestamp', row.name)),
                    float(str(row.get('open_price', row.get('open', 0)))), 
                    float(str(row.get('high_price', row.get('high', 0)))),
                    float(str(row.get('low_price', row.get('low', 0)))), 
                    float(str(row.get('close_price', row.get('close', 0)))),
                    int(float(str(row.get('volume', 0)))), 
                    None
                ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            st.error(f"Error storing market data: {str(e)}")
            self.connection.rollback()
            return False
    
    def get_market_data(self, symbol, asset_type, days_back=30):
        """Retrieve market data from database"""
        if not self.connection:
            return None
            
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            self.cursor.execute("""
                SELECT * FROM market_data
                WHERE symbol = %s AND asset_type = %s AND timestamp >= %s
                ORDER BY timestamp DESC
            """, (symbol, asset_type, cutoff_date))
            
            results = self.cursor.fetchall()
            if results:
                return pd.DataFrame([dict(row) for row in results])
            return None
            
        except Exception as e:
            st.error(f"Error retrieving market data: {str(e)}")
            return None
    
    def store_economic_indicators(self, indicators_data):
        """Store economic indicators in database"""
        if not self.connection:
            return False
            
        try:
            for indicator_name, data in indicators_data.items():
                self.cursor.execute("""
                    INSERT INTO economic_indicators 
                    (indicator_name, value, timestamp, additional_data)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (indicator_name, timestamp, country) DO UPDATE SET
                    value = EXCLUDED.value,
                    additional_data = EXCLUDED.additional_data
                """, (
                    indicator_name, 
                    float(data.get('value', 0)),
                    data.get('timestamp', datetime.now()),
                    json.dumps(data)
                ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            st.error(f"Error storing economic indicators: {str(e)}")
            self.connection.rollback()
            return False
    
    def store_portfolio_recommendation(self, user_session, investment_amount, risk_profile, allocation, analysis):
        """Store portfolio recommendation"""
        if not self.connection:
            return False
            
        try:
            self.cursor.execute("""
                INSERT INTO portfolio_recommendations 
                (user_session, investment_amount, risk_profile, recommended_allocation, ai_analysis)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                user_session, investment_amount, risk_profile,
                json.dumps(allocation), analysis
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            st.error(f"Error storing portfolio recommendation: {str(e)}")
            self.connection.rollback()
            return False
    
    def get_latest_portfolio_recommendation(self, user_session):
        """Get latest portfolio recommendation for user"""
        if not self.connection:
            return None
            
        try:
            self.cursor.execute("""
                SELECT * FROM portfolio_recommendations
                WHERE user_session = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_session,))
            
            result = self.cursor.fetchone()
            return dict(result) if result else None
            
        except Exception as e:
            st.error(f"Error retrieving portfolio recommendation: {str(e)}")
            return None
    
    def data_freshness_check(self, symbol, asset_type, max_age_hours=1):
        """Check if data is fresh enough"""
        if not self.connection:
            return False
            
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            self.cursor.execute("""
                SELECT COUNT(*) as count FROM market_data
                WHERE symbol = %s AND asset_type = %s AND created_at >= %s
            """, (symbol, asset_type, cutoff_time))
            
            result = self.cursor.fetchone()
            return result['count'] > 0
            
        except Exception as e:
            return False
    
    def get_asset_universe(self):
        """Get all available assets in database"""
        if not self.connection:
            return []
            
        try:
            self.cursor.execute("""
                SELECT DISTINCT symbol, asset_type, COUNT(*) as data_points
                FROM market_data
                GROUP BY symbol, asset_type
                ORDER BY asset_type, symbol
            """)
            
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            st.error(f"Error getting asset universe: {str(e)}")
            return []
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.cursor.close()
            self.connection.close()