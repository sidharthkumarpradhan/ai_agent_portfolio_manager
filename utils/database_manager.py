import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseManager:
    def __init__(self):
        """Initialize database connection - SQLite by default, PostgreSQL if available"""
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.db_available = False
        self.db_type = "none"
        
        # Try PostgreSQL first (for production/advanced users)
        database_url = os.getenv('DATABASE_URL')
        if database_url and database_url.startswith('postgresql'):
            try:
                self.conn = psycopg2.connect(database_url)
                self.conn.autocommit = True
                self.db_available = True
                self.db_type = "postgresql"
                self.logger.info("Connected to PostgreSQL database")
                self.init_tables_postgresql()
                return
            except Exception as e:
                self.logger.warning(f"PostgreSQL connection failed: {str(e)}, falling back to SQLite")
        
        # Use SQLite as default (lightweight, no setup required)
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            db_path = os.path.join('data', 'portfolio_manager.db')
            
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.db_available = True
            self.db_type = "sqlite"
            self.logger.info(f"Connected to SQLite database at {db_path}")
            self.init_tables_sqlite()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            self.conn = None
            self.db_available = False
    
    def init_tables_sqlite(self):
        """Initialize SQLite database tables"""
        try:
            cursor = self.conn.cursor()
            
            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    market_cap REAL,
                    additional_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, asset_type, timestamp)
                )
            """)
            
            # Economic indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_name TEXT NOT NULL,
                    value REAL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, timestamp)
                )
            """)
            
            # Portfolio recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_session TEXT NOT NULL,
                    investment_amount REAL,
                    risk_profile TEXT,
                    recommended_allocation TEXT,
                    ai_analysis TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # API usage tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_name TEXT NOT NULL,
                    endpoint TEXT,
                    requests_count INTEGER DEFAULT 1,
                    last_used TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(api_name, endpoint)
                )
            """)
            
            self.conn.commit()
            self.logger.info("SQLite database tables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing SQLite tables: {str(e)}")
    
    def init_tables_postgresql(self):
        """Initialize PostgreSQL database tables"""
        try:
            cursor = self.conn.cursor()
            
            # Market data table
            cursor.execute("""
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id SERIAL PRIMARY KEY,
                    indicator_name VARCHAR(100) NOT NULL,
                    value DECIMAL(15,8),
                    timestamp TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, timestamp)
                )
            """)
            
            # Portfolio recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                    id SERIAL PRIMARY KEY,
                    user_session VARCHAR(100) NOT NULL,
                    investment_amount DECIMAL(15,2),
                    risk_profile VARCHAR(50),
                    recommended_allocation JSONB,
                    ai_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.logger.info("PostgreSQL database tables initialized successfully")
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing PostgreSQL tables: {str(e)}")
    
    def store_market_data(self, symbol, asset_type, data_df):
        """Store market data in database (works with both SQLite and PostgreSQL)"""
        if not self.db_available:
            return False
            
        try:
            cursor = self.conn.cursor()
            stored_count = 0
            
            for _, row in data_df.iterrows():
                if self.db_type == "sqlite":
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, asset_type, timestamp, open_price, high_price, low_price, close_price, volume, additional_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                else:  # PostgreSQL
                    cursor.execute("""
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
                stored_count += 1
            
            if self.db_type == "sqlite":
                self.conn.commit()
            
            cursor.close()
            self.logger.info(f"Successfully stored {stored_count} records for {symbol} in {self.db_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            return False
    
    def get_market_data(self, symbol, asset_type, days_back=30):
        """Retrieve market data from database"""
        if not self.db_available:
            return None
            
        try:
            cursor = self.conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            if self.db_type == "sqlite":
                cursor.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND asset_type = ? AND datetime(timestamp) >= datetime(?)
                    ORDER BY timestamp DESC
                """, (symbol, asset_type, cutoff_date.isoformat()))
            else:  # PostgreSQL
                cursor.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = %s AND asset_type = %s AND timestamp >= %s
                    ORDER BY timestamp DESC
                """, (symbol, asset_type, cutoff_date))
            
            results = cursor.fetchall()
            
            if results and self.db_type == "postgresql":
                # Get column names for PostgreSQL
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                cursor.close()
                return pd.DataFrame([dict(zip(columns, row)) for row in results])
            elif results:
                # SQLite returns Row objects
                cursor.close()
                return pd.DataFrame([dict(row) for row in results])
            else:
                cursor.close()
                return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {str(e)}")
            return None
    
    def get_asset_universe(self):
        """Get all available assets in database"""
        if not self.db_available:
            return []
            
        try:
            cursor = self.conn.cursor()
            
            if self.db_type == "sqlite":
                cursor.execute("""
                    SELECT DISTINCT symbol, asset_type, COUNT(*) as data_points
                    FROM market_data
                    GROUP BY symbol, asset_type
                    ORDER BY asset_type, symbol
                """)
            else:  # PostgreSQL
                cursor.execute("""
                    SELECT DISTINCT symbol, asset_type, COUNT(*) as data_points
                    FROM market_data
                    GROUP BY symbol, asset_type
                    ORDER BY asset_type, symbol
                """)
            
            results = cursor.fetchall()
            
            if self.db_type == "sqlite":
                # SQLite cursor returns Row objects that can be converted to dict
                cursor.close()
                return [dict(row) for row in results]
            else:
                # PostgreSQL cursor returns tuples, need column names
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                cursor.close()
                return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error getting asset universe: {str(e)}")
            return []
    
    def store_portfolio_recommendation(self, user_session, investment_amount, risk_profile, allocation, analysis):
        """Store portfolio recommendation"""
        if not self.db_available:
            return False
            
        try:
            cursor = self.conn.cursor()
            
            if self.db_type == "sqlite":
                cursor.execute("""
                    INSERT INTO portfolio_recommendations 
                    (user_session, investment_amount, risk_profile, recommended_allocation, ai_analysis)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_session, investment_amount, risk_profile, json.dumps(allocation), analysis))
                self.conn.commit()
            else:  # PostgreSQL
                cursor.execute("""
                    INSERT INTO portfolio_recommendations 
                    (user_session, investment_amount, risk_profile, recommended_allocation, ai_analysis)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_session, investment_amount, risk_profile, json.dumps(allocation), analysis))
            
            cursor.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing portfolio recommendation: {str(e)}")
            return False
    
    def data_freshness_check(self, symbol, asset_type, max_age_hours=1):
        """Check if data is fresh enough"""
        if not self.db_available:
            return False
            
        try:
            cursor = self.conn.cursor()
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            if self.db_type == "sqlite":
                cursor.execute("""
                    SELECT COUNT(*) FROM market_data 
                    WHERE symbol = ? AND asset_type = ? AND datetime(created_at) >= datetime(?)
                """, (symbol, asset_type, cutoff_time.isoformat()))
            else:  # PostgreSQL
                cursor.execute("""
                    SELECT COUNT(*) FROM market_data 
                    WHERE symbol = %s AND asset_type = %s AND created_at >= %s
                """, (symbol, asset_type, cutoff_time))
            
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count > 0
            
        except Exception as e:
            return False
    
    def get_database_info(self):
        """Get database information for display"""
        if not self.db_available:
            return {"type": "none", "status": "unavailable"}
        
        try:
            cursor = self.conn.cursor()
            
            # Get record counts
            cursor.execute("SELECT COUNT(*) FROM market_data")
            market_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM portfolio_recommendations")
            portfolio_count = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                "type": self.db_type,
                "status": "connected",
                "market_records": market_count,
                "portfolio_records": portfolio_count,
                "file_path": "data/portfolio_manager.db" if self.db_type == "sqlite" else "PostgreSQL"
            }
            
        except Exception as e:
            return {"type": self.db_type, "status": "error", "error": str(e)}
    
    def is_available(self):
        """Check if database is available"""
        return self.db_available
    
    def close_connection(self):
        """Close database connection"""
        if self.conn and self.db_available:
            self.conn.close()
            self.logger.info(f"{self.db_type.title()} database connection closed")