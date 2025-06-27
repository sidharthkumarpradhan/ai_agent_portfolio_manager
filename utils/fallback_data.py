import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FallbackDataProvider:
    """Provides fallback market data when APIs are unavailable"""
    
    def __init__(self):
        self.base_prices = {
            # Major stocks
            'AAPL': 190.0,
            'MSFT': 420.0,
            'GOOGL': 140.0,
            'TSLA': 240.0,
            'NVDA': 900.0,
            'META': 520.0,
            'AMZN': 180.0,
            'SPY': 480.0,
            'QQQ': 420.0,
            'IWM': 200.0,
            'VTI': 240.0,
            
            # Crypto (in USD)
            'BTC': 45000.0,
            'ETH': 2500.0,
            
            # Commodities
            'GLD': 185.0,
            'SLV': 22.0,
            'USO': 75.0,
            'DBA': 19.0,
            
            # Bonds
            'TLT': 92.0,
            'IEF': 100.0,
            'SHY': 82.0,
            'HYG': 78.0,
            
            # Forex
            'UUP': 29.0,
            'FXE': 105.0,
            'FXY': 67.0,
            'EWZ': 31.0
        }
    
    def generate_stock_data(self, symbol, days=100):
        """Generate realistic stock data for fallback"""
        if symbol not in self.base_prices:
            return pd.DataFrame()
        
        base_price = self.base_prices[symbol]
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        returns = np.random.normal(0.0005, 0.02, days)  # Small daily returns with volatility
        
        prices = [base_price]
        for i in range(1, days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            daily_vol = abs(np.random.normal(0, 0.01))  # Daily volatility
            
            high = close * (1 + daily_vol)
            low = close * (1 - daily_vol)
            open_price = close * (1 + np.random.normal(0, 0.005))
            
            # Ensure logical OHLC order
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = int(np.random.normal(1000000, 300000))  # Realistic volume
            volume = max(volume, 100000)  # Minimum volume
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def get_market_summary(self):
        """Get a summary of major market movements"""
        summary = {}
        
        for symbol in ['SPY', 'QQQ', 'BTC', 'GLD']:
            data = self.generate_stock_data(symbol, 30)
            if not data.empty:
                current_price = data['close'].iloc[-1]
                prev_price = data['close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                summary[symbol] = {
                    'price': current_price,
                    'change_pct': round(change_pct, 2),
                    'status': 'fallback_data'
                }
        
        return summary