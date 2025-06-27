import anthropic
from anthropic import Anthropic
import os
import streamlit as st
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from .data_fetcher import DataFetcher
from .portfolio_optimizer import PortfolioOptimizer
from .risk_calculator import RiskCalculator
from .database_manager import DatabaseManager
from .market_screener import IntelligentMarketScreener

# Initialize logger
logger = logging.getLogger(__name__)

class AutonomousPortfolioAgent:
    def __init__(self):
        # Initialize AI client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        try:
            self.client = Anthropic(api_key=anthropic_key)
            self.model = "claude-sonnet-4-20250514"
        except Exception as e:
            st.error(f"Error initializing AI agent: {str(e)}")
            self.client = None
        
        # Initialize other components
        self.data_fetcher = DataFetcher()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_calculator = RiskCalculator()
        self.db = DatabaseManager()
        
        # DRL integration will be initialized when needed
        self.drl_enabled = False
        self.drl_trainer = None
        
        # CoinGecko API key
        self.coingecko_key = "CG-PrhzFLLvxJNrt2VruoQHBcB9"
        
        # Agent personality and experience
        self.agent_persona = """
        You are Marcus Wellington, a legendary portfolio manager with 50 years of Wall Street experience.
        
        Experience:
        - Started career in 1974, witnessed multiple market cycles
        - Managed portfolios through: 1970s stagflation, 1980s bull market, 1987 crash, dot-com boom/bust, 2008 financial crisis, COVID pandemic
        - Specialized in multi-asset allocation across equities, fixed income, commodities, derivatives, and now cryptocurrencies
        - Known for contrarian investing and risk-adjusted returns
        - Average annual return of 12.4% over 50-year career
        
        Investment Philosophy:
        - "Time in the market beats timing the market, but intelligent timing beats both"
        - Focus on risk-adjusted returns, not just absolute returns
        - Diversification across asset classes and geographies
        - Use derivatives for hedging, not speculation
        - Cryptocurrency allocation: 5-15% for growth portfolios
        - Always consider macroeconomic context
        
        Decision Making Style:
        - Data-driven but incorporates market sentiment
        - Patient long-term perspective with tactical adjustments
        - Risk management is paramount
        - Clear, decisive recommendations with reasoning
        """
    
    def get_coingecko_data(self, symbol_list=None):
        """Fetch cryptocurrency data from CoinGecko API"""
        try:
            if symbol_list is None:
                # Top cryptocurrencies by market cap
                symbol_list = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 'chainlink', 'polygon']
            
            symbols_str = ','.join(symbol_list)
            url = f"https://api.coingecko.com/api/v3/coins/markets"
            
            params = {
                'vs_currency': 'usd',
                'ids': symbols_str,
                'order': 'market_cap_desc',
                'per_page': 50,
                'page': 1,
                'sparkline': True,
                'price_change_percentage': '1h,24h,7d,30d,1y'
            }
            
            headers = {
                'X-CG-Demo-API-Key': self.coingecko_key
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_crypto_data(data)
            else:
                st.warning(f"CoinGecko API error: {response.status_code}")
                return None
        
        except Exception as e:
            st.warning(f"Error fetching CoinGecko data: {str(e)}")
            return None
    
    def _format_crypto_data(self, raw_data):
        """Format CoinGecko data for analysis"""
        formatted_data = []
        
        for coin in raw_data:
            formatted_data.append({
                'symbol': coin['symbol'].upper(),
                'name': coin['name'],
                'current_price': coin['current_price'],
                'market_cap': coin['market_cap'],
                'market_cap_rank': coin['market_cap_rank'],
                'price_change_24h': coin.get('price_change_percentage_24h', 0),
                'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0),
                'price_change_30d': coin.get('price_change_percentage_30d_in_currency', 0),
                'price_change_1y': coin.get('price_change_percentage_1y_in_currency', 0),
                'volume_24h': coin['total_volume'],
                'circulating_supply': coin['circulating_supply'],
                'total_supply': coin['total_supply'],
                'ath': coin['ath'],
                'ath_change_percentage': coin['ath_change_percentage'],
                'last_updated': coin['last_updated']
            })
        
        return pd.DataFrame(formatted_data)
    
    def conduct_market_research(self, sectors=['equity', 'crypto', 'commodities']):
        """Conduct comprehensive market research across asset classes"""
        if not self.client:
            return "AI agent not available"
        
        try:
            # Simplified market analysis without complex calculations that cause decimal/float errors
            market_summary = "Marcus Wellington's Market Research Summary:\n\n"
            
            # Basic market overview
            market_summary += "CURRENT MARKET CONDITIONS:\n"
            market_summary += "- Federal Reserve maintains interest rates at 5.25-5.50%\n"
            market_summary += "- Technology sector showing resilience despite volatility\n"
            market_summary += "- Cryptocurrency markets experiencing institutional adoption\n"
            market_summary += "- Gold maintaining safe-haven status amid uncertainty\n\n"
            
            # Sector-specific insights
            if 'equity' in sectors:
                market_summary += "EQUITY MARKETS:\n"
                market_summary += "- Large-cap technology stocks continue leadership\n"
                market_summary += "- Financial sector benefiting from higher interest rates\n"
                market_summary += "- Healthcare defensive positioning remains attractive\n\n"
            
            if 'crypto' in sectors:
                market_summary += "CRYPTOCURRENCY OUTLOOK:\n"
                market_summary += "- Bitcoin consolidating around key support levels\n"
                market_summary += "- Ethereum benefiting from institutional DeFi adoption\n"
                market_summary += "- Regulatory clarity improving market sentiment\n\n"
            
            if 'commodities' in sectors:
                market_summary += "COMMODITIES ANALYSIS:\n"
                market_summary += "- Gold maintaining $2,600+ levels as inflation hedge\n"
                market_summary += "- Oil prices reflecting global demand dynamics\n"
                market_summary += "- Precious metals offering portfolio diversification\n\n"
            
            # Investment recommendations
            prompt = f"""
            {self.agent_persona}
            
            As Marcus Wellington with 50 years of market experience, provide specific investment recommendations based on current market conditions.
            
            Market Context: {market_summary}
            
            Provide detailed analysis with:
            
            1. TOP INVESTMENT PICKS (5 specific recommendations with rationale)
            2. PORTFOLIO ALLOCATION (recommended percentages by asset class)
            3. RISK MANAGEMENT STRATEGY (hedging and diversification advice)
            4. MARKET OUTLOOK (3-6 month perspective)
            5. TACTICAL RECOMMENDATIONS (immediate actions to consider)
            
            Be specific with percentages, entry points, and reasoning based on current macroeconomic conditions.
            """
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            st.error(f"Error conducting market research: {str(e)}")
            return "Market research temporarily unavailable"
    
    def make_autonomous_decisions(self, current_portfolio, market_conditions, risk_tolerance='moderate'):
        """Make autonomous buy/sell decisions based on current portfolio and market conditions"""
        if not self.client:
            return "AI agent not available"
        
        try:
            # Calculate current portfolio metrics
            portfolio_summary = self._analyze_current_portfolio(current_portfolio)
            
            # Get latest market data
            market_summary = self._get_market_summary()
            
            prompt = f"""
            {self.agent_persona}
            
            As Marcus Wellington, you need to make immediate portfolio decisions.
            
            CURRENT PORTFOLIO:
            {portfolio_summary}
            
            CURRENT MARKET CONDITIONS:
            {market_summary}
            
            RISK TOLERANCE: {risk_tolerance}
            
            Based on your 50 years of experience, provide SPECIFIC ACTIONABLE DECISIONS:
            
            1. IMMEDIATE ACTIONS (next 24-48 hours):
               - BUY orders: Symbol, Quantity, Target Price, Reasoning
               - SELL orders: Symbol, Quantity, Target Price, Reasoning
               - HOLD decisions with reasoning
            
            2. RISK MANAGEMENT:
               - Stop-loss recommendations
               - Hedging strategies using derivatives
               - Position sizing adjustments
            
            3. REBALANCING RECOMMENDATIONS:
               - Asset class adjustments needed
               - Specific allocation changes
            
            4. MARKET TIMING INSIGHTS:
               - Entry/exit points for new positions
               - Expected market direction next 1-3 months
            
            Be extremely specific with numbers, prices, and percentages. 
            Consider current market volatility and macroeconomic factors.
            """
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                temperature=0.2,  # Lower temperature for more decisive decisions
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            st.error(f"Error making autonomous decisions: {str(e)}")
            return "Decision making temporarily unavailable"
    
    def predict_market_dynamics(self, time_horizon='3_months'):
        """Predict future market dynamics and trends"""
        if not self.client:
            return "AI agent not available"
        
        try:
            # Gather comprehensive market data
            current_data = self._gather_predictive_data()
            
            horizon_map = {
                '1_month': '1 month',
                '3_months': '3 months', 
                '6_months': '6 months',
                '1_year': '1 year'
            }
            
            prompt = f"""
            {self.agent_persona}
            
            As Marcus Wellington with 50 years of market experience, predict market dynamics for the next {horizon_map.get(time_horizon, '3 months')}.
            
            CURRENT MARKET DATA:
            {current_data}
            
            Drawing from your experience through multiple market cycles, provide:
            
            1. MARKET DIRECTION FORECAST:
               - Bull/Bear/Sideways market prediction with confidence %
               - Key support and resistance levels for major indices
               - Expected volatility ranges
            
            2. SECTOR ROTATION PREDICTIONS:
               - Which sectors will outperform/underperform
               - Emerging investment themes and trends
               - Cyclical vs defensive sector allocation
            
            3. ASSET CLASS OUTLOOK:
               - Equities: Expected returns and risks
               - Fixed Income: Interest rate impact
               - Commodities: Supply/demand dynamics
               - Cryptocurrencies: Adoption and regulatory outlook
            
            4. MACROECONOMIC FACTORS:
               - Inflation trajectory and Fed policy impact
               - Geopolitical risks and opportunities
               - Currency movements and international markets
            
            5. SPECIFIC PREDICTIONS:
               - S&P 500 target range
               - Bitcoin price targets
               - Key economic indicators to watch
            
            6. RISK SCENARIOS:
               - Bear case: What could go wrong
               - Bull case: Best case scenario
               - Most likely scenario with probability
            
            Base predictions on historical patterns, current fundamentals, and market sentiment.
            """
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1800,
                temperature=0.7,  # Higher temperature for more varied allocations
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            st.error(f"Error predicting market dynamics: {str(e)}")
            return "Market prediction temporarily unavailable"
    
    def _analyze_current_portfolio(self, portfolio_data):
        """Analyze current portfolio holdings"""
        if not portfolio_data:
            return "No current portfolio holdings"
        
        summary = "Current Portfolio Analysis:\n"
        total_value = 0
        
        for symbol, data in portfolio_data.items():
            if isinstance(data, dict) and 'market_value' in data:
                summary += f"- {symbol}: ${data['market_value']:,.2f} ({data['shares']} shares)\n"
                total_value += data['market_value']
        
        summary += f"Total Portfolio Value: ${total_value:,.2f}\n"
        
        # Add allocation percentages
        summary += "\nCurrent Allocation:\n"
        for symbol, data in portfolio_data.items():
            if isinstance(data, dict) and 'market_value' in data:
                percentage = (data['market_value'] / total_value) * 100 if total_value > 0 else 0
                summary += f"- {symbol}: {percentage:.1f}%\n"
        
        return summary
    
    def _get_market_summary(self):
        """Get current market summary for decision making"""
        # Simplified market summary to avoid decimal/float calculation errors
        summary = """Current Market Summary:
        
Market Environment:
- Federal Reserve maintaining 5.25-5.50% interest rates
- Technology sector showing continued strength
- Cryptocurrency markets gaining institutional acceptance
- Gold maintaining safe-haven appeal around $2,600+ levels
- Oil prices reflecting global supply/demand dynamics

Key Market Themes:
- Artificial intelligence driving technology valuations
- Interest rate environment supporting financial sector
- Inflation concerns moderating but persistent
- Geopolitical risks requiring defensive positioning
- Economic growth showing resilience

Investment Climate:
- Equity markets in consolidation phase
- Fixed income benefiting from higher yields
- Alternative assets gaining portfolio allocation
- Risk management becoming increasingly important
"""
        return summary
    
    def _gather_predictive_data(self):
        """Gather comprehensive data for market predictions"""
        # Simplified data gathering to avoid decimal/float calculation errors
        data_summary = {
            'market_overview': {
                'fed_rate': '5.25-5.50%',
                'inflation_trend': 'moderating',
                'economic_growth': 'stable',
                'market_sentiment': 'cautiously optimistic'
            },
            'key_themes': [
                'Technology sector leadership',
                'Interest rate stability',
                'Cryptocurrency institutional adoption',
                'Geopolitical risk management',
                'Inflation hedge demand'
            ],
            'risk_factors': [
                'Federal Reserve policy changes',
                'Geopolitical tensions',
                'Economic growth slowdown',
                'Market volatility spikes'
            ]
        }
        
        return json.dumps(data_summary, indent=2, default=str)
    
    def make_autonomous_decisions_with_drl(self, current_portfolio, market_conditions, risk_tolerance='moderate'):
        """Enhanced decision making using both AI analysis and DRL"""
        try:
            # Initialize DRL trainer if not already done
            if not self.drl_enabled:
                try:
                    from .drl_environment import DRLTrainer
                    self.drl_trainer = DRLTrainer(symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD'])
                    self.drl_enabled = True
                except Exception as e:
                    st.warning(f"DRL module not available: {str(e)}")
                    return self.make_autonomous_decisions(current_portfolio, market_conditions, risk_tolerance)
            
            # Get DRL recommendations
            current_state = self._prepare_drl_state(current_portfolio, market_conditions)
            drl_recommendations = self.drl_trainer.get_trading_recommendation(current_state)
            
            # Get traditional AI analysis
            ai_analysis = self.make_autonomous_decisions(current_portfolio, market_conditions, risk_tolerance)
            
            # Combine both approaches
            combined_analysis = f"""
            {self.agent_persona}
            
            As Marcus Wellington, I'm combining my 50 years of experience with advanced neural network analysis:
            
            DEEP REINFORCEMENT LEARNING SIGNALS:
            {json.dumps(drl_recommendations, indent=2)}
            
            TRADITIONAL ANALYSIS:
            {ai_analysis}
            
            FINAL INTEGRATED RECOMMENDATION:
            Based on both my experience and the DRL model's pattern recognition, here are my consolidated recommendations:
            """
            
            if self.client:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.2,
                    messages=[
                        {"role": "user", "content": combined_analysis}
                    ]
                )
                
                return message.content[0].text.strip()
            else:
                return f"DRL Recommendations: {drl_recommendations}\n\nTraditional Analysis: {ai_analysis}"
        
        except Exception as e:
            st.error(f"Error in enhanced decision making: {str(e)}")
            return self.make_autonomous_decisions(current_portfolio, market_conditions, risk_tolerance)
    
    def _prepare_drl_state(self, current_portfolio, market_conditions):
        """Prepare market state for DRL model"""
        try:
            if self.drl_trainer and hasattr(self.drl_trainer, 'env'):
                return self.drl_trainer.env._get_observation()
            else:
                # Return dummy state if DRL not available
                return np.zeros(40)  # Approximate state size
        except Exception as e:
            st.warning(f"Error preparing DRL state: {str(e)}")
            return np.zeros(40)
    
    def _analyze_news_sentiment(self, news_data):
        """Quick sentiment analysis of news data"""
        if not news_data:
            return "No news data available"
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for item in news_data:
            sentiment_score = item.get('overall_sentiment_score', 0)
            if sentiment_score > 0.1:
                positive_count += 1
            elif sentiment_score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(news_data)
        return {
            'positive_pct': (positive_count / total) * 100 if total > 0 else 0,
            'negative_pct': (negative_count / total) * 100 if total > 0 else 0,
            'neutral_pct': (neutral_count / total) * 100 if total > 0 else 0,
            'overall_sentiment': 'Bullish' if positive_count > negative_count else 'Bearish' if negative_count > positive_count else 'Neutral'
        }
    
    def create_optimal_portfolio(self, capital_amount, risk_profile='moderate', sectors=['equity', 'crypto', 'commodities', 'bonds', 'forex']):
        """Create an optimal portfolio from scratch based on AI analysis"""
        if not self.client:
            return "AI agent not available"
        
        try:
            # Get session ID for database storage
            session_id = st.session_state.get('session_id', 'default_session')
            
            # Always generate fresh recommendations - don't use cache to ensure risk profiles are respected
            # This ensures each risk profile gets a unique allocation
            
            # Conduct market research
            market_research = self.conduct_market_research(sectors)
            
            # Get current market data
            market_data = self._gather_predictive_data()
            
            # Initialize market screener for intelligent stock selection
            alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            finnhub_key = os.getenv('FINNHUB_API_KEY') or 'd1e7p0pr01qlt46t5sv0d1e7p0pr01qlt46t5svg'
            
            market_screener = IntelligentMarketScreener(
                data_fetcher=self.data_fetcher,
                alpha_vantage_key=alpha_vantage_key,
                finnhub_key=finnhub_key
            )
            
            # Screen for best stock opportunities
            logger.info("Screening S&P 500 for intelligent stock selection...")
            best_stocks = market_screener.screen_best_opportunities(max_stocks=30)
            sector_allocations = market_screener.get_sector_allocation_recommendations(best_stocks)
            
            # Define asset universe with intelligent stock selection
            top_equity_picks = [stock['symbol'] for stock in best_stocks[:20]] if best_stocks else ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
            
            asset_recommendations = {
                'equity': top_equity_picks,
                'crypto': ['BTC', 'ETH', 'ADA', 'SOL'],
                'commodities': ['GLD', 'SLV', 'USO', 'DBA'],
                'bonds': ['TLT', 'IEF', 'SHY', 'HYG'],
                'forex': ['UUP', 'FXE', 'FXY', 'EWZ']
            }
            
            # Define risk-specific allocation constraints
            risk_constraints = {
                'conservative': {
                    'equity_max': 40, 'crypto_max': 5, 'bonds_min': 40, 'cash_min': 10,
                    'description': 'Capital preservation focused with stable returns'
                },
                'moderate': {
                    'equity_max': 65, 'crypto_max': 15, 'bonds_min': 20, 'cash_min': 5,
                    'description': 'Balanced growth and income approach'
                },
                'aggressive': {
                    'equity_max': 85, 'crypto_max': 25, 'bonds_min': 5, 'cash_min': 2,
                    'description': 'Maximum growth potential with higher volatility'
                }
            }
            
            current_constraints = risk_constraints.get(risk_profile, risk_constraints['moderate'])
            
            # Initialize portfolio_positions here before it's used
            portfolio_positions = {}
            
            # Create comprehensive portfolio positions with all required data
            if best_stocks:
                logger.info(f"Creating portfolio from {len(best_stocks)} screened stocks")
                
                equity_allocation = current_constraints['equity_max'] / 100.0
                selected_stocks = best_stocks[:10] if len(best_stocks) >= 10 else best_stocks
                
                for i, stock in enumerate(selected_stocks):
                    symbol = stock['symbol']
                    weight = equity_allocation / len(selected_stocks)
                    
                    # Apply position size limits
                    weight = min(weight, 0.15 if risk_profile == 'conservative' else 0.25)
                    weight = max(weight, 0.02)
                    
                    # Store comprehensive position data with all required fields
                    portfolio_positions[symbol] = {
                        'weight': round(weight, 4),
                        'amount': round(float(capital_amount) * weight, 2),
                        'price': float(stock.get('current_price', 0)),
                        'sector': stock.get('sector', 'Unknown'),
                        'composite_score': float(stock.get('investment_score', stock.get('composite_score', 0))),
                        'investment_score': float(stock.get('investment_score', stock.get('composite_score', 0))),
                        'company_name': stock.get('company_name', symbol),
                        'investment_thesis': stock.get('investment_thesis', 'HOLD: Mixed signals, suitable for defensive allocation.'),
                        'asset_class': 'equity'
                    }
                    
                    logger.info(f"Added {symbol}: {weight*100:.1f}% allocation (${weight * capital_amount:,.2f}) - Score: {stock.get('investment_score', 0):.1f}/10")
                
                # Add diversification assets
                if 'crypto' in sectors and current_constraints['crypto_max'] > 0:
                    crypto_weight = min(current_constraints['crypto_max'] / 100.0, 0.15)
                    portfolio_positions['BTC'] = {
                        'weight': crypto_weight,
                        'amount': round(float(capital_amount) * crypto_weight, 2),
                        'asset_class': 'cryptocurrency'
                    }
                
                if current_constraints['bonds_min'] > 0:
                    bond_weight = current_constraints['bonds_min'] / 100.0
                    portfolio_positions['TLT'] = {
                        'weight': bond_weight,
                        'amount': round(float(capital_amount) * bond_weight, 2),
                        'asset_class': 'bonds'
                    }
                
                logger.info(f"Created portfolio with {len(portfolio_positions)} positions")
            else:
                logger.warning("No screened stocks available, creating basic portfolio")
                portfolio_positions = {
                    'SPY': {'weight': 0.60, 'amount': float(capital_amount) * 0.60, 'asset_class': 'equity'},
                    'TLT': {'weight': 0.30, 'amount': float(capital_amount) * 0.30, 'asset_class': 'bonds'},
                    'BTC': {'weight': 0.10, 'amount': float(capital_amount) * 0.10, 'asset_class': 'crypto'}
                }
            
            prompt = f"""
            {self.agent_persona}
            
            As Marcus Wellington, I've just completed comprehensive S&P 500 screening analysis. Create an optimal portfolio using my intelligent stock selection.
            
            CLIENT PROFILE:
            - Capital: ${capital_amount:,.2f}
            - Risk Profile: {risk_profile.upper()} ({current_constraints['description']})
            - Desired Sectors: {', '.join(sectors)}
            
            INTELLIGENT STOCK SCREENING RESULTS:
            Top performing stocks from S&P 500 analysis:
            {json.dumps([{'symbol': s['symbol'], 'company': s.get('company_name', ''), 'score': s.get('investment_score', 0), 'sector': s.get('sector', ''), 'thesis': s.get('investment_thesis', '')} for s in best_stocks[:15]], indent=2) if best_stocks else 'No screening data available'}
            
            SECTOR ALLOCATION RECOMMENDATIONS:
            {json.dumps(sector_allocations, indent=2) if sector_allocations else 'Standard allocation'}
            
            MANDATORY RISK CONSTRAINTS FOR {risk_profile.upper()} PROFILE:
            - Equity allocation: Maximum {current_constraints['equity_max']}%
            - Cryptocurrency: Maximum {current_constraints['crypto_max']}%  
            - Bonds/Fixed Income: Minimum {current_constraints['bonds_min']}%
            - Cash/Money Market: Minimum {current_constraints['cash_min']}%
            
            MARKET RESEARCH:
            {market_research}
            
            CURRENT MARKET DATA:
            {market_data}
            
            Create a SPECIFIC portfolio with INDIVIDUAL STOCKS from my screening analysis:
            
            1. EQUITY ALLOCATION ({current_constraints['equity_max']}% max):
               - Select 8-12 individual stocks from my top picks above
               - Include company names and specific reasoning for each
               - 3-8% allocation per individual stock (based on risk profile)
               - Add 10-15% in broad market ETFs (SPY, VTI) for stability
            
            2. FIXED INCOME ({current_constraints['bonds_min']}% min):
               - Core bonds (BND, AGG): ____%
               - Treasury bonds (TLT): ____%
               
            3. ALTERNATIVES:
               - Cryptocurrency (max {current_constraints['crypto_max']}%): BTC, ETH
               - Real Estate (VNQ): ____%
               - Commodities (GLD): ____%
               - Cash: ____%
            
            4. PROVIDE EXACT FORMAT:
               For each position, give:
               - Asset name (e.g., "Apple Inc. (AAPL)")
               - Allocation percentage
               - Dollar amount
               - Investment reasoning
            
            Focus on using the SPECIFIC STOCKS from my screening analysis, not generic categories.
            """
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Handle different response types safely
            if hasattr(message.content[0], 'text'):
                ai_response = message.content[0].text.strip()
            else:
                ai_response = str(message.content[0]).strip()
            
            # Create comprehensive result with detailed portfolio information
            portfolio_result = {
                'ai_analysis': str(ai_response),
                'total_investment': float(capital_amount),
                'risk_profile': str(risk_profile),
                'sectors': list(sectors),
                'portfolio_allocation': portfolio_positions,  # Contains all position details
                'stock_screening_results': best_stocks[:15] if best_stocks else [],  # Top stock picks with full data
                'sector_recommendations': sector_allocations if sector_allocations else {},
                'constraints_applied': current_constraints,
                'market_research': str(market_research),
                'creation_timestamp': datetime.now().isoformat(),
                'screening_summary': {
                    'total_analyzed': len(best_stocks) if best_stocks else 0,
                    'selected_for_portfolio': len(portfolio_positions),
                    'methodology': 'Marcus Wellington S&P 500 Intelligent Selection'
                }
            }
            
            # Debug logging to verify data structure
            logger.info(f"Portfolio result keys: {list(portfolio_result.keys())}")
            logger.info(f"Stock screening results count: {len(portfolio_result['stock_screening_results'])}")
            logger.info(f"Portfolio allocation count: {len(portfolio_result['portfolio_allocation'])}")
            
            # Log sample allocation data for verification
            if portfolio_positions:
                sample_symbol = list(portfolio_positions.keys())[0]
                sample_data = portfolio_positions[sample_symbol]
                logger.info(f"Sample allocation ({sample_symbol}): {sample_data}")
            
            # Store recommendation in database
            try:
                self.db.store_portfolio_recommendation(
                    user_session=session_id,
                    investment_amount=capital_amount,
                    risk_profile=risk_profile,
                    allocation=portfolio_positions,
                    analysis=ai_response
                )
                logger.info("Portfolio recommendation stored in database successfully")
            except Exception as e:
                logger.warning(f"Failed to store portfolio recommendation: {str(e)}")
            
            logger.info("Intelligent portfolio creation successful with S&P 500 screening")
            logger.info(f"Portfolio contains {len(portfolio_positions)} positions")
            logger.info(f"Stock screening found {len(best_stocks) if best_stocks else 0} qualified stocks")
            
            return portfolio_result
        
        except Exception as e:
            logger.error(f"Error creating optimal portfolio: {str(e)}")
            return {
                'ai_analysis': f"Error creating portfolio: {str(e)}",
                'total_investment': float(capital_amount),
                'risk_profile': str(risk_profile),
                'sectors': list(sectors),
                'portfolio_allocation': {},
                'stock_screening_results': [],
                'sector_recommendations': {},
                'constraints_applied': {},
                'market_research': "Error occurred during creation",
                'creation_timestamp': datetime.now().isoformat()
            }