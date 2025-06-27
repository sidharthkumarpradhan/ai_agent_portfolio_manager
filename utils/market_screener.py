import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class IntelligentMarketScreener:
    """
    Advanced market screener that analyzes stocks like a 50-year veteran trader
    Focuses on fundamental analysis, technical indicators, and market opportunities
    """
    
    def __init__(self, data_fetcher=None, alpha_vantage_key=None, finnhub_key=None):
        self.data_fetcher = data_fetcher
        self.alpha_vantage_key = alpha_vantage_key
        self.finnhub_key = finnhub_key
        
        # S&P 500 components (top 100 most liquid and important)
        self.sp500_top100 = [
            # Technology Giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'CRM',
            'ADBE', 'NFLX', 'AMD', 'INTC', 'CSCO', 'QCOM', 'TXN', 'IBM', 'INTU', 'MU',
            
            # Healthcare & Pharma
            'JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'DHR', 'BMY', 'MDT',
            'AMGN', 'GILD', 'CVS', 'CI', 'ISRG', 'SYK', 'BSX', 'REGN', 'VRTX', 'HUM',
            
            # Financial Services
            'BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI',
            'USB', 'TFC', 'PNC', 'COF', 'SCHW', 'CB', 'MMC', 'ICE', 'CME', 'AON',
            
            # Consumer & Retail
            'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'TGT', 'COST', 'WMT',
            'PG', 'KO', 'PEP', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'MDLZ', 'CPB',
            
            # Energy & Utilities
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'OXY', 'DVN', 'HAL',
            'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'PEG', 'ED', 'D', 'PCG',
            
            # Industrial & Materials
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'DE', 'EMR', 'ITW', 'PH', 'ROK', 'ETN', 'CARR', 'OTIS', 'FDX', 'CSX'
        ]
        
        # Market cap categories
        self.large_cap_threshold = 10_000_000_000  # $10B+
        self.mid_cap_threshold = 2_000_000_000     # $2B-$10B
        # Small cap: Under $2B
        
    def get_comprehensive_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive stock data including fundamentals and technicals"""
        try:
            stock_data = {}
            
            # Basic price data
            if self.data_fetcher:
                price_data = self.data_fetcher.get_stock_data(symbol)
                if price_data is not None and not price_data.empty:
                    recent_data = price_data.tail(30)
                    current_price = float(recent_data['close'].iloc[-1])
                    
                    # Calculate basic metrics
                    returns = recent_data['close'].pct_change().dropna()
                    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
                    
                    # Momentum indicators
                    price_30d = float(recent_data['close'].iloc[0]) if len(recent_data) >= 30 else current_price
                    momentum_30d = ((current_price / price_30d) - 1) * 100 if price_30d > 0 else 0
                    
                    stock_data.update({
                        'symbol': symbol,
                        'current_price': current_price,
                        'volatility_annualized': volatility,
                        'momentum_30d': momentum_30d,
                        'avg_volume_30d': float(recent_data['volume'].mean()),
                        'price_data_available': True
                    })
            
            # Fundamental data from Finnhub
            if self.finnhub_key:
                fundamental_data = self._get_finnhub_fundamentals(symbol)
                if fundamental_data:
                    stock_data.update(fundamental_data)
            
            return stock_data if stock_data else None
            
        except Exception as e:
            logger.warning(f"Error getting comprehensive data for {symbol}: {str(e)}")
            return None
    
    def _get_finnhub_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data from Finnhub API"""
        try:
            # Company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={self.finnhub_key}"
            profile_response = requests.get(profile_url, timeout=10)
            
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                
                # Basic metrics
                metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.finnhub_key}"
                metrics_response = requests.get(metrics_url, timeout=10)
                
                fundamental_data = {
                    'company_name': profile_data.get('name', symbol),
                    'industry': profile_data.get('finnhubIndustry', 'Unknown'),
                    'sector': profile_data.get('gicsSector', 'Unknown'),
                    'market_cap': profile_data.get('marketCapitalization', 0) * 1_000_000,  # Convert to dollars
                    'country': profile_data.get('country', 'US'),
                    'ipo_date': profile_data.get('ipo', ''),
                    'employee_count': profile_data.get('employeeTotal', 0)
                }
                
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    metric_values = metrics_data.get('metric', {})
                    
                    # Add key financial metrics
                    fundamental_data.update({
                        'pe_ratio': metric_values.get('peBasicExclExtraTTM', 0),
                        'pb_ratio': metric_values.get('pbQuarterly', 0),
                        'roe': metric_values.get('roeRfy', 0),
                        'roa': metric_values.get('roaRfy', 0),
                        'profit_margin': metric_values.get('profitMarginRfy', 0),
                        'debt_to_equity': metric_values.get('totalDebt/totalEquityQuarterly', 0),
                        'current_ratio': metric_values.get('currentRatioQuarterly', 0),
                        'revenue_growth': metric_values.get('revenueGrowthRfy', 0),
                        'eps_growth': metric_values.get('epsGrowthRfy', 0)
                    })
                
                return fundamental_data
            
        except Exception as e:
            logger.warning(f"Error getting fundamentals for {symbol}: {str(e)}")
        
        return None
    
    def screen_best_opportunities(self, max_stocks: int = 50) -> List[Dict]:
        """
        Screen the market for best investment opportunities
        Uses 50+ years of investing wisdom to select quality stocks
        """
        logger.info(f"Screening S&P 500 top 100 stocks for best opportunities...")
        
        all_stocks_data = []
        processed_count = 0
        
        for symbol in self.sp500_top100:
            if processed_count >= max_stocks:
                break
                
            stock_data = self.get_comprehensive_stock_data(symbol)
            if stock_data and stock_data.get('price_data_available'):
                all_stocks_data.append(stock_data)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} stocks...")
        
        logger.info(f"Successfully analyzed {len(all_stocks_data)} stocks")
        
        # Apply Marcus Wellington's 50-year screening criteria
        scored_stocks = self._apply_veteran_screening_criteria(all_stocks_data)
        
        # Sort by investment score (highest first)
        best_opportunities = sorted(scored_stocks, key=lambda x: x.get('investment_score', 0), reverse=True)
        
        return best_opportunities[:30]  # Return top 30 opportunities
    
    def _apply_veteran_screening_criteria(self, stocks_data: List[Dict]) -> List[Dict]:
        """
        Apply 50 years of Wall Street experience to score stocks
        Focus on quality, growth, value, and momentum
        """
        scored_stocks = []
        
        for stock in stocks_data:
            try:
                score = 0.0
                scoring_details = []
                
                # Quality Score (30% weight)
                quality_score = self._calculate_quality_score(stock)
                if quality_score is not None:
                    score += quality_score * 0.30
                    scoring_details.append(f"Quality: {quality_score:.1f}/10")
                
                # Growth Score (25% weight)
                growth_score = self._calculate_growth_score(stock)
                if growth_score is not None:
                    score += growth_score * 0.25
                    scoring_details.append(f"Growth: {growth_score:.1f}/10")
                
                # Value Score (25% weight)
                value_score = self._calculate_value_score(stock)
                if value_score is not None:
                    score += value_score * 0.25
                    scoring_details.append(f"Value: {value_score:.1f}/10")
                
                # Momentum Score (20% weight)
                momentum_score = self._calculate_momentum_score(stock)
                if momentum_score is not None:
                    score += momentum_score * 0.20
                    scoring_details.append(f"Momentum: {momentum_score:.1f}/10")
                
                # Only include stocks with valid scores
                if score > 0:
                    stock['investment_score'] = round(score, 2)
                    stock['composite_score'] = round(score, 2)  # Add composite_score alias
                    stock['scoring_breakdown'] = scoring_details
                    stock['investment_thesis'] = self._generate_investment_thesis(stock)
                    scored_stocks.append(stock)
                    
            except Exception as e:
                logger.warning(f"Error scoring stock {stock.get('symbol', 'Unknown')}: {str(e)}")
                continue
        
        # Sort by score descending and return top performers
        scored_stocks.sort(key=lambda x: x.get('investment_score', 0), reverse=True)
        return scored_stocks
    
    def _calculate_quality_score(self, stock: Dict) -> float:
        """Calculate quality score based on financial health"""
        score = 5.0  # Base score
        
        # ROE analysis
        roe = stock.get('roe', 0)
        if roe > 20:
            score += 2
        elif roe > 15:
            score += 1
        elif roe < 5:
            score -= 1
        
        # Profit margin
        profit_margin = stock.get('profit_margin', 0)
        if profit_margin > 20:
            score += 1.5
        elif profit_margin > 10:
            score += 0.5
        elif profit_margin < 5:
            score -= 1
        
        # Debt management
        debt_equity = stock.get('debt_to_equity', 0)
        if debt_equity < 0.3:
            score += 1
        elif debt_equity > 1.0:
            score -= 1
        
        # Current ratio (liquidity)
        current_ratio = stock.get('current_ratio', 0)
        if current_ratio > 2:
            score += 0.5
        elif current_ratio < 1:
            score -= 1
        
        return max(0, min(10, score))
    
    def _calculate_growth_score(self, stock: Dict) -> float:
        """Calculate growth potential score"""
        score = 5.0
        
        # Revenue growth
        revenue_growth = stock.get('revenue_growth', 0) * 100
        if revenue_growth > 20:
            score += 2
        elif revenue_growth > 10:
            score += 1
        elif revenue_growth < 0:
            score -= 2
        
        # EPS growth
        eps_growth = stock.get('eps_growth', 0) * 100
        if eps_growth > 15:
            score += 2
        elif eps_growth > 5:
            score += 1
        elif eps_growth < -10:
            score -= 2
        
        # Sector growth (Tech and Healthcare get bonus)
        sector = stock.get('sector', '')
        if 'Technology' in sector or 'Health' in sector:
            score += 0.5
        
        return max(0, min(10, score))
    
    def _calculate_value_score(self, stock: Dict) -> float:
        """Calculate value investment score with safe comparisons"""
        try:
            score = 5.0
            
            # P/E ratio analysis with null safety
            pe_ratio = stock.get('pe_ratio')
            if pe_ratio is not None and pe_ratio > 0:
                pe_ratio = float(pe_ratio)
                if pe_ratio < 15:
                    score += 2
                elif pe_ratio < 25:
                    score += 1
                elif pe_ratio > 40:
                    score -= 1
            
            # P/B ratio with null safety
            pb_ratio = stock.get('pb_ratio')
            if pb_ratio is not None and pb_ratio > 0:
                pb_ratio = float(pb_ratio)
                if pb_ratio < 2:
                    score += 1
                elif pb_ratio > 5:
                    score -= 1
            
            # Market cap consideration with null safety
            market_cap = stock.get('market_cap')
            if market_cap is not None:
                market_cap = float(market_cap)
                if market_cap > self.large_cap_threshold:
                    score += 0.5
            
            return max(0.0, min(10.0, score))
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error calculating value score for {stock.get('symbol', 'Unknown')}: {str(e)}")
            return 5.0
    
    def _calculate_momentum_score(self, stock: Dict) -> float:
        """Calculate momentum and technical score with safe comparisons"""
        try:
            score = 5.0
            
            # 30-day momentum with null safety
            momentum_30d = stock.get('momentum_30d')
            if momentum_30d is not None:
                momentum_30d = float(momentum_30d)
                if momentum_30d > 10:
                    score += 2
                elif momentum_30d > 5:
                    score += 1
                elif momentum_30d < -10:
                    score -= 2
                elif momentum_30d < -5:
                    score -= 1
            
            # Volatility consideration with null safety (lower is better)
            volatility = stock.get('volatility_annualized')
            if volatility is not None:
                volatility = float(volatility)
                if volatility < 0.20:  # 20% annual volatility
                    score += 1
                elif volatility > 0.40:  # 40% annual volatility
                    score -= 1
            
            # Volume consideration
            avg_volume = stock.get('avg_volume_30d')
            if avg_volume is not None:
                avg_volume = float(avg_volume)
                if avg_volume > 5000000:  # High volume
                    score += 0.5
                elif avg_volume < 500000:  # Low volume
                    score -= 0.5
            
            return max(0.0, min(10.0, score))
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating momentum score for {stock.get('symbol', 'Unknown')}: {str(e)}")
            return 5.0
    
    def _generate_investment_thesis(self, stock: Dict) -> str:
        """Generate Marcus Wellington's investment thesis for the stock"""
        symbol = stock.get('symbol', '')
        company_name = stock.get('company_name', symbol)
        sector = stock.get('sector', 'Unknown')
        score = stock.get('investment_score', 0)
        
        if score >= 8:
            strength = "STRONG BUY"
            rationale = "exceptional fundamentals and growth prospects"
        elif score >= 7:
            strength = "BUY"
            rationale = "solid fundamentals with good upside potential"
        elif score >= 6:
            strength = "MODERATE BUY"
            rationale = "decent fundamentals but requires monitoring"
        elif score >= 5:
            strength = "HOLD"
            rationale = "mixed signals, suitable for defensive allocation"
        else:
            strength = "AVOID"
            rationale = "weak fundamentals or overvalued"
        
        return f"{strength} - {company_name} ({sector}): {rationale}. Score: {score}/10"
    
    def get_sector_allocation_recommendations(self, screened_stocks: List[Dict]) -> Dict[str, float]:
        """Get sector allocation recommendations based on current market conditions"""
        sector_scores = {}
        sector_counts = {}
        
        # Calculate average scores by sector
        for stock in screened_stocks:
            sector = stock.get('sector', 'Unknown')
            score = stock.get('investment_score', 0)
            
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(score)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Calculate recommended allocations
        sector_allocations = {}
        total_weight = 0
        
        for sector, scores in sector_scores.items():
            avg_score = sum(scores) / len(scores)
            stock_count = len(scores)
            
            # Weight by both quality and diversification
            allocation_weight = (avg_score / 10) * min(stock_count / 3, 1.0)
            sector_allocations[sector] = allocation_weight
            total_weight += allocation_weight
        
        # Normalize to 100%
        if total_weight > 0:
            for sector in sector_allocations:
                sector_allocations[sector] = (sector_allocations[sector] / total_weight) * 100
        
        return dict(sorted(sector_allocations.items(), key=lambda x: x[1], reverse=True))