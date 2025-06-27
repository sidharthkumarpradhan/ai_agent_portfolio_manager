import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
from datetime import datetime, timedelta

class RiskCalculator:
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]  # 95% and 99% confidence levels
    
    def calculate_var(self, returns, confidence_level=0.95, time_horizon=1):
        """Calculate Value at Risk (VaR)"""
        try:
            if returns.empty:
                return None
            
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate percentile
            index = int((1 - confidence_level) * len(sorted_returns))
            var = sorted_returns[index]
            
            # Scale for time horizon
            var_scaled = var * np.sqrt(time_horizon)
            
            return {
                'var': var_scaled,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
        
        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
            return None
    
    def calculate_cvar(self, returns, confidence_level=0.95, time_horizon=1):
        """Calculate Conditional Value at Risk (CVaR) - Expected Shortfall"""
        try:
            if returns.empty:
                return None
            
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate cutoff index
            index = int((1 - confidence_level) * len(sorted_returns))
            
            # CVaR is the mean of returns below VaR
            cvar = np.mean(sorted_returns[:index])
            
            # Scale for time horizon
            cvar_scaled = cvar * np.sqrt(time_horizon)
            
            return {
                'cvar': cvar_scaled,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
        
        except Exception as e:
            st.error(f"Error calculating CVaR: {str(e)}")
            return None
    
    def calculate_portfolio_risk_metrics(self, portfolio_data, returns_data):
        """Calculate comprehensive risk metrics for the portfolio"""
        try:
            if returns_data.empty or not portfolio_data:
                return None
            
            # Calculate portfolio returns
            weights = []
            symbols = []
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            
            for symbol, data in portfolio_data.items():
                market_value = data.get('market_value', 0)
                weight = market_value / total_value if total_value > 0 else 0
                weights.append(weight)
                symbols.append(symbol)
            
            weights = np.array(weights)
            
            # Calculate returns for symbols in portfolio
            portfolio_returns = []
            for i, symbol in enumerate(symbols):
                if symbol in returns_data.columns:
                    symbol_returns = returns_data[symbol].pct_change().dropna()
                    portfolio_returns.append(symbol_returns * weights[i])
            
            if portfolio_returns:
                portfolio_returns_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            else:
                return None
            
            # Calculate risk metrics
            risk_metrics = {}
            
            # Basic statistics
            risk_metrics['volatility'] = portfolio_returns_series.std() * np.sqrt(252)  # Annualized
            risk_metrics['skewness'] = stats.skew(portfolio_returns_series.dropna())
            risk_metrics['kurtosis'] = stats.kurtosis(portfolio_returns_series.dropna())
            
            # VaR and CVaR
            for confidence_level in self.confidence_levels:
                var_result = self.calculate_var(portfolio_returns_series, confidence_level)
                cvar_result = self.calculate_cvar(portfolio_returns_series, confidence_level)
                
                if var_result and cvar_result:
                    risk_metrics[f'var_{int(confidence_level*100)}'] = var_result['var']
                    risk_metrics[f'cvar_{int(confidence_level*100)}'] = cvar_result['cvar']
            
            # Maximum Drawdown
            drawdown_result = self.calculate_max_drawdown(portfolio_returns_series)
            if drawdown_result:
                risk_metrics['max_drawdown'] = drawdown_result['max_drawdown']
            
            # Beta relative to market (using SPY as proxy)
            if 'SPY' in returns_data.columns:
                spy_returns = returns_data['SPY'].pct_change().dropna()
                aligned_returns = pd.concat([portfolio_returns_series, spy_returns], axis=1).dropna()
                if len(aligned_returns) > 30:  # Need sufficient data
                    covariance = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])[0, 1]
                    market_variance = np.var(aligned_returns.iloc[:, 1])
                    risk_metrics['beta'] = covariance / market_variance if market_variance != 0 else 0
                else:
                    risk_metrics['beta'] = None
            else:
                risk_metrics['beta'] = None
            
            # Tracking Error (if benchmark available)
            if 'SPY' in returns_data.columns:
                spy_returns = returns_data['SPY'].pct_change().dropna()
                aligned_returns = pd.concat([portfolio_returns_series, spy_returns], axis=1).dropna()
                if len(aligned_returns) > 30:
                    tracking_error = (aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]).std() * np.sqrt(252)
                    risk_metrics['tracking_error'] = tracking_error
                else:
                    risk_metrics['tracking_error'] = None
            else:
                risk_metrics['tracking_error'] = None
            
            return risk_metrics
        
        except Exception as e:
            st.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return None
    
    def calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown"""
        try:
            if returns_series.empty:
                return None
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_series).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            
            # Find drawdown periods
            drawdown_start = None
            drawdown_end = None
            max_dd_end = drawdown.idxmin()
            
            # Find the start of the max drawdown period
            for i in range(len(cumulative_returns[:max_dd_end])):
                if cumulative_returns.iloc[i] == running_max.loc[max_dd_end]:
                    drawdown_start = cumulative_returns.index[i]
            
            return {
                'max_drawdown': max_drawdown,
                'drawdown_series': drawdown,
                'drawdown_start': drawdown_start,
                'drawdown_end': max_dd_end,
                'recovery_date': None  # Would need to calculate when drawdown recovered
            }
        
        except Exception as e:
            st.error(f"Error calculating max drawdown: {str(e)}")
            return None
    
    def calculate_correlation_risk(self, returns_data, portfolio_data):
        """Calculate correlation-based risk metrics"""
        try:
            if returns_data.empty or not portfolio_data:
                return None
            
            # Get symbols in portfolio
            portfolio_symbols = list(portfolio_data.keys())
            
            # Filter returns data for portfolio symbols
            portfolio_returns = returns_data[portfolio_symbols].pct_change().dropna()
            
            if portfolio_returns.empty:
                return None
            
            # Calculate correlation matrix
            correlation_matrix = portfolio_returns.corr()
            
            # Calculate average correlation
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    correlations.append(correlation_matrix.iloc[i, j])
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Calculate concentration risk (Herfindahl-Hirschman Index)
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            weights = []
            for symbol, data in portfolio_data.items():
                weight = data.get('market_value', 0) / total_value if total_value > 0 else 0
                weights.append(weight)
            
            hhi = sum([w**2 for w in weights])
            
            # Risk assessment
            correlation_risk = "High" if avg_correlation > 0.7 else "Medium" if avg_correlation > 0.4 else "Low"
            concentration_risk = "High" if hhi > 0.25 else "Medium" if hhi > 0.15 else "Low"
            
            return {
                'correlation_matrix': correlation_matrix,
                'average_correlation': avg_correlation,
                'correlation_risk_level': correlation_risk,
                'concentration_index': hhi,
                'concentration_risk_level': concentration_risk,
                'diversification_ratio': 1 / hhi if hhi > 0 else 0
            }
        
        except Exception as e:
            st.error(f"Error calculating correlation risk: {str(e)}")
            return None
    
    def stress_test_portfolio(self, portfolio_data, returns_data, stress_scenarios):
        """Perform stress testing on the portfolio"""
        try:
            if returns_data.empty or not portfolio_data:
                return None
            
            # Calculate portfolio weights
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            weights = {}
            for symbol, data in portfolio_data.items():
                weight = data.get('market_value', 0) / total_value if total_value > 0 else 0
                weights[symbol] = weight
            
            stress_results = {}
            
            # Default stress scenarios if none provided
            if not stress_scenarios:
                stress_scenarios = {
                    'Market Crash': {'all_stocks': -0.30, 'BTC-USD': -0.50},
                    'Tech Selloff': {'AAPL': -0.25, 'MSFT': -0.25, 'GOOGL': -0.30, 'TSLA': -0.40},
                    'Crypto Crash': {'BTC-USD': -0.70},
                    'Interest Rate Shock': {'all_stocks': -0.15, 'BTC-USD': -0.25},
                    'Recession': {'all_stocks': -0.20, 'BTC-USD': -0.30}
                }
            
            for scenario_name, scenario_shocks in stress_scenarios.items():
                portfolio_shock = 0
                
                for symbol, weight in weights.items():
                    if symbol in scenario_shocks:
                        shock = scenario_shocks[symbol]
                    elif 'all_stocks' in scenario_shocks and symbol != 'BTC-USD':
                        shock = scenario_shocks['all_stocks']
                    else:
                        shock = 0
                    
                    portfolio_shock += weight * shock
                
                stress_results[scenario_name] = {
                    'portfolio_impact': portfolio_shock,
                    'dollar_impact': portfolio_shock * total_value,
                    'new_portfolio_value': total_value * (1 + portfolio_shock)
                }
            
            return stress_results
        
        except Exception as e:
            st.error(f"Error performing stress test: {str(e)}")
            return None
    
    def calculate_risk_score(self, risk_metrics):
        """Calculate overall risk score (0-100, higher = riskier)"""
        try:
            if not risk_metrics:
                return None
            
            score = 0
            factors = 0
            
            # Volatility component (0-30 points)
            if 'volatility' in risk_metrics and risk_metrics['volatility'] is not None:
                vol = risk_metrics['volatility']
                vol_score = min(30, vol * 100)  # Scale volatility
                score += vol_score
                factors += 1
            
            # VaR component (0-25 points)
            if 'var_95' in risk_metrics and risk_metrics['var_95'] is not None:
                var = abs(risk_metrics['var_95'])
                var_score = min(25, var * 500)  # Scale VaR
                score += var_score
                factors += 1
            
            # Max Drawdown component (0-25 points)
            if 'max_drawdown' in risk_metrics and risk_metrics['max_drawdown'] is not None:
                dd = abs(risk_metrics['max_drawdown'])
                dd_score = min(25, dd * 100)  # Scale drawdown
                score += dd_score
                factors += 1
            
            # Concentration component (0-20 points)
            # This would need additional data about portfolio concentration
            
            # Average the score
            if factors > 0:
                final_score = score / factors * (100/30)  # Normalize to 0-100
                return min(100, max(0, final_score))
            
            return None
        
        except Exception as e:
            st.error(f"Error calculating risk score: {str(e)}")
            return None
