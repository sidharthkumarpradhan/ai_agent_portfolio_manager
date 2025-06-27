import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        try:
            portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return None
    
    def optimize_portfolio(self, returns_data, optimization_type='sharpe'):
        """
        Optimize portfolio allocation
        optimization_type: 'sharpe', 'min_vol', 'max_return'
        """
        try:
            if returns_data.empty:
                return None
            
            # Calculate returns and covariance matrix
            returns = returns_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            num_assets = len(returns.columns)
            
            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1/num_assets] * num_assets)
            
            # Objective functions
            def negative_sharpe_ratio(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return -metrics['sharpe_ratio'] if metrics else float('inf')
            
            def portfolio_volatility(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return metrics['volatility'] if metrics else float('inf')
            
            def negative_return(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return -metrics['return'] if metrics else float('inf')
            
            # Choose objective function
            if optimization_type == 'sharpe':
                objective = negative_sharpe_ratio
            elif optimization_type == 'min_vol':
                objective = portfolio_volatility
            elif optimization_type == 'max_return':
                objective = negative_return
            else:
                objective = negative_sharpe_ratio
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(optimal_weights, returns, cov_matrix)
                
                return {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'metrics': metrics,
                    'success': True
                }
            else:
                return {'success': False, 'message': 'Optimization failed'}
        
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")
            return None
    
    def generate_efficient_frontier(self, returns_data, num_portfolios=100):
        """Generate efficient frontier data"""
        try:
            if returns_data.empty:
                return None
            
            returns = returns_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(returns.columns)
            
            # Target returns for efficient frontier
            min_ret = mean_returns.min() * 252
            max_ret = mean_returns.max() * 252
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x, target=target_return: 
                     np.sum(mean_returns * x) * 252 - target}
                ]
                
                bounds = tuple((0, 1) for _ in range(num_assets))
                initial_guess = np.array([1/num_assets] * num_assets)
                
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                
                result = minimize(
                    portfolio_volatility,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    weights = result.x
                    volatility = portfolio_volatility(weights)
                    sharpe = (target_return - self.risk_free_rate) / volatility
                    
                    efficient_portfolios.append({
                        'return': target_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        'weights': dict(zip(returns.columns, weights))
                    })
            
            return efficient_portfolios
        
        except Exception as e:
            st.error(f"Error generating efficient frontier: {str(e)}")
            return None
    
    def calculate_current_allocation(self, portfolio_data):
        """Calculate current portfolio allocation percentages"""
        try:
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            
            if total_value == 0:
                return {}
            
            allocation = {}
            for symbol, data in portfolio_data.items():
                market_value = data.get('market_value', 0)
                allocation[symbol] = market_value / total_value
            
            return allocation
        
        except Exception as e:
            st.error(f"Error calculating current allocation: {str(e)}")
            return {}
    
    def suggest_rebalancing(self, current_portfolio, target_allocation, total_value):
        """Suggest specific rebalancing actions"""
        try:
            current_allocation = self.calculate_current_allocation(current_portfolio)
            suggestions = []
            
            for symbol in set(list(current_allocation.keys()) + list(target_allocation.keys())):
                current_weight = current_allocation.get(symbol, 0)
                target_weight = target_allocation.get(symbol, 0)
                
                difference = target_weight - current_weight
                dollar_difference = difference * total_value
                
                if abs(dollar_difference) > 100:  # Only suggest if difference > $100
                    if dollar_difference > 0:
                        action = "BUY"
                    else:
                        action = "SELL"
                        dollar_difference = abs(dollar_difference)
                    
                    suggestions.append({
                        'symbol': symbol,
                        'action': action,
                        'amount': dollar_difference,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_difference': difference
                    })
            
            # Sort by largest absolute difference
            suggestions.sort(key=lambda x: abs(x['amount']), reverse=True)
            
            return suggestions
        
        except Exception as e:
            st.error(f"Error generating rebalancing suggestions: {str(e)}")
            return []
    
    def monte_carlo_simulation(self, returns_data, weights, num_simulations=1000, time_horizon=252):
        """Run Monte Carlo simulation for portfolio performance"""
        try:
            if returns_data.empty:
                return None
            
            returns = returns_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Portfolio statistics
            portfolio_mean = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Monte Carlo simulation
            simulated_returns = np.random.normal(
                portfolio_mean, 
                portfolio_std, 
                (num_simulations, time_horizon)
            )
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
            final_values = cumulative_returns[:, -1]
            
            # Calculate statistics
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
            
            return {
                'simulations': cumulative_returns,
                'final_values': final_values,
                'percentiles': {
                    '5th': percentiles[0],
                    '25th': percentiles[1],
                    'median': percentiles[2],
                    '75th': percentiles[3],
                    '95th': percentiles[4]
                },
                'mean_final_value': np.mean(final_values),
                'std_final_value': np.std(final_values)
            }
        
        except Exception as e:
            st.error(f"Error running Monte Carlo simulation: {str(e)}")
            return None
    
    def calculate_drawdown(self, returns_series):
        """Calculate maximum drawdown for a returns series"""
        try:
            cumulative = (1 + returns_series).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                'max_drawdown': max_drawdown,
                'drawdown_series': drawdown
            }
        
        except Exception as e:
            st.error(f"Error calculating drawdown: {str(e)}")
            return None
