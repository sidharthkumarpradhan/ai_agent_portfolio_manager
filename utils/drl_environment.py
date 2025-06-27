import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from datetime import datetime, timedelta
import streamlit as st
from .data_fetcher import DataFetcher
from .autonomous_agent import AutonomousPortfolioAgent

class TradingEnvironment(gym.Env):
    """
    Deep Reinforcement Learning Trading Environment
    State: Market data, portfolio status, technical indicators
    Actions: Buy, Sell, Hold for multiple assets
    Reward: Portfolio returns with risk adjustment
    """
    
    def __init__(self, 
                 symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD'],
                 initial_capital=100000,
                 lookback_window=60,
                 transaction_cost=0.001):
        
        super(TradingEnvironment, self).__init__()
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher()
        
        # Action space: [position_change] for each asset
        # -1 = sell all, 0 = hold, +1 = buy with available cash
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(len(symbols),), 
            dtype=np.float32
        )
        
        # Observation space: [price_features, technical_indicators, portfolio_state]
        # Price features: OHLCV for each asset (5 * n_assets)
        # Technical indicators: RSI, MACD, Bollinger Bands (3 * n_assets)
        # Portfolio state: current_positions, cash_ratio, total_return
        obs_dim = len(symbols) * 8 + 3  # 8 features per asset + 3 portfolio features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.position_values = {symbol: 0 for symbol in self.symbols}
        
        # Load historical data
        self._load_market_data()
        
        # Calculate initial observation
        observation = self._get_observation()
        
        return observation
    
    def _load_market_data(self):
        """Load historical market data for training"""
        self.market_data = {}
        
        for symbol in self.symbols:
            try:
                if symbol == 'BTC-USD':
                    data = self.data_fetcher.get_crypto_data('BTC')
                else:
                    data = self.data_fetcher.get_stock_data(symbol)
                
                if data is not None and not data.empty:
                    # Ensure we have enough data
                    if len(data) > self.lookback_window + 100:
                        self.market_data[symbol] = data
                    else:
                        st.warning(f"Insufficient data for {symbol}")
                else:
                    st.warning(f"No data available for {symbol}")
            
            except Exception as e:
                st.error(f"Error loading data for {symbol}: {str(e)}")
        
        if not self.market_data:
            st.error("No market data loaded for DRL environment")
            return
        
        # Find common date range
        all_indices = [data.index for data in self.market_data.values()]
        self.common_dates = all_indices[0]
        for idx in all_indices[1:]:
            self.common_dates = self.common_dates.intersection(idx)
        
        self.common_dates = self.common_dates.sort_values()
        
        if len(self.common_dates) < self.lookback_window + 50:
            st.error("Insufficient overlapping data for DRL training")
            return
        
        self.max_steps = len(self.common_dates) - self.lookback_window - 1
    
    def _get_observation(self):
        """Get current environment observation"""
        if not hasattr(self, 'common_dates') or len(self.common_dates) == 0:
            return np.zeros(self.observation_space.shape[0])
        
        current_date = self.common_dates[min(self.current_step + self.lookback_window, len(self.common_dates) - 1)]
        
        observation = []
        
        # Market data features for each symbol
        for symbol in self.symbols:
            if symbol in self.market_data:
                data = self.market_data[symbol]
                
                # Get current price data
                try:
                    current_row = data.loc[current_date]
                    
                    # OHLCV features (normalized)
                    close_price = float(current_row['close'])
                    open_price = float(current_row['open'])
                    high_price = float(current_row['high'])
                    low_price = float(current_row['low'])
                    volume = float(current_row['volume'])
                    
                    # Normalize prices relative to close
                    obs_features = [
                        (open_price - close_price) / close_price,
                        (high_price - close_price) / close_price,
                        (low_price - close_price) / close_price,
                        np.log(volume + 1) / 20,  # Log-normalized volume
                    ]
                    
                    # Technical indicators
                    tech_indicators = self._calculate_technical_indicators(symbol, current_date)
                    obs_features.extend(tech_indicators)
                    
                except:
                    # If data not available, use zeros
                    obs_features = [0.0] * 8
                
            else:
                obs_features = [0.0] * 8
            
            observation.extend(obs_features)
        
        # Portfolio state features
        total_invested = sum(self.position_values.values())
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        portfolio_features = [cash_ratio, total_return, total_invested / self.portfolio_value if self.portfolio_value > 0 else 0]
        observation.extend(portfolio_features)
        
        return np.array(observation, dtype=np.float32)
    
    def _calculate_technical_indicators(self, symbol, current_date):
        """Calculate technical indicators for a symbol"""
        if symbol not in self.market_data:
            return [0.0, 0.0, 0.0, 0.0]
        
        data = self.market_data[symbol]
        
        try:
            # Get historical data up to current date
            hist_data = data[data.index <= current_date].tail(30)
            
            if len(hist_data) < 14:
                return [0.0, 0.0, 0.0, 0.0]
            
            closes = hist_data['close'].values
            
            # RSI (Relative Strength Index)
            rsi = self._calculate_rsi(closes)
            
            # Simple moving averages
            sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            
            current_price = closes[-1]
            
            # Moving average convergence
            ma_signal = (sma_10 - sma_20) / current_price if current_price > 0 else 0
            
            # Price momentum (20-day)
            momentum = (current_price - closes[0]) / closes[0] if len(closes) >= 20 and closes[0] > 0 else 0
            
            return [rsi / 100, ma_signal, momentum, 0.0]  # Normalized values
        
        except:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def step(self, action):
        """Execute one step in the environment"""
        if not hasattr(self, 'common_dates') or len(self.common_dates) == 0:
            return np.zeros(self.observation_space.shape[0]), 0, True, {}
        
        # Execute trades based on action
        reward = self._execute_trades(action)
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate portfolio metrics for info
        info = {
            'portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'positions': self.positions.copy(),
            'cash': self.cash
        }
        
        return observation, reward, done, info
    
    def _execute_trades(self, action):
        """Execute trades based on DRL agent action"""
        if not hasattr(self, 'common_dates') or len(self.common_dates) == 0:
            return 0
        
        current_date = self.common_dates[min(self.current_step + self.lookback_window, len(self.common_dates) - 1)]
        
        # Update current portfolio value
        total_portfolio_value = self.cash
        
        # Calculate current position values
        for i, symbol in enumerate(self.symbols):
            if symbol in self.market_data and self.positions[symbol] > 0:
                try:
                    current_price = float(self.market_data[symbol].loc[current_date, 'close'])
                    position_value = self.positions[symbol] * current_price
                    self.position_values[symbol] = position_value
                    total_portfolio_value += position_value
                except:
                    self.position_values[symbol] = 0
        
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value = total_portfolio_value
        
        # Execute trades based on action
        for i, symbol in enumerate(self.symbols):
            if symbol in self.market_data:
                try:
                    current_price = float(self.market_data[symbol].loc[current_date, 'close'])
                    action_value = action[i]
                    
                    # Determine trade action
                    if action_value > 0.1:  # Buy signal
                        # Use portion of available cash
                        cash_to_use = self.cash * min(action_value, 0.5)  # Max 50% of cash per trade
                        if cash_to_use > 100:  # Minimum trade size
                            shares_to_buy = cash_to_use / current_price
                            transaction_cost = cash_to_use * self.transaction_cost
                            
                            self.positions[symbol] += shares_to_buy
                            self.cash -= (cash_to_use + transaction_cost)
                    
                    elif action_value < -0.1 and self.positions[symbol] > 0:  # Sell signal
                        # Sell portion of position
                        shares_to_sell = self.positions[symbol] * min(abs(action_value), 1.0)
                        if shares_to_sell > 0:
                            sale_value = shares_to_sell * current_price
                            transaction_cost = sale_value * self.transaction_cost
                            
                            self.positions[symbol] -= shares_to_sell
                            self.cash += (sale_value - transaction_cost)
                
                except Exception as e:
                    pass  # Skip problematic trades
        
        # Calculate reward based on portfolio performance
        if previous_portfolio_value > 0:
            portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
            
            # Risk-adjusted reward (Sharpe-like)
            # Penalize large drawdowns and reward consistent returns
            reward = portfolio_return
            
            # Add risk penalty for high concentration
            concentration_penalty = self._calculate_concentration_penalty()
            reward -= concentration_penalty
            
            return reward
        
        return 0
    
    def _calculate_concentration_penalty(self):
        """Calculate penalty for portfolio concentration"""
        if self.portfolio_value <= 0:
            return 0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        weights = []
        for symbol in self.symbols:
            weight = self.position_values.get(symbol, 0) / self.portfolio_value
            weights.append(weight)
        
        # Add cash weight
        cash_weight = self.cash / self.portfolio_value
        weights.append(cash_weight)
        
        hhi = sum(w**2 for w in weights)
        
        # Penalty increases with concentration (HHI > 0.25 is considered concentrated)
        concentration_penalty = max(0, (hhi - 0.25) * 0.1)
        
        return concentration_penalty

class DRLAgent:
    """Deep Reinforcement Learning Agent using Actor-Critic (PPO-style)"""
    
    def __init__(self, state_size, action_size, learning_rate=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training history
        self.training_history = []
    
    def _build_actor(self):
        """Build actor network (policy)"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='tanh')  # Actions between -1 and 1
        ])
        
        model.compile(optimizer=self.actor_optimizer, loss='mse')
        return model
    
    def _build_critic(self):
        """Build critic network (value function)"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # Value estimation
        ])
        
        model.compile(optimizer=self.critic_optimizer, loss='mse')
        return model
    
    def get_action(self, state, training=True):
        """Get action from current policy"""
        state = tf.expand_dims(state, axis=0)
        action = self.actor(state)[0]
        
        if training:
            # Add exploration noise
            noise = tf.random.normal(shape=action.shape, stddev=0.1)
            action = tf.clip_by_value(action + noise, -1.0, 1.0)
        
        return action.numpy()
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one training step"""
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones)
        
        # Calculate advantages
        current_values = self.critic(states)
        next_values = self.critic(next_states)
        
        # Temporal difference targets
        td_targets = rewards + 0.99 * next_values * (1 - dones)
        advantages = td_targets - current_values
        
        # Train critic
        with tf.GradientTape() as tape:
            values = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(td_targets - values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Train actor
        with tf.GradientTape() as tape:
            predicted_actions = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(advantages * predicted_actions)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return float(actor_loss), float(critic_loss)
    
    def save_models(self, filepath_prefix):
        """Save trained models"""
        try:
            self.actor.save(f"{filepath_prefix}_actor.h5")
            self.critic.save(f"{filepath_prefix}_critic.h5")
            return True
        except Exception as e:
            st.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, filepath_prefix):
        """Load trained models"""
        try:
            self.actor = keras.models.load_model(f"{filepath_prefix}_actor.h5")
            self.critic = keras.models.load_model(f"{filepath_prefix}_critic.h5")
            return True
        except Exception as e:
            st.warning(f"Could not load models: {str(e)}")
            return False

class DRLTrainer:
    """Training manager for DRL trading agent"""
    
    def __init__(self, symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD']):
        self.symbols = symbols
        self.env = TradingEnvironment(symbols=symbols)
        
        # Initialize agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        self.agent = DRLAgent(state_size, action_size)
        
        # Training parameters
        self.episodes_trained = 0
        self.best_performance = -np.inf
        
    def train_agent(self, episodes=100, batch_size=32):
        """Train the DRL agent"""
        training_results = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Storage for batch training
            states_batch = []
            actions_batch = []
            rewards_batch = []
            next_states_batch = []
            dones_batch = []
            
            done = False
            while not done and episode_steps < 1000:  # Max steps per episode
                # Get action from agent
                action = self.agent.get_action(state, training=True)
                
                # Execute action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                states_batch.append(state)
                actions_batch.append(action)
                rewards_batch.append(reward)
                next_states_batch.append(next_state)
                dones_batch.append(done)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Train when batch is full
                if len(states_batch) >= batch_size:
                    actor_loss, critic_loss = self.agent.train_step(
                        states_batch, actions_batch, rewards_batch,
                        next_states_batch, dones_batch
                    )
                    
                    # Clear batch
                    states_batch = []
                    actions_batch = []
                    rewards_batch = []
                    next_states_batch = []
                    dones_batch = []
            
            # Train on remaining experiences
            if len(states_batch) > 0:
                actor_loss, critic_loss = self.agent.train_step(
                    states_batch, actions_batch, rewards_batch,
                    next_states_batch, dones_batch
                )
            
            # Record training results
            final_portfolio_value = self.env.portfolio_value
            total_return = (final_portfolio_value - self.env.initial_capital) / self.env.initial_capital
            
            training_results.append({
                'episode': episode + 1,
                'total_return': total_return,
                'final_value': final_portfolio_value,
                'episode_reward': episode_reward,
                'steps': episode_steps
            })
            
            # Save best model
            if total_return > self.best_performance:
                self.best_performance = total_return
                self.agent.save_models("models/best_drl_agent")
            
            self.episodes_trained += 1
        
        return training_results
    
    def get_trading_recommendation(self, current_market_state):
        """Get trading recommendation from trained agent"""
        if hasattr(self.agent, 'actor'):
            try:
                action = self.agent.get_action(current_market_state, training=False)
                
                recommendations = []
                for i, symbol in enumerate(self.symbols):
                    action_value = action[i]
                    
                    if action_value > 0.2:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'strength': float(action_value),
                            'confidence': min(float(action_value) * 100, 100)
                        })
                    elif action_value < -0.2:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'strength': float(abs(action_value)),
                            'confidence': min(float(abs(action_value)) * 100, 100)
                        })
                    else:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'HOLD',
                            'strength': 0,
                            'confidence': 50
                        })
                
                return recommendations
            
            except Exception as e:
                st.error(f"Error getting DRL recommendation: {str(e)}")
                return []
        
        return []