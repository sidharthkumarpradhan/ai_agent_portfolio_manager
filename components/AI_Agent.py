import streamlit as st
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.autonomous_agent import AutonomousPortfolioAgent
from utils.data_fetcher import DataFetcher
from utils.database_manager import DatabaseManager
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.risk_calculator import RiskCalculator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Set up logging for this module
logger = logging.getLogger(__name__)

def init_components():
    """Initialize all required components with logging"""
    logger.info("Initializing AI Agent components...")
    
    if 'autonomous_agent' not in st.session_state:
        logger.info("Creating AutonomousPortfolioAgent")
        st.session_state.autonomous_agent = AutonomousPortfolioAgent()
    
    if 'data_fetcher' not in st.session_state:
        logger.info("Creating DataFetcher for AI Agent")
        st.session_state.data_fetcher = DataFetcher()
    
    if 'db_manager' not in st.session_state:
        logger.info("Creating DatabaseManager for AI Agent")
        st.session_state.db_manager = DatabaseManager()
    
    if 'portfolio_optimizer' not in st.session_state:
        logger.info("Creating PortfolioOptimizer")
        st.session_state.portfolio_optimizer = PortfolioOptimizer()
    
    if 'risk_calculator' not in st.session_state:
        logger.info("Creating RiskCalculator")
        st.session_state.risk_calculator = RiskCalculator()
    
    logger.info("AI Agent components initialized successfully")

def main():
    logger.info("Starting AI Agent main function")
    
    try:
        # Initialize components
        init_components()
        
        agent = st.session_state.autonomous_agent
        data_fetcher = st.session_state.data_fetcher
        db = st.session_state.db_manager
        optimizer = st.session_state.portfolio_optimizer
        risk_calc = st.session_state.risk_calculator
        
        logger.info("All components loaded successfully for AI Agent")
    except Exception as e:
        logger.error(f"Error initializing AI Agent components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")
        return
    
    st.title("🧠 Marcus Wellington - AI Portfolio Agent")
    st.markdown("### Your AI financial advisor with 50 years of Wall Street experience")
    
    # Action selection with proper state management using radio buttons
    st.subheader("🎯 What would you like Marcus to do for you?")
    
    # Initialize session state
    if 'selected_action' not in st.session_state:
        st.session_state.selected_action = "create_portfolio"
    
    # Action options mapping
    action_options = {
        "💰 Create Optimal Portfolio": "create_portfolio",
        "📊 Conduct Market Research": "market_research", 
        "🎲 Run Trading Simulation": "trading_simulation",
        "🔮 Predict Market Dynamics": "market_prediction",
        "📈 Full Portfolio Analysis": "portfolio_analysis"
    }
    
    # Find current selection index
    current_index = 0
    for i, (display_name, action_key) in enumerate(action_options.items()):
        if st.session_state.selected_action == action_key:
            current_index = i
            break
    
    # Create two-column layout for radio buttons
    col1, col2 = st.columns(2)
    
    # Split options into two columns
    option_list = list(action_options.items())
    left_options = option_list[:3]  # First 3 options
    right_options = option_list[3:]  # Last 3 options
    
    selected_action_key = None
    
    with col1:
        for display_name, action_key in left_options:
            if st.button(display_name, 
                        type="primary" if st.session_state.selected_action == action_key else "secondary",
                        use_container_width=True,
                        key=f"btn_{action_key}"):
                st.session_state.selected_action = action_key
                st.rerun()
    
    with col2:
        for display_name, action_key in right_options:
            if st.button(display_name, 
                        type="primary" if st.session_state.selected_action == action_key else "secondary",
                        use_container_width=True,
                        key=f"btn_{action_key}"):
                st.session_state.selected_action = action_key
                st.rerun()
    
    st.markdown("---")
    
    # Handle selected actions
    if 'selected_action' in st.session_state:
        action = st.session_state.selected_action
        
        if action == "create_portfolio":
            st.subheader("💰 Create Your Optimal Portfolio")
            
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                investment_amount = st.number_input(
                    "Investment Amount ($)",
                    min_value=1000.0,
                    max_value=10000000.0,
                    value=50000.0,
                    step=1000.0
                )
                
                risk_profile = st.selectbox(
                    "Risk Profile",
                    options=['conservative', 'moderate', 'aggressive'],
                    index=1
                )
            
            with col_input2:
                st.markdown("**Asset Classes to Include:**")
                include_equity = st.checkbox("📈 Equity", value=True)
                include_crypto = st.checkbox("₿ Cryptocurrency", value=True)
                include_bonds = st.checkbox("🏛️ Bonds", value=True)
                include_commodities = st.checkbox("🥇 Commodities", value=True)
                include_forex = st.checkbox("💱 Forex", value=False)
            
            # Build sectors list
            sectors = []
            if include_equity: sectors.append('equity')
            if include_crypto: sectors.append('crypto')
            if include_bonds: sectors.append('bonds')
            if include_commodities: sectors.append('commodities')
            if include_forex: sectors.append('forex')
            
            if st.button("🚀 Generate Portfolio", type="primary"):
                if sectors:
                    logger.info(f"User requesting portfolio creation: ${investment_amount}, {risk_profile}, {sectors}")
                    with st.spinner("Marcus is screening S&P 500 stocks and creating your intelligent portfolio..."):
                        try:
                            portfolio_result = agent.create_optimal_portfolio(
                                capital_amount=investment_amount,
                                risk_profile=risk_profile,
                                sectors=sectors
                            )
                            
                            # Check if we got a structured result dictionary
                            if isinstance(portfolio_result, dict) and 'portfolio_allocation' in portfolio_result:
                                logger.info("Intelligent portfolio creation successful - displaying results")
                                st.success("✅ Intelligent Portfolio Created Successfully!")
                                
                                # Enhanced stock screening display with full data
                                if 'stock_screening_results' in portfolio_result and portfolio_result['stock_screening_results']:
                                    st.subheader("🎯 Marcus Wellington's Top Stock Picks")
                                    st.caption("Selected from comprehensive S&P 500 analysis")
                                    
                                    screening_data = []
                                    for stock in portfolio_result['stock_screening_results'][:10]:
                                        # Get comprehensive stock data
                                        symbol = stock.get('symbol', 'N/A')
                                        company = stock.get('company_name', symbol)
                                        sector = stock.get('sector', 'Unknown')
                                        
                                        # Handle sector mapping for major stocks
                                        if sector == 'Unknown' or not sector:
                                            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']:
                                                sector = 'Technology'
                                            elif symbol in ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'DHR', 'BMY', 'MDT']:
                                                sector = 'Healthcare'
                                            elif symbol in ['JPM', 'BAC', 'WFC', 'GS']:
                                                sector = 'Financial'
                                        
                                        # Get investment score
                                        score = stock.get('investment_score', stock.get('composite_score', 0))
                                        
                                        # Get investment thesis
                                        thesis = stock.get('investment_thesis', 'HOLD: Mixed signals, suitable for defensive allocation.')
                                        
                                        screening_data.append({
                                            "Symbol": symbol,
                                            "Company": company[:35] if len(company) > 35 else company,
                                            "Sector": sector,
                                            "Score": f"{float(score):.1f}/10",
                                            "Investment Thesis": thesis[:90] + "..." if len(thesis) > 90 else thesis
                                        })
                                    
                                    if screening_data:
                                        screening_df = pd.DataFrame(screening_data)
                                        screening_df.index = screening_df.index + 1  # Start from 1
                                        st.dataframe(screening_df, use_container_width=True, height=400)
                                
                                # Enhanced portfolio allocation display
                                st.subheader("📊 Your Intelligent Portfolio Allocation")
                                
                                if portfolio_result['portfolio_allocation']:
                                    allocation_data = []
                                    total_investment = portfolio_result.get('total_investment', 50000)
                                    
                                    # Debug logging
                                    logger.info(f"Processing {len(portfolio_result['portfolio_allocation'])} portfolio positions")
                                    
                                    for asset_name, allocation_info in portfolio_result['portfolio_allocation'].items():
                                        logger.info(f"Processing {asset_name}: {allocation_info}")
                                        
                                        # Extract weight and calculate percentage and dollar amount
                                        weight = allocation_info.get('weight', 0)
                                        percentage = weight * 100
                                        dollar_amount = allocation_info.get('amount', weight * total_investment)
                                        
                                        # Determine asset type
                                        asset_class = allocation_info.get('asset_class', 'equity')
                                        if asset_class == 'equity' or asset_name in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY', 'VTI']:
                                            asset_type = 'Equity'
                                        elif asset_class == 'cryptocurrency' or asset_name in ['BTC', 'ETH']:
                                            asset_type = 'Cryptocurrency' 
                                        elif asset_class == 'bonds' or asset_name in ['TLT', 'BND', 'AGG']:
                                            asset_type = 'Bonds'
                                        else:
                                            asset_type = asset_class.title()
                                        
                                        # Get investment score
                                        score = allocation_info.get('composite_score', allocation_info.get('investment_score', 0))
                                        score_display = f"{float(score):.1f}/10" if score > 0 else "N/A"
                                        
                                        allocation_data.append({
                                            "Asset": asset_name,
                                            "Allocation %": f"{percentage:.1f}%",
                                            "Dollar Amount": f"${dollar_amount:,.2f}",
                                            "Type": asset_type,
                                            "Score": score_display
                                        })
                                    
                                    if allocation_data:
                                        allocation_df = pd.DataFrame(allocation_data)
                                        allocation_df.index = allocation_df.index + 1  # Start from 1
                                        st.dataframe(allocation_df, use_container_width=True, height=300)
                                        
                                        logger.info(f"Displayed allocation table with {len(allocation_data)} rows")
                                        
                                        # Enhanced pie chart with proper data extraction
                                        st.subheader("📈 Portfolio Visualization")
                                        
                                        # Extract percentage values for pie chart
                                        pie_data = []
                                        pie_labels = []
                                        for item in allocation_data:
                                            # Remove % symbol and convert to float
                                            percentage_str = item['Allocation %'].replace('%', '')
                                            percentage_val = float(percentage_str)
                                            pie_data.append(percentage_val)
                                            pie_labels.append(item['Asset'])
                                        
                                        import plotly.express as px
                                        fig = px.pie(
                                            values=pie_data,
                                            names=pie_labels,
                                            title=f"Intelligent {risk_profile.title()} Portfolio - ${investment_amount:,.0f}"
                                        )
                                        fig.update_traces(textposition='inside', textinfo='percent+label')
                                        fig.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Performance metrics with proper spacing
                                        st.markdown("---")  # Add visual separator
                                        
                                        # Calculate stock allocation percentage
                                        stock_allocation = sum(
                                            float(item['Allocation %'].replace('%', ''))
                                            for item in allocation_data 
                                            if item['Type'] == 'Equity'
                                        )
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Stock Allocation", f"{stock_allocation:.1f}%")
                                        with col2:
                                            crypto_allocation = sum(
                                                float(item['Allocation %'].replace('%', ''))
                                                for item in allocation_data 
                                                if item['Type'] == 'Cryptocurrency'
                                            )
                                            st.metric("Crypto Allocation", f"{crypto_allocation:.1f}%")
                                        with col3:
                                            bonds_allocation = sum(
                                                float(item['Allocation %'].replace('%', ''))
                                                for item in allocation_data 
                                                if item['Type'] == 'Bonds'
                                            )
                                            st.metric("Bonds Allocation", f"{bonds_allocation:.1f}%")
                                        
                                        if stock_allocation > 0:
                                            st.info(f"📈 **Portfolio Intelligence**: {stock_allocation:.1f}% allocated to hand-picked individual stocks from comprehensive S&P 500 analysis")
                                    else:
                                        st.warning("No portfolio allocation data to display")
                                else:
                                    st.warning("Portfolio allocation data is empty")
                                
                                # AI Analysis with proper spacing
                                st.markdown("---")  # Clear visual separator
                                st.subheader("🧠 Marcus Wellington's Professional Analysis")
                                if 'ai_analysis' in portfolio_result:
                                    st.write(portfolio_result['ai_analysis'])
                                else:
                                    st.write("AI analysis not available")
                                
                                # Store for later use
                                st.session_state.latest_portfolio = {
                                    'result': portfolio_result,
                                    'amount': investment_amount,
                                    'risk': risk_profile,
                                    'sectors': sectors
                                }
                                
                            elif isinstance(portfolio_result, str) and len(portfolio_result) > 50:
                                # Fallback for text-only response
                                logger.info("Received text-only portfolio result")
                                st.success("✅ Portfolio Created Successfully!")
                                st.markdown("### Marcus Wellington's Recommendation")
                                st.write(portfolio_result)
                                
                                st.session_state.latest_portfolio = {
                                    'recommendation': portfolio_result,
                                    'amount': investment_amount,
                                    'risk': risk_profile,
                                    'sectors': sectors
                                }
                            else:
                                logger.error(f"Invalid portfolio result type: {type(portfolio_result)}")
                                st.error(f"Error creating optimal portfolio: {portfolio_result}")
                                
                        except Exception as e:
                            logger.error(f"Portfolio creation error: {str(e)}")
                            st.error(f"Error creating optimal portfolio: {str(e)}")
                else:
                    logger.warning("User tried to generate portfolio without selecting asset classes")
                    st.error("Please select at least one asset class.")
        
        elif action == "market_research":
            st.subheader("📊 Comprehensive Market Research")
            
            research_sectors = st.multiselect(
                "Select sectors for research:",
                options=['equity', 'crypto', 'commodities', 'bonds', 'forex'],
                default=['equity', 'crypto', 'commodities']
            )
            
            if st.button("🔍 Conduct Research", type="primary"):
                if research_sectors:
                    logger.info(f"User requesting market research for sectors: {research_sectors}")
                    with st.spinner("Marcus is conducting comprehensive market research..."):
                        try:
                            research_results = agent.conduct_market_research(research_sectors)
                            
                            if research_results:
                                logger.info("Market research completed successfully")
                                st.success("✅ Research Complete")
                                st.markdown("### Market Research Results")
                                st.write(research_results)
                            else:
                                logger.warning("Market research returned empty results")
                                st.error("Research failed. Please check your API keys and try again.")
                        except Exception as e:
                            logger.error(f"Market research error: {str(e)}")
                            st.error(f"Research error: {str(e)}")
                else:
                    logger.warning("User tried to conduct research without selecting sectors")
                    st.error("Please select at least one sector.")
        
        elif action == "trading_simulation":
            st.subheader("🎲 AI Trading Simulation")
            
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                sim_capital = st.number_input("Simulation Capital ($)", min_value=10000.0, value=100000.0, step=5000.0)
                sim_duration = st.selectbox("Simulation Duration", ["1 week", "1 month", "3 months", "6 months"])
            
            with col_sim2:
                sim_risk = st.selectbox("Risk Tolerance", ['conservative', 'moderate', 'aggressive'], index=1)
                sim_assets = st.multiselect("Assets to Trade", 
                                          ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC', 'ETH', 'GLD', 'SPY'],
                                          default=['AAPL', 'MSFT', 'BTC'])
            
            if st.button("🚀 Run Simulation", type="primary"):
                with st.spinner("Running AI trading simulation..."):
                    # Simulation logic would go here
                    st.success("✅ Simulation Complete")
                    st.info("Trading simulation feature will be implemented with real backtesting engine.")
        
        elif action == "trading_decisions":
            st.subheader("⚡ AI Trading Decisions")
            
            if 'portfolio_data' in st.session_state and st.session_state.portfolio_data:
                col_decision1, col_decision2 = st.columns(2)
                
                with col_decision1:
                    decision_risk = st.selectbox("Risk Tolerance", ['conservative', 'moderate', 'aggressive'], index=1)
                
                with col_decision2:
                    use_drl = st.checkbox("Use Deep Learning Enhancement", value=False)
                
                if st.button("🧠 Get Trading Decisions", type="primary"):
                    with st.spinner("Marcus is analyzing your portfolio and market conditions..."):
                        # Get current portfolio data
                        current_portfolio = {}
                        for symbol, holdings in st.session_state.portfolio_data.items():
                            if symbol == 'BTC-USD':
                                btc_data = data_fetcher.get_crypto_data('BTC')
                                if btc_data is not None and not btc_data.empty:
                                    current_price = float(btc_data.iloc[-1]['close'])
                                    current_portfolio[symbol] = {
                                        'shares': holdings['shares'],
                                        'current_price': current_price,
                                        'market_value': holdings['shares'] * current_price,
                                        'avg_cost': holdings['avg_cost']
                                    }
                            else:
                                stock_data = data_fetcher.get_stock_data(symbol)
                                if stock_data is not None and not stock_data.empty:
                                    current_price = float(stock_data.iloc[-1]['close'])
                                    current_portfolio[symbol] = {
                                        'shares': holdings['shares'],
                                        'current_price': current_price,
                                        'market_value': holdings['shares'] * current_price,
                                        'avg_cost': holdings['avg_cost']
                                    }
                        
                        if use_drl:
                            logger.info("User requested DRL-enhanced trading decisions")
                            decisions = agent.make_autonomous_decisions_with_drl(
                                current_portfolio, "current_market", decision_risk
                            )
                        else:
                            logger.info("Using traditional AI analysis for trading decisions")
                            decisions = agent.make_autonomous_decisions(
                                current_portfolio, "current_market", decision_risk
                            )
                        
                        if decisions and decisions != "AI agent not available":
                            st.success("✅ Trading Decisions Ready")
                            st.markdown("### Marcus Wellington's Trading Recommendations")
                            st.write(decisions)
                        else:
                            st.error("Trading decision analysis temporarily unavailable")
            else:
                st.warning("No portfolio data found. Please create a portfolio first or add positions in the main dashboard.")
        
        elif action == "market_prediction":
            st.subheader("🔮 Market Dynamics Prediction")
            
            prediction_horizon = st.selectbox(
                "Prediction Time Horizon",
                ["1_week", "1_month", "3_months", "6_months", "1_year"],
                index=2
            )
            
            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Marcus is analyzing market patterns and generating predictions..."):
                    predictions = agent.predict_market_dynamics(prediction_horizon)
                    
                    if predictions:
                        st.success("✅ Predictions Generated")
                        st.markdown("### Market Dynamics Forecast")
                        st.write(predictions)
                    else:
                        st.error("Prediction analysis temporarily unavailable")
        
        elif action == "portfolio_analysis":
                st.markdown("### Input Your Current Holdings")
                st.info("Enter your existing investments across all asset classes for comprehensive analysis")
                
                # Portfolio name input
                portfolio_name = st.text_input("Portfolio Name", value="My Portfolio", help="Give your portfolio a descriptive name")
                
                # Initialize manual portfolio in session state
                if 'manual_portfolio' not in st.session_state:
                    st.session_state.manual_portfolio = {}
                
                    # Simplified investment input
                st.markdown("#### Add Your Investments")
                st.info("Enter the name of any investment (company name, cryptocurrency, commodity, etc.) and the dollar amount invested")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    investment_name = st.text_input(
                        "Investment Name", 
                        key="investment_name",
                        help="e.g., Amazon, Apple, Bitcoin, Gold, Treasury Bonds, Crude Oil, Microsoft"
                    )
            
                with col2:
                    investment_amount = st.number_input(
                        "Amount Invested ($)", 
                        min_value=0.0,
                        step=100.0,
                        key="investment_amount",
                        help="Enter the dollar amount"
                    )
                
                with col3:
                    if st.button("➕ Add Investment", key="add_investment"):
                        if investment_name and investment_amount > 0:
                            # Add to session state
                            st.session_state.manual_portfolio[investment_name] = investment_amount
                            st.success(f"Added {investment_name}: ${investment_amount:,.2f}")
                            # Clear inputs by rerunning
                            st.rerun()
                        else:
                            st.error("Please enter both investment name and amount")
                
                # Display current portfolio
                if st.session_state.manual_portfolio:
                    st.markdown("#### Current Portfolio Holdings")
                    
                    total_value = 0
                    for name, amount in st.session_state.manual_portfolio.items():
                        col_name, col_amount, col_remove = st.columns([3, 1, 1])
                        with col_name:
                            st.write(f"**{name}**")
                        with col_amount:
                            st.write(f"${amount:,.2f}")
                        with col_remove:
                            if st.button("🗑️", key=f"remove_{name}", help="Remove investment"):
                                del st.session_state.manual_portfolio[name]
                                st.rerun()
                        total_value += amount
                    
                    st.markdown("---")
                    st.markdown(f"### **Total Portfolio Value: ${total_value:,.2f}**")
                
                # Analysis button for manual portfolio
                if ('manual_holdings' in st.session_state.manual_portfolio and 
                    st.session_state.manual_portfolio['manual_holdings'] and 
                    st.button("🔍 Analyze My Complete Portfolio", type="primary")):
                    
                    with st.spinner("Marcus Wellington is conducting comprehensive portfolio analysis..."):
                        try:
                            # Prepare holdings for analysis
                            holdings = st.session_state.manual_portfolio['manual_holdings']
                            
                            # Create analysis prompt with all holdings
                            holdings_summary = []
                            total_value = 0
                            for name, data in holdings.items():
                                holdings_summary.append(f"- {data['name']}: ${data['amount']:,.2f}")
                                total_value += data['amount']
                            
                            # Create structured analysis prompt for better formatting
                            analysis_prompt = f"""
                            As Marcus Wellington, analyze this ${total_value:,.2f} portfolio:

                            HOLDINGS:
                            {chr(10).join(holdings_summary)}

                            Provide analysis in this exact structure:

                            MARKET OUTLOOK:
                            Brief assessment of current market conditions and key factors affecting this portfolio.

                            INDIVIDUAL ASSETS:
                            For each holding, provide: Current assessment, outlook, and specific recommendation.

                            FUTURE VALUE PREDICTIONS:
                            1-Year Target: $ amount (assumptions)
                            5-Year Target: $ amount (assumptions) 
                            10-Year Target: $ amount (assumptions)
                            Include best/likely/worst case for each timeframe.

                            INVESTMENT RECOMMENDATIONS:
                            
                            STOCKS TO BUY:
                            1. [Ticker] - Company Name - $ allocation - Reason
                            2. [Ticker] - Company Name - $ allocation - Reason
                            3. [Ticker] - Company Name - $ allocation - Reason

                            BONDS:
                            - Government allocation: $ amount
                            - Corporate allocation: $ amount
                            - TIPS allocation: $ amount

                            COMMODITIES:
                            - Gold ETF (GLD): $ amount
                            - Energy (USO): $ amount
                            - Agriculture: $ amount

                            FOREX:
                            - Currency recommendations with specific pairs

                            REAL ESTATE:
                            - REIT recommendations with tickers

                            IPO OPPORTUNITIES:
                            - 2-3 upcoming IPOs worth considering

                            REBALANCING PLAN:
                            - What to sell: specific amounts
                            - What to buy: specific amounts
                            - Target percentages for each asset class

                            Keep responses concise and actionable. Provide specific dollar amounts and percentages.
                            """
                            
                            # Use AI analyzer for comprehensive analysis
                            ai_analyzer = st.session_state.get('ai_analyzer')
                            if not ai_analyzer:
                                from utils.ai_analyzer import AIAnalyzer
                                ai_analyzer = AIAnalyzer()
                            
                            # Get comprehensive analysis using market analysis method
                            logger.info(f"Sending analysis request for {len(holdings)} holdings: {list(holdings.keys())}")
                            analysis = ai_analyzer.get_market_analysis(analysis_prompt)
                            
                            if analysis and analysis != "AI analysis not available":
                                st.success("✅ Comprehensive Analysis Complete")
                                
                                # Display portfolio summary first
                                st.markdown("#### Portfolio Summary")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Value", f"${total_value:,.2f}")
                                with col2:
                                    st.metric("Number of Holdings", len(holdings))
                                
                                # Show holdings breakdown
                                st.markdown("#### Holdings Breakdown")
                                for name, data in holdings.items():
                                    percentage = (data['amount'] / total_value) * 100
                                    st.write(f"• **{data['name']}**: ${data['amount']:,.2f} ({percentage:.1f}%)")
                                
                                st.markdown(f"### Marcus Wellington's Complete Portfolio Analysis")
                                
                                # Parse and display analysis sections properly
                                analysis_sections = parse_analysis_sections(analysis)
                                
                                # Display Market Outlook & Individual Assets
                                with st.expander("📊 Market Outlook & Individual Assets", expanded=True):
                                    if "market_outlook" in analysis_sections:
                                        st.markdown("**MARKET OUTLOOK:**")
                                        st.write(analysis_sections["market_outlook"])
                                        st.markdown("---")
                                    
                                    if "individual_assets" in analysis_sections:
                                        st.markdown("**INDIVIDUAL ASSETS:**")
                                        st.write(analysis_sections["individual_assets"])
                                    else:
                                        # Show first part of analysis if sections not properly parsed
                                        analysis_parts = analysis.split('\n\n')
                                        for i, part in enumerate(analysis_parts[:3]):
                                            if part.strip():
                                                st.write(part.strip())
                                                if i < 2:
                                                    st.markdown("---")
                                
                                # Display Future Value Predictions
                                with st.expander("🔮 Future Value Predictions", expanded=True):
                                    if "future_predictions" in analysis_sections:
                                        st.write(analysis_sections["future_predictions"])
                                    else:
                                        # Look for prediction keywords in the full text
                                        prediction_keywords = ["FUTURE VALUE", "1-Year", "5-Year", "10-Year", "Target:", "Outlook"]
                                        prediction_found = False
                                        
                                        for keyword in prediction_keywords:
                                            start_idx = analysis.upper().find(keyword.upper())
                                            if start_idx != -1:
                                                # Find the end of this section (next major section or end)
                                                next_section_keywords = ["INVESTMENT RECOMMENDATIONS", "REBALANCING", "BONDS:", "STOCKS TO BUY"]
                                                end_idx = len(analysis)
                                                for next_keyword in next_section_keywords:
                                                    next_idx = analysis.upper().find(next_keyword.upper(), start_idx + len(keyword))
                                                    if next_idx != -1:
                                                        end_idx = min(end_idx, next_idx)
                                                
                                                prediction_text = analysis[start_idx:end_idx].strip()
                                                if len(prediction_text) > 50:  # Valid section found
                                                    st.write(prediction_text)
                                                    prediction_found = True
                                                    break
                                        
                                        if not prediction_found:
                                            st.warning("Future value predictions not found. Please request specific 1, 5, and 10-year projections.")
                                
                                # Display Investment Recommendations
                                with st.expander("💡 Investment Recommendations", expanded=True):
                                    if "investment_recommendations" in analysis_sections:
                                        st.write(analysis_sections["investment_recommendations"])
                                    else:
                                        # Look for recommendation keywords
                                        rec_keywords = ["INVESTMENT RECOMMENDATIONS", "STOCKS TO BUY", "BONDS:", "COMMODITIES:", "FOREX:", "REAL ESTATE:", "IPO"]
                                        rec_found = False
                                        
                                        for keyword in rec_keywords:
                                            start_idx = analysis.upper().find(keyword.upper())
                                            if start_idx != -1:
                                                # Get remaining text from this point
                                                rec_text = analysis[start_idx:].strip()
                                                if len(rec_text) > 50:
                                                    st.write(rec_text)
                                                    rec_found = True
                                                    break
                                        
                                        if not rec_found:
                                            st.warning("Investment recommendations not found. Please request specific stock, bond, and commodity suggestions.")
                                
                                # Show full analysis in a separate expander
                                with st.expander("📄 Complete Analysis (Full Text)", expanded=False):
                                    st.text_area("Full Analysis", analysis, height=400)
                                
                                # Store analysis in session
                                st.session_state.analyzed_manual_portfolio = {
                                    'name': portfolio_name,
                                    'holdings': holdings,
                                    'total_value': total_value,
                                    'analysis': analysis,
                                    'analysis_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                                }
                            else:
                                st.error("Analysis temporarily unavailable. Please check API connectivity.")
                            
                        except Exception as e:
                            logger.error(f"Manual portfolio analysis failed: {str(e)}")
                            st.error(f"Analysis failed: {str(e)}")
                
                # Show previous analysis if available
                if 'analyzed_manual_portfolio' in st.session_state:
                    prev_analysis = st.session_state.analyzed_manual_portfolio
                    st.markdown("#### Previous Analysis")
                    st.info(f"Analysis for '{prev_analysis['name']}' - {prev_analysis.get('analysis_date', 'Unknown date')}")
                    
                    if st.button("📋 Show Previous Analysis"):
                        st.markdown("### Previous Portfolio Analysis")
                        st.write(prev_analysis['analysis'])
    
def parse_analysis_sections(analysis):
    """Parse analysis text into structured sections"""
    sections = {}
    
    # Define section markers and their keys
    section_markers = {
        "MARKET OUTLOOK:": "market_outlook",
        "INDIVIDUAL ASSETS:": "individual_assets", 
        "FUTURE VALUE PREDICTIONS:": "future_predictions",
        "INVESTMENT RECOMMENDATIONS:": "investment_recommendations"
    }
    
    # Find all section positions
    section_positions = []
    for marker, key in section_markers.items():
        pos = analysis.upper().find(marker.upper())
        if pos != -1:
            section_positions.append((pos, marker, key))
    
    # Sort by position
    section_positions.sort()
    
    # Extract content for each section
    for i, (pos, marker, key) in enumerate(section_positions):
        start = pos + len(marker)
        
        # Find end position (start of next section or end of text)
        if i < len(section_positions) - 1:
            end = section_positions[i + 1][0]
        else:
            end = len(analysis)
        
        content = analysis[start:end].strip()
        if content:
            sections[key] = content
    
    return sections
    


    
    # Database status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Database Status")
    
    try:
        db_manager = DatabaseManager()
        if db_manager.is_available():
            asset_count = len(db_manager.get_asset_universe())
            st.sidebar.metric("Assets Tracked", asset_count)
            st.sidebar.success("Database Connected")
        else:
            st.sidebar.error("Database Offline")
    except Exception as e:
        st.sidebar.error("Database Error")

if __name__ == "__main__":
    main()