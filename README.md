# AI-Powered Portfolio Management System

A sophisticated AI-powered portfolio management system built with Streamlit that combines real-time market data analysis, AI insights, and comprehensive portfolio analytics for retail investors.

## Features

- **AI-Powered Analysis**: Uses Anthropic's Claude AI for intelligent market analysis and investment recommendations
- **Multi-Asset Support**: Stocks, cryptocurrencies, bonds, commodities, forex, and real estate
- **Real-Time Data**: Live market data from Alpha Vantage and CoinGecko APIs
- **Future Predictions**: 1, 5, and 10-year portfolio value projections
- **Manual Portfolio Input**: Easy-to-use interface for entering your existing investments
- **Comprehensive Analysis**: Risk assessment, sector allocation, and rebalancing recommendations

## Demo

You can try the live application at: [Your Streamlit App URL]

## Quick Start

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Anthropic Claude API (required)
  - Alpha Vantage API (required - free tier available)
  - CoinGecko API (optional - for enhanced crypto data)
  - Finnhub API (optional - for stock data backup)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-portfolio-manager.git
cd ai-portfolio-manager
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` file and add your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
COINGECKO_API_KEY=your_coingecko_api_key_here  # Optional
FINNHUB_API_KEY=your_finnhub_api_key_here      # Optional
DATABASE_URL=your_postgresql_url_here          # Optional
```

5. Run the application:
```bash
streamlit run app.py
```

## API Keys Setup

### Anthropic API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an account and get your API key
3. Add it to your `.env` file

### Alpha Vantage API Key
1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Get a free API key (500 requests/day)
3. Add it to your `.env` file

### Optional API Keys

#### CoinGecko API Key
1. Go to [CoinGecko API](https://www.coingecko.com/en/api/pricing)
2. Free tier: 10,000 requests/month
3. Pro tier: Enhanced data and higher limits

#### Finnhub API Key
1. Go to [Finnhub](https://finnhub.io/register)
2. Free tier: 60 requests/minute
3. Used as backup for stock data when Alpha Vantage hits limits

#### Database Configuration (Automatic)
1. **Default**: SQLite database (automatic, no setup required)
   - Lightweight file-based database stored in `data/portfolio_manager.db`
   - Perfect for development and small-scale deployments
2. **Production**: PostgreSQL (optional upgrade)
   - Set DATABASE_URL to your PostgreSQL connection string
   - Format: `postgresql://username:password@host:port/database`
   - Providers: Neon, Supabase, Railway, etc.
   - Automatically detected when DATABASE_URL is set

## Deployment

### Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Deploy from your forked repository
5. Add your API keys in the Streamlit Cloud secrets management:
   - Go to your app settings
   - Add secrets in the format:
   ```toml
   ANTHROPIC_API_KEY = "your_key_here"
   ALPHA_VANTAGE_API_KEY = "your_key_here"
   COINGECKO_API_KEY = "your_key_here"      # Optional
   FINNHUB_API_KEY = "your_key_here"        # Optional
   DATABASE_URL = "your_postgres_url_here"  # Optional
   ```

### Local Development

The application runs on `http://localhost:8501` by default when using `streamlit run app.py`.

## Usage

1. **Market Analysis**: View real-time market data and AI-powered insights
2. **Manual Portfolio Input**: 
   - Enter your portfolio name
   - Add investments by name and amount
   - Get comprehensive AI analysis with future predictions
3. **AI Recommendations**: Receive specific investment suggestions across all asset classes

## Project Structure

```
├── app.py                 # Main Streamlit application
├── components/            # UI components
│   ├── AI_Agent.py       # Portfolio analysis interface
│   ├── Market_Analysis.py # Market data dashboard
│   └── Data_Management.py # Database management
├── utils/                 # Core utilities
│   ├── ai_analyzer.py    # AI analysis engine
│   ├── data_fetcher.py   # Market data fetching
│   ├── autonomous_agent.py # Advanced AI agent
│   └── portfolio_optimizer.py # Portfolio optimization
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md            # This file
```

## Technologies Used

- **Frontend**: Streamlit
- **AI**: Anthropic Claude API
- **Data Sources**: Alpha Vantage, CoinGecko
- **Analytics**: Pandas, NumPy, SciPy
- **Visualization**: Plotly

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:
1. Check the existing [Issues](https://github.com/yourusername/ai-portfolio-manager/issues)
2. Create a new issue with detailed information about your problem
3. Include your Python version and any error messages

## Acknowledgments

- Powered by Anthropic's Claude AI
- Market data provided by Alpha Vantage and CoinGecko
- Built with Streamlit framework