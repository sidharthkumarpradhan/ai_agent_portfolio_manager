# Deployment Guide

## Files to Upload to GitHub

Upload these files and folders to your GitHub repository:

### Required Files:
- `app.py` - Main application
- `streamlit_config.py` - Streamlit configuration
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.env.example` - Environment variables template
- `.gitignore` - Git ignore file

### Required Folders:
- `components/` - UI components
  - `AI_Agent.py`
  - `Market_Analysis.py` 
  - `Data_Management.py`
- `utils/` - Core utilities
  - `ai_analyzer.py`
  - `data_fetcher.py`
  - `autonomous_agent.py`
  - `database_manager.py`
  - `portfolio_optimizer.py`
  - `risk_calculator.py`
  - `market_screener.py`
  - `fallback_data.py`

### DO NOT Upload:
- `.env` - Contains your actual API keys
- `__pycache__/` - Python cache files
- `portfolio_app.log` - Log files
- `.replit` - Replit configuration
- `pyproject.toml` - Replit package config
- `uv.lock` - Replit lock file

## Deployment Steps

### 1. Prepare for GitHub

1. Create a new repository on GitHub
2. Clone it locally or upload files directly
3. Copy the files listed above to your repository
4. Commit and push to GitHub

### 2. Deploy to Streamlit Cloud

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### 3. Configure Secrets

In Streamlit Cloud app settings, add these secrets:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key_here"
```

### 4. Get API Keys

**Anthropic API Key:**
1. Go to https://console.anthropic.com/
2. Create account and get API key
3. Add to Streamlit secrets

**Alpha Vantage API Key:**
1. Go to https://www.alphavantage.co/support/#api-key
2. Get free API key (500 requests/day)
3. Add to Streamlit secrets

## Git Commands

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit"

# Connect to GitHub
git remote add origin https://github.com/yourusername/ai-portfolio-manager.git
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Update description"
git push
```

## Environment Variables

The app will automatically detect if it's running on:
- Streamlit Cloud (uses st.secrets)
- Local development (uses .env file)
- Other platforms (uses environment variables)

## Troubleshooting

**Common Issues:**

1. **Missing API Keys Error:**
   - Ensure API keys are added to Streamlit Cloud secrets
   - Check spelling of secret names

2. **Import Errors:**
   - Verify all required files are uploaded
   - Check requirements.txt has correct versions

3. **App Won't Start:**
   - Check Streamlit Cloud logs
   - Ensure main file is set to `app.py`

4. **API Rate Limits:**
   - Alpha Vantage free tier: 500 requests/day
   - Upgrade to premium if needed

## Production Considerations

- **Database**: Optional PostgreSQL for caching (advanced feature)
- **Monitoring**: Streamlit Cloud provides basic analytics
- **Scaling**: Consider upgrading Streamlit plan for higher traffic
- **Security**: Never commit API keys to version control