# chronos-agent

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-orange.svg?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-red.svg?style=flat-square)](https://facebook.github.io/prophet/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-lightgrey.svg?style=flat-square&logo=xgboost)](https://xgboost.readthedocs.io/)
[![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-FastMCP-blueviolet.svg?style=flat-square)](https://modelcontextprotocol.io/)

chronos-agent is a multi-sector, general-purpose time-series forecasting engine and AI Agent designed to model and project trends across financial, climate, and macroeconomic domains. It combines the statistical strength of Meta's Prophet with the short-term autoregressive pattern-recognition power of XGBoost. It exposes its modeling tools via an interactive, GDS-compliant web portal and a Model Context Protocol (MCP) server for LLM integration.

---

## Architecture Overview

The system features a modular architecture split into four distinct layers:

1. **Analytical Layer (`core/data.py` and `core/models.py`):**
   * **Multi-Sector Data Integration:** Fetches financial tickers via yfinance, coordinates-based daily temperatures via Open-Meteo, and macroeconomic indicators via pandas-datareader (FRED).
   * **Prophet Baseline:** Fits an additive regression model to extract yearly and weekly seasonality cycles and long-term trend lines.
   * **XGBoost Residual Corrector:** Trains on engineered lag features, rolling averages, and rolling returns of residuals to correct short-term drift.
   * **Autoregressive Forecast Loop:** Evaluates predictions step-by-step by feeding projections back into the feature matrices.
   * **Statistical Diagnostics:** Computes backtest diagnostics (coefficient of determination R2, Mean Absolute Error MAE), Durbin-Watson autocorrelation statistics, and extracts seasonality curves and changepoint shifts.

2. **Caching Layer (`core/cache.py`):**
   * Implements a local file-based serialization system for model binary states.
   * Prevents repeating heavy model fits for requests made within 24 hours, dropping analytical retrieval latency from ~3 seconds to under 0.1 seconds.

3. **UI Layer (`dashboard/app.py`):**
   * Follows the UK Government Web Design System (GDS) guidelines for visual styling and accessibility.
   * Employs flat card components, inset warnings, summary tables, and full-width grid layouts (expanded to 1200px for dense visualization).
   * Visualizes forecasts using Plotly with interactive range selectors, range sliders, uncertainty interval shading, historical average/max/min benchmark lines, forecast-era overlays, and custom dynamic tooltips.
   * Features a full-screen loading overlay with a GDS spinner and cycling status messages describing the underlying machine learning pipeline.

4. **Interface Layer (`mcp_server/server.py`):**
   * Implements the Model Context Protocol (MCP) to expose the forecasting engine as an executable tool for AI assistants like Claude Desktop.

---

## Repository Structure

```text
chronos-agent/
├── README.md               # Project documentation
├── requirements.txt        # Package dependencies
│
├── core/                   # Analytical & Caching Core
│   ├── __init__.py
│   ├── data.py             # Data fetching pipelines
│   ├── models.py           # Hybrid training and forecasting loop
│   └── cache.py            # Model serialization logic
│
├── dashboard/              # Flask Frontend
│   ├── app.py              # Application controller & routes
│   └── templates/          # GDS HTML Layout templates
│       ├── base.html       # Base template with loading overlay
│       └── index.html      # Dynamic dashboard panels
│
├── mcp_server/             # AI Agent Interface
│   ├── __init__.py
│   └── server.py           # FastMCP tool server
│
└── models/                 # Model cache directory (auto-created)
```

---

## Quick Start & Installation

### 1. Set Up Virtual Environment
Clone the repository and set up a clean Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Web Portal

To boot the GDS-styled dashboard:
```bash
python dashboard/app.py
```
Open your browser to `http://localhost:8501`. 

### Key Features:
* **Finance Tab:** Query any ticker (e.g. AAPL, BTC-USD) with live Yahoo Finance geocoding.
* **Weather Tab:** Enter any global city or region (e.g. London, Bhubaneswar) using Open-Meteo geocoding search to forecast daily max temperature trends.
* **Economics Tab:** Search and select from 33 of the most popular FRED macroeconomic indicators (e.g. GDP, Unemployment Rate, CPI Inflation, WTI Oil Price, 10-Year Bond Yield).
* **Interactive Exploration:** Hover over Plotly charts to view metrics, zoom in/out with the range slider, and view analytical stats panels (backtest results, seasonality peaks, Durbin-Watson statistics).

---

## Connecting to Claude Desktop (MCP)

To configure the Model Context Protocol (MCP) server for Claude Desktop, open your configuration file:
* **Linux:** `~/.config/Claude/claude_desktop_config.json`
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the following config (ensuring paths match your absolute system location):

```json
{
  "mcpServers": {
    "chronos-agent": {
      "command": "/absolute/path/to/chronos-agent/venv/bin/python",
      "args": [
        "/absolute/path/to/chronos-agent/mcp_server/server.py"
      ]
    }
  }
}
```

Restart Claude Desktop, and you can programmatically command Claude to run time-series forecasts:
* *Forecast the price of TSLA for the next 30 days.*
* *What is the forecasted temperature trend for Paris next week?*
* *Run a hybrid forecast on the US Unemployment rate (UNRATE) for the next 12 periods.*
```,Description:
