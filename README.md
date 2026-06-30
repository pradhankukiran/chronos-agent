# ChronosAgent ⏱️

**ChronosAgent** is a hybrid time-series forecasting engine and AI Agent designed to predict stock and cryptocurrency prices. It combines the statistical strength of **Meta's Prophet** with the short-term pattern-recognition power of **XGBoost**, exposing the forecasting tools via both an **interactive dashboard** (following the UK Government GDS design guidelines) and a **Model Context Protocol (MCP) server** for LLM integration.

---

## 🏗️ Architecture Overview

The application features a modular, production-grade architecture split into three layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Chatbot / LLM                        │
│                 (via FastMCP Server)                    │
└────────────────────────────┬────────────────────────────┘
                             │ (JSON RPC over stdio)
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 Flask Web Dashboard                     │
│        (UK Gov GDS Accessibility Theme UI)              │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                Hybrid Forecast Engine                   │
│  ┌───────────────────────┐   ┌───────────────────────┐  │
│  │     Prophet Model     │   │     XGBoost Model     │  │
│  │ (Long-Term Seasonal)  │   │  (Short-Term Lags/MA) │  │
│  └───────────┬───────────┘   └───────────┬───────────┘  │
│              └─────────────┬─────────────┘              │
│                            ▼                            │
│                  [Ensemble Prediction]                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                     Caching Layer                       │
│    (24-Hour File-Based Serialization for fast reads)    │
└─────────────────────────────────────────────────────────┘
```

1. **Analytical Layer (`core/data.py` & `core/models.py`):**
   * **Data Collection:** Downloads ticker histories dynamically using `yfinance`.
   * **Prophet:** Fits an additive regression model to extract yearly and weekly seasonality and the macro trend.
   * **XGBoost:** Trains on technical features (Moving Averages, Lag prices, Daily Returns) + Prophet's trend output.
   * **Autoregressive Forecast Loop:** Predicts future periods step-by-step ($T+1, T+2, \dots$) by feeding back predictions as lags for subsequent steps.
2. **Caching Layer (`core/cache.py`):**
   * Avoids running heavy training math on every single chat message.
   * Serializes fitted models to disk (`models/`) and checks file age. Reuses models if they are under 24 hours old (reducing execution time from ~3 seconds to under 0.1 seconds).
3. **UI Layer (`dashboard/app.py`):**
   * Designed strictly under the **UK Gov Web Design System (GDS)** specs.
   * Features high-contrast, accessible layouts, a classic `#0b0c0c` black GDS banner, green sharp-edged GDS buttons (`#00703c`), and left-accented blue callout boxes.
4. **Interface Layer (`mcp_server/server.py`):**
   * Implements the **Model Context Protocol (MCP)** using the Python SDK. Exposes the forecasting model as a tool that can be executed by LLMs.

---

## 📁 Repository Structure

```text
chronos-agent/
├── README.md               # You are here
├── requirements.txt        # Package dependencies
│
├── core/                   # Mathematical & Data Logic
│   ├── __init__.py
│   ├── data.py             # Data downloads & Technical indicator formulas
│   ├── models.py           # Hybrid training & autoregressive forecasting
│   └── cache.py            # Local model serialization & caching
│
├── dashboard/              # Visual Frontend
│   └── app.py              # Flask GDS Dashboard
│
├── mcp_server/             # AI Agent Integration
│   ├── __init__.py
│   └── server.py           # FastMCP tool server
│
└── models/                 # Model Directory (Auto-created; ignored in git)
    └── [TICKER]_hybrid.pkl # Saved model binary
```

---

## 🚀 Quick Start & Installation

### 1. Set Up Virtual Environment
Clone the repository, enter the directory, and set up a clean Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📈 Running the Flask Dashboard

To open the UK GDS styled interactive web interface:
```bash
python dashboard/app.py
```
This opens the app in your browser at `http://localhost:8501`. You can enter any ticker symbol, choose the forecast horizon, and view interactive charts displaying the performance of the Prophet, XGBoost, and combined Hybrid models.

---

## 🤖 Connecting to Claude Desktop (MCP)

To let Claude Desktop use your forecasting engine, open your Claude config file:
* **Linux:** `~/.config/Claude/claude_desktop_config.json`
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the server configuration pointing to your project's virtual environment:

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

*Note: Replace `/absolute/path/to/chronos-agent` with the actual path of your project directory.*

Restart Claude Desktop, and you can now ask Claude:
* *"Forecast the price of TSLA for the next 15 days."*
* *"Is Bitcoin expected to go up or down next week?"*
