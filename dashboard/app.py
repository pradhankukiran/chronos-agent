import os
import sys
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    make_response,
    send_from_directory,
    jsonify,
)

# Make the project root importable so `core` resolves when launched from
# the dashboard directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.models import get_forecast, get_generic_forecast
from core.data import fetch_stock_data, fetch_company_info, fetch_weather_data, fetch_economic_data
from core.cache import load_model

import pandas as pd                           # noqa: E402
import plotly.graph_objects as go            # noqa: E402
import requests                               # noqa: E402


POPULAR_FRED_SERIES = [
    {"id": "GDP", "name": "Gross Domestic Product (GDP)", "freq": "Quarterly"},
    {"id": "GDPC1", "name": "Real Gross Domestic Product (Real GDP)", "freq": "Quarterly"},
    {"id": "UNRATE", "name": "U.S. Unemployment Rate (%)", "freq": "Monthly"},
    {"id": "CPIAUCSL", "name": "U.S. Consumer Price Index / Inflation", "freq": "Monthly"},
    {"id": "FEDFUNDS", "name": "Effective Federal Funds Interest Rate (%)", "freq": "Monthly"},
    {"id": "PAYEMS", "name": "Total Nonfarm Payroll Employees", "freq": "Monthly"},
    {"id": "T10Y2Y", "name": "Treasury Yield Spread (10-Year minus 2-Year)", "freq": "Daily"},
    {"id": "DGS10", "name": "10-Year Treasury Bond Yield (%)", "freq": "Daily"},
    {"id": "DGS2", "name": "2-Year Treasury Bond Yield (%)", "freq": "Daily"},
    {"id": "DGS30", "name": "30-Year Treasury Bond Yield (%)", "freq": "Daily"},
    {"id": "M2SL", "name": "M2 Money Supply (Billions)", "freq": "Monthly"},
    {"id": "INDPRO", "name": "Industrial Production Index", "freq": "Monthly"},
    {"id": "HOUST", "name": "U.S. Housing Starts (New Construction)", "freq": "Monthly"},
    {"id": "UMCSENT", "name": "University of Michigan Consumer Sentiment Index", "freq": "Monthly"},
    {"id": "RSXFS", "name": "U.S. Advance Retail Sales", "freq": "Monthly"},
    {"id": "WALCL", "name": "Federal Reserve System Total Assets", "freq": "Weekly"},
    {"id": "DGORDER", "name": "Manufacturers' New Orders: Durable Goods", "freq": "Monthly"},
    {"id": "PPIACO", "name": "Producer Price Index (All Commodities)", "freq": "Monthly"},
    {"id": "ICSA", "name": "Weekly Initial Jobless Claims", "freq": "Weekly"},
    {"id": "SP500", "name": "S&P 500 Stock Market Index Price", "freq": "Daily"},
    {"id": "NASDAQ100", "name": "NASDAQ 100 Stock Index Price", "freq": "Daily"},
    {"id": "BAMLH0A0HYM2", "name": "US High Yield Corporate Bond Option-Adjusted Spread (%)", "freq": "Daily"},
    {"id": "DTWEXBGS", "name": "Trade-Weighted U.S. Dollar Index", "freq": "Weekly"},
    {"id": "CP", "name": "Corporate Profits After Tax", "freq": "Quarterly"},
    {"id": "BUSINV", "name": "Total Business Inventories", "freq": "Monthly"},
    {"id": "PCE", "name": "Personal Consumption Expenditures", "freq": "Monthly"},
    {"id": "PSAVERT", "name": "Personal Saving Rate (%)", "freq": "Monthly"},
    {"id": "FYFSD", "name": "U.S. Federal Surplus or Deficit", "freq": "Yearly"},
    {"id": "GFDEGDQ188S", "name": "Total Federal Public Debt as % of GDP", "freq": "Quarterly"},
    {"id": "CSUSHPINSA", "name": "S&P/Case-Shiller U.S. National Home Price Index", "freq": "Monthly"},
    {"id": "DCOILWTICO", "name": "WTI Crude Oil Price (USD/Barrel)", "freq": "Daily"},
    {"id": "GASREGCOW", "name": "U.S. Regular Gasoline Retail Price (USD/Gallon)", "freq": "Weekly"},
    {"id": "MORTGAGE30US", "name": "30-Year Fixed Rate Mortgage Average (%)", "freq": "Weekly"}
]

def create_app() -> Flask:
    app = Flask(__name__)

    # Ensure static resources (like govuk-frontend) are correctly resolvable.
    # The GDS frontend was compiled into the static directory.
    @app.route("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(
            os.path.join(app.root_path, "static", "assets"), filename
        )

    # Route: /
    # Redirects to /finance by default
    @app.route("/", methods=["GET"])
    def index():
        return redirect(url_for("finance"))

    # Route: /finance
    @app.route("/finance", methods=["GET"])
    def finance():
        ticker = (request.args.get("ticker") or "").upper().strip()
        horizon_raw = request.args.get("horizon")

        has_results = False
        error_msg = None
        context = {
            "active_tab": "finance",
            "has_results": has_results,
            "error_msg": error_msg,
            "ticker": ticker,
            "horizon": horizon_raw or "30",
        }

        if ticker:
            try:
                horizon = _parse_horizon(horizon_raw)
                forecast_data = _build_finance_forecast(ticker, horizon)
                context.update(forecast_data)
                context["has_results"] = True
            except Exception as e:
                context["error_msg"] = str(e)

        return render_template("index.html", **context)

    # Route: /weather
    @app.route("/weather", methods=["GET"])
    def weather():
        lat_raw = request.args.get("lat") or ""
        lon_raw = request.args.get("lon") or ""
        city_name = request.args.get("city_name") or ""
        horizon_raw = request.args.get("horizon")

        has_results = False
        error_msg = None
        context = {
            "active_tab": "weather",
            "has_results": has_results,
            "error_msg": error_msg,
            "lat": lat_raw,
            "lon": lon_raw,
            "city_name": city_name,
            "horizon": horizon_raw or "30",
        }

        if lat_raw and lon_raw:
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
                horizon = _parse_horizon(horizon_raw)
                forecast_data = _build_weather_forecast(lat, lon, city_name, horizon)
                context.update(forecast_data)
                context["has_results"] = True
            except Exception as e:
                context["error_msg"] = str(e)

        return render_template("index.html", **context)

    # Route: /economics
    @app.route("/economics", methods=["GET"])
    def economics():
        series_raw = request.args.get("series_id") or ""
        series_id = series_raw.upper().strip()
        series_name = request.args.get("series_name") or ""
        horizon_raw = request.args.get("horizon")

        has_results = False
        error_msg = None
        context = {
            "active_tab": "economics",
            "has_results": has_results,
            "error_msg": error_msg,
            "series_id": series_id,
            "series_name": series_name,
            "horizon": horizon_raw or "30",
        }

        if series_raw:
            try:
                horizon = _parse_horizon(horizon_raw)
                forecast_data = _build_economics_forecast(series_id, horizon)
                context.update(forecast_data)
                context["has_results"] = True
            except Exception as e:
                context["error_msg"] = str(e)

        return render_template("index.html", **context)

    # Auto-complete API route for Finance tab (search ticker)
    @app.route("/api/search", methods=["GET"])
    def api_search():
        query = request.args.get("q", "").strip()
        if len(query) < 1:
            return jsonify([])

        # Proxy yfinance search API (CORS bypass + clean JSON formatting)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(url, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            quotes = data.get("quotes", [])
            results = []
            for q in quotes:
                # Capture stock and index tickers, filter out empty fields
                symbol = q.get("symbol")
                name = q.get("longname") or q.get("shortname")
                exch = q.get("exchange")
                if symbol and name:
                    results.append({"ticker": symbol, "name": name, "exchange": exch})
            return jsonify(results[:10])
        except Exception:
            return jsonify([])

    # Auto-complete API route for Weather tab (search location)
    @app.route("/api/weather/search", methods=["GET"])
    def api_weather_search():
        query = request.args.get("q", "").strip()
        if len(query) < 2:
            return jsonify([])

        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=10&language=en&format=json"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            out = []
            for item in results:
                name = item.get("name")
                admin1 = item.get("admin1")
                country = item.get("country")
                lat = item.get("latitude")
                lon = item.get("longitude")
                
                # Format location label
                location_label = f"{name}"
                if admin1:
                    location_label += f", {admin1}"
                if country:
                    location_label += f", {country}"
                    
                out.append({
                    "label": location_label,
                    "lat": lat,
                    "lon": lon
                })
            return jsonify(out)
        except Exception:
            return jsonify([])

    # Auto-complete API route for Economics tab
    @app.route("/api/economics/search", methods=["GET"])
    def api_economics_search():
        query = request.args.get("q", "").strip().lower()
        if len(query) < 1:
            return jsonify([])

        results = []
        for s in POPULAR_FRED_SERIES:
            if query in s["id"].lower() or query in s["name"].lower():
                results.append({
                    "id": s["id"],
                    "label": f"{s['id']} - {s['name']} ({s['freq']})",
                    "name": s["name"]
                })
        return jsonify(results[:10])

    return app


def _parse_horizon(value_str: str) -> int:
    """Parse horizon day counts safely."""
    try:
        value = int(value_str)
    except (ValueError, TypeError):
        return 30
    return min(max(value, 5), 90)


def _build_finance_forecast(ticker: str, horizon: int) -> dict:
    # Recent actuals (for chart context + current price).
    hist = fetch_stock_data(ticker, period="3mo")
    if hist.empty:
        raise ValueError(f"No historical data was found for ticker '{ticker}' over the last 3 months.")
    current_price = float(hist["y"].iloc[-1])

    # Fetch company metadata and valuation info
    company_info = fetch_company_info(ticker)

    forecast_df = get_forecast(ticker, days_to_predict=horizon)
    
    # Load model diagnostics
    model_data = load_model(ticker)
    diagnostics = model_data.get("diagnostics", {})

    final_predicted = float(forecast_df["hybrid_val"].iloc[-1])

    price_change = final_predicted - current_price
    pct_change = (price_change / current_price) * 100.0 if current_price else 0.0

    chart_html = _render_chart(hist, forecast_df, horizon, y_title="Price (USD)", tick_prefix="$", tick_suffix="", diagnostics=diagnostics)

    rows = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "prophet": row["prophet_val"],
            "xgb": row["xgb_val"],
            "hybrid": row["hybrid_val"],
        }
        for row in forecast_df.to_dict("records")
    ]

    return {
        "current_price": current_price,
        "final_predicted": final_predicted,
        "price_change": price_change,
        "pct_change": pct_change,
        "forecast_rows": rows,
        "chart_html": chart_html,
        "horizon": horizon,
        "ticker": ticker,
        "company_info": company_info,
        "diagnostics": diagnostics,
        "unit_prefix": "$",
        "unit_suffix": ""
    }


def _build_weather_forecast(lat: float, lon: float, city_name: str, horizon: int) -> dict:
    cache_key = f"WEATHER_{lat:.4f}_{lon:.4f}".replace("-", "MINUS")
    
    # Fetch weather coordinates
    hist = fetch_weather_data(lat, lon)
    if hist.empty:
        raise ValueError(f"No historical weather data was found for {city_name}.")
    
    current_price = float(hist["y"].iloc[-1])

    # Run generic model
    forecast_df = get_generic_forecast(cache_key, horizon, lambda: fetch_weather_data(lat, lon))
    
    # Load model diagnostics
    model_data = load_model(cache_key)
    diagnostics = model_data.get("diagnostics", {})

    final_predicted = float(forecast_df["hybrid_val"].iloc[-1])

    price_change = final_predicted - current_price
    pct_change = (price_change / current_price) * 100.0 if current_price else 0.0

    # Render with custom labels
    chart_html = _render_chart(hist, forecast_df, horizon, y_title="Max Temperature (°C)", tick_prefix="", tick_suffix="°C", diagnostics=diagnostics)

    rows = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "prophet": row["prophet_val"],
            "xgb": row["xgb_val"],
            "hybrid": row["hybrid_val"],
        }
        for row in forecast_df.to_dict("records")
    ]

    company_info = {
        "name": f"Daily Weather Profile: {city_name}",
        "sector": "Meteorology & Climate",
        "industry": f"Coordinates: {lat:.4f}°N, {lon:.4f}°E",
        "employees": "N/A",
        "summary": f"Maximum daily surface temperature predictions for {city_name} modeled dynamically using high-resolution Open-Meteo climate datasets. Standard Bayesian trend and short-term residuals are calculated below.",
        "market_cap": "N/A",
        "pe_ratio": "N/A",
        "beta": "N/A",
        "recommendation": "N/A"
    }

    return {
        "current_price": current_price,
        "final_predicted": final_predicted,
        "price_change": price_change,
        "pct_change": pct_change,
        "forecast_rows": rows,
        "chart_html": chart_html,
        "horizon": horizon,
        "ticker": cache_key,
        "company_info": company_info,
        "diagnostics": diagnostics,
        "unit_prefix": "",
        "unit_suffix": "°C",
        "lat": lat,
        "lon": lon,
        "city_name": city_name
    }


def _build_economics_forecast(series_id: str, horizon: int) -> dict:
    # Find series metadata in POPULAR_FRED_SERIES
    series_info = next((s for s in POPULAR_FRED_SERIES if s["id"] == series_id), None)
    if series_info:
        series_name = series_info["name"]
        unit_freq = series_info["freq"]
    else:
        series_name = f"Economic Indicator {series_id}"
        unit_freq = "Monthly"
        
    unit_suffix = "%" if "%" in series_name else ""
    unit_prefix = "$" if "Price" in series_name or "Sales" in series_name or "Profits" in series_name or "Spend" in series_name else ""

    # Fetch FRED indicator
    hist = fetch_economic_data(series_id)
    if hist.empty:
        raise ValueError(f"No historical macroeconomic data was found for series ID '{series_id}' (FRED).")
    
    current_price = float(hist["y"].iloc[-1])

    # Run generic model
    forecast_df = get_generic_forecast(series_id, horizon, lambda: fetch_economic_data(series_id))
    
    # Load model diagnostics
    model_data = load_model(series_id)
    diagnostics = model_data.get("diagnostics", {})

    final_predicted = float(forecast_df["hybrid_val"].iloc[-1])

    price_change = final_predicted - current_price
    pct_change = (price_change / current_price) * 100.0 if current_price else 0.0

    # Render with custom economic labels
    chart_html = _render_chart(hist, forecast_df, horizon, y_title=f"Value ({unit_suffix or 'units'})", tick_prefix=unit_prefix, tick_suffix=unit_suffix, diagnostics=diagnostics)

    rows = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "prophet": row["prophet_val"],
            "xgb": row["xgb_val"],
            "hybrid": row["hybrid_val"],
        }
        for row in forecast_df.to_dict("records")
    ]

    company_info = {
        "name": f"Economic Indicator Profile: {series_name}",
        "sector": "Macroeconomic Policy",
        "industry": f"Frequency: {unit_freq}",
        "employees": "N/A",
        "summary": f"Historical and forecasted time-series rates for the {series_name} series sourced directly from the Federal Reserve Bank (FRED) data repository.",
        "market_cap": "N/A",
        "pe_ratio": "N/A",
        "beta": "N/A",
        "recommendation": "N/A"
    }

    return {
        "current_price": current_price,
        "final_predicted": final_predicted,
        "price_change": price_change,
        "pct_change": pct_change,
        "forecast_rows": rows,
        "chart_html": chart_html,
        "horizon": horizon,
        "ticker": series_id,
        "company_info": company_info,
        "diagnostics": diagnostics,
        "unit_prefix": unit_prefix,
        "unit_suffix": unit_suffix
    }


def _render_chart(hist: pd.DataFrame, forecast: pd.DataFrame, horizon: int, y_title: str = "Price (USD)", tick_prefix: str = "$", tick_suffix: str = "", diagnostics: dict = None) -> str:
    """Build a Plotly line chart and return its embeddable HTML."""
    fig = go.Figure()

    # 1. Add Custom Hover Templates for standard traces
    hist_hover = "<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> " + tick_prefix + "%{y:.2f}" + tick_suffix + "<extra></extra>"
    hybrid_hover = "<b>Date:</b> %{x|%Y-%m-%d}<br><b>Hybrid Forecast:</b> " + tick_prefix + "%{y:.2f}" + tick_suffix + "<extra></extra>"
    prophet_hover = "<b>Date:</b> %{x|%Y-%m-%d}<br><b>Prophet Baseline:</b> " + tick_prefix + "%{y:.2f}" + tick_suffix + "<extra></extra>"
    xgb_hover = "<b>Date:</b> %{x|%Y-%m-%d}<br><b>XGBoost Residual:</b> " + tick_prefix + "%{y:.2f}" + tick_suffix + "<extra></extra>"

    # 2. Plotly Traces
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="lines",
            name="Actual history",
            line={"color": "#0b0c0c", "width": 2},
            hovertemplate=hist_hover
        )
    )
    
    # Add Uncertainty Bands (Prophet lower/upper boundaries)
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["hybrid_lower"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["hybrid_upper"],
            mode="lines",
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(29, 112, 184, 0.12)', # GDS Blue with 12% opacity
            name="Uncertainty interval",
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["hybrid_val"],
            mode="lines",
            name="Hybrid forecast",
            line={"color": "#1d70b8", "width": 2.5},
            hovertemplate=hybrid_hover
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["prophet_val"],
            mode="lines",
            name="Prophet component",
            line={"color": "#d4351c", "dash": "dot"},
            hovertemplate=prophet_hover
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["xgb_val"],
            mode="lines",
            name="XGBoost component",
            line={"color": "#00703c", "dash": "dot"},
            hovertemplate=xgb_hover
        )
    )

    # 3. Add Vertical Forecast Divider Shape
    forecast_start = forecast["ds"].iloc[0]
    forecast_end = forecast["ds"].iloc[-1]
    
    # Draw Divider line
    fig.add_vline(
        x=forecast_start,
        line_width=1.5,
        line_dash="dash",
        line_color="#505a5f",
        opacity=0.8
    )
    
    # Shade Forecast region
    fig.add_vrect(
        x0=forecast_start,
        x1=forecast_end,
        fillcolor="#0b0c0c",
        opacity=0.04, # extremely light GDS gray overlay
        layer="below",
        line_width=0
    )

    # 4. Add Prophet Changepoint indicators
    if diagnostics and "changepoints" in diagnostics:
        for cp in diagnostics["changepoints"]:
            cp_date = pd.to_datetime(cp["date"])
            # Ensure the changepoint lies within the historical range we are plotting
            if cp_date in hist["ds"].values:
                fig.add_vline(
                    x=cp_date,
                    line_width=1,
                    line_dash="dot",
                    line_color="#a1a1a1",
                    opacity=0.6
                )

    # 5. Add Horizontal Benchmark lines (Average, Max, Min of historical data)
    hist_avg = hist["y"].mean()
    hist_max = hist["y"].max()
    hist_min = hist["y"].min()
    
    fig.add_hline(
        y=hist_avg,
        line_width=1,
        line_dash="dash",
        line_color="#28a197", # GDS turquoise
        opacity=0.7,
        annotation_text="Avg",
        annotation_position="bottom left"
    )
    fig.add_hline(
        y=hist_max,
        line_width=1,
        line_dash="dash",
        line_color="#b1b4b6",
        opacity=0.5,
        annotation_text="Max",
        annotation_position="top left"
    )
    fig.add_hline(
        y=hist_min,
        line_width=1,
        line_dash="dash",
        line_color="#b1b4b6",
        opacity=0.5,
        annotation_text="Min",
        annotation_position="bottom left"
    )

    # 6. Configure Range Zoom & Slider
    # Zoom pre-set to last 60 periods to avoid crowded starts
    default_zoom_periods = 60 if len(hist) > 60 else len(hist)
    default_start = hist["ds"].iloc[-default_zoom_periods]
    
    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=520, # Increased height to accommodate the range slider comfortably
        legend={
            "orientation": "h",
            "y": -0.35,
            "x": 0,
        },
        xaxis=dict(
            title=None,
            type="date",
            range=[default_start, forecast_end],
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=14, label="14d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                font=dict(size=12, family="Arial")
            ),
            rangeslider=dict(visible=True)
        ),
        yaxis={"title": y_title, "tickprefix": tick_prefix, "ticksuffix": tick_suffix},
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)