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
        horizon_raw = request.args.get("horizon")

        has_results = False
        error_msg = None
        context = {
            "active_tab": "economics",
            "has_results": has_results,
            "error_msg": error_msg,
            "series_id": series_id or "UNRATE",
            "horizon": horizon_raw or "30",
        }

        # List of preset indicators
        context["indicators"] = [
            {"id": "UNRATE", "name": "U.S. Unemployment Rate (%)"},
            {"id": "CPIAUCSL", "name": "U.S. Consumer Price Index (CPI)"},
            {"id": "FEDFUNDS", "name": "Effective Federal Funds Interest Rate (%)"}
        ]

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

    chart_html = _render_chart(hist, forecast_df, horizon, y_title="Price (USD)", tick_prefix="$", tick_suffix="")

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
    chart_html = _render_chart(hist.tail(60), forecast_df, horizon, y_title="Max Temperature (°C)", tick_prefix="", tick_suffix="°C")

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
    series_map = {
        "UNRATE": {"name": "U.S. Unemployment Rate", "suffix": "%"},
        "CPIAUCSL": {"name": "U.S. Consumer Price Index (CPI)", "suffix": ""},
        "FEDFUNDS": {"name": "Effective Federal Funds Interest Rate", "suffix": "%"}
    }
    
    series_info = series_map.get(series_id, series_map["UNRATE"])
    series_name = series_info["name"]
    unit_suffix = series_info["suffix"]

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
    chart_html = _render_chart(hist.tail(24), forecast_df, horizon, y_title=f"Rate/Value ({unit_suffix or 'index'})", tick_prefix="", tick_suffix=unit_suffix)

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
        "industry": "National Statistics",
        "employees": "N/A",
        "summary": f"Historical and forecasted monthly rates for the {series_name} series sourced directly from the Federal Reserve Bank (FRED) data repository.",
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
        "unit_prefix": "",
        "unit_suffix": unit_suffix
    }


def _render_chart(hist: pd.DataFrame, forecast: pd.DataFrame, horizon: int, y_title: str = "Price (USD)", tick_prefix: str = "$", tick_suffix: str = "") -> str:
    """Build a Plotly line chart and return its embeddable HTML."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="lines",
            name="Actual history",
            line={"color": "#0b0c0c", "width": 2},
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
            fillcolor='rgba(29, 112, 184, 0.15)', # GDS Blue with 15% opacity
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
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["prophet_val"],
            mode="lines",
            name="Prophet component",
            line={"color": "#d4351c", "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["xgb_val"],
            mode="lines",
            name="XGBoost component",
            line={"color": "#00703c", "dash": "dot"},
        )
    )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=420,
        legend={
            "orientation": "h",
            "y": -0.2,
            "x": 0,
        },
        xaxis={"title": None},
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