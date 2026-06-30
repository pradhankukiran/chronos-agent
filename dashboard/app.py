"""Chronos Forecasting Service - Flask web application.

A GOV.UK Design System (govuk-frontend) front-end for the hybrid
Prophet + XGBoost forecasting engine in `core.models`.
"""
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
# the dashboard directory (e.g. `flask --app dashboard.app run`).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.models import get_forecast          # noqa: E402
from core.data import fetch_stock_data        # noqa: E402

import pandas as pd                           # noqa: E402
import plotly.graph_objects as go            # noqa: E402
import requests                               # noqa: E402



def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

    # The compiled govuk-frontend CSS references fonts and images with
    # root-relative URLs (`/assets/fonts/...`, `/assets/images/...`). Serve
    # them from the local static/assets directory so they resolve correctly.
    @app.route("/assets/<path:filename>")
    def govuk_assets(filename: str):
        return send_from_directory(
            os.path.join(app.static_folder, "assets"), filename
        )

    @app.route("/api/search")
    def search_tickers():
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify([])
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=8&newsCount=0"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                quotes = data.get("quotes", [])
                results = [
                    {
                        "symbol": q.get("symbol"),
                        "name": q.get("shortname") or q.get("longname") or q.get("symbol"),
                        "exch": q.get("exchange")
                    }
                    for q in quotes if q.get("symbol")
                ]
                return jsonify(results)
            return jsonify([])
        except Exception:
            return jsonify([])

    @app.route("/cookie-preference", methods=["POST"])
    def cookie_preference():
        """Record the user's cookie choice and dismiss the banner."""
        choice = request.form.get("cookies", "reject")
        resp = redirect(url_for("index", **request.args))
        resp.set_cookie(
            "cookies_policy",
            value=f"{{\"analytics\":{str(choice == 'accept').lower()}}}",
            max_age=365 * 24 * 60 * 60,
            httponly=True,
            samesite="Lax",
        )
        return resp

    @app.route("/", methods=["GET"])
    def index():
        ticker = (request.args.get("ticker") or "").upper().strip()
        horizon_raw = request.args.get("horizon", "30")

        context = {
            "ticker": ticker,
            "horizon": _coerce_horizon(horizon_raw),
            "has_results": False,
            "error": None,
        }

        if ticker:
            try:
                context.update(_build_forecast(ticker, context["horizon"]))
                context["has_results"] = True
            except Exception as exc:  # noqa: BLE001 - surface a friendly message
                context["error"] = str(exc)

        return render_template("index.html", **context)

    return app


def _coerce_horizon(raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 30
    return min(max(value, 5), 90)


def _build_forecast(ticker: str, horizon: int) -> dict:
    # Recent actuals (for chart context + current price).
    hist = fetch_stock_data(ticker, period="3mo")
    current_price = float(hist["y"].iloc[-1])

    forecast_df = get_forecast(ticker, days_to_predict=horizon)
    final_predicted = float(forecast_df["hybrid_val"].iloc[-1])

    price_change = final_predicted - current_price
    pct_change = (price_change / current_price) * 100.0 if current_price else 0.0

    chart_html = _render_chart(hist, forecast_df, horizon)

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
    }


def _render_chart(hist: pd.DataFrame, forecast: pd.DataFrame, horizon: int) -> str:
    """Build a Plotly line chart and return its embeddable HTML."""
    fig = go.Figure()

    actual = hist.tail(60)
    fig.add_trace(
        go.Scatter(
            x=actual["ds"],
            y=actual["y"],
            mode="lines",
            name="Actual price",
            line={"color": "#0b0c0c", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["hybrid_val"],
            mode="lines",
            name="Hybrid forecast",
            line={"color": "#1d70b8", "width": 2},
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
        yaxis={"title": "Price (USD)", "tickprefix": "$"},
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True},
    )


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False)