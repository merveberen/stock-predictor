from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import pandas as pd
from ml_models import predict_linear_regression, predict_moving_average, predict_random_forest
from datetime import datetime, timedelta
import json
import numpy as np
CACHE_DURATION = 86400  # 1 day in seconds

app = FastAPI()

# Allow CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Polygon.io API key (set your key here or use environment variable)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "HhBBxnV4WKUzoE08bEJXNgU3JoutncqA")

ALLOWED_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V", "JNJ", "UNH", "XOM", "WMT", "PG", "MA", "CVX", "HD", "KO", "PFE", "PEP", "ABBV", "MRK", "COST", "DIS", "MCD", "INTC", "CSCO", "CMCSA", "GLD", "SLV", "USO", "SPY", "IWM", "EFA", "EEM"
]

PREDICTION_INTERVALS = [
    ("1_week", 5),
    ("2_weeks", 10),
    ("1_month", 21),
    ("3_months", 63),
    ("6_months", 126),
    ("1_year", 252),
    ("2_years", 504),
]

class PredictRequest(BaseModel):
    symbols: List[str]
    model: str

class PredictResponse(BaseModel):
    symbol: str
    predicted: List[float]
    model: str

class HistoryResponse(BaseModel):
    symbol: str
    dates: List[str]
    prices: List[float]

class PredictSummaryResponse(BaseModel):
    summary: dict

def load_cache(symbol):
    try:
        with open(f"cache_{symbol}.json", "r") as f:
            data = json.load(f)
        # Check if cache is for today
        from datetime import date
        if data["date"] == str(date.today()):
            return data["results"]
    except Exception:
        pass
    return None

def save_cache(symbol, results):
    from datetime import date
    with open(f"cache_{symbol}.json", "w") as f:
        json.dump({"date": str(date.today()), "results": results}, f)

@app.get("/history", response_model=HistoryResponse)
def get_history(symbol: str):
    symbol = symbol.upper()
    if symbol not in ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail="Symbol not supported yet.")
    # Try cache first
    results = load_cache(symbol)
    if results is None:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=730)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not fetch data from Polygon.io.")
        data = resp.json()
        if "results" not in data:
            raise HTTPException(status_code=400, detail="No data found for symbol.")
        results = data["results"]
        save_cache(symbol, results)
    dates = [str(pd.to_datetime(r["t"], unit="ms").date()) for r in results]
    prices = [r["c"] for r in results]
    return HistoryResponse(symbol=symbol, dates=dates, prices=prices)

@app.post("/predict", response_model=PredictSummaryResponse)
def predict_summary(request: PredictRequest):
    if not request.symbols or len(request.symbols) > 5:
        raise HTTPException(status_code=400, detail="Select 1-5 symbols.")
    for symbol in request.symbols:
        if symbol not in ALLOWED_SYMBOLS:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not supported.")
    summary = {}
    model_map = {
        "linear_regression": predict_linear_regression,
        "moving_average": predict_moving_average,
        "random_forest": predict_random_forest,
    }
    if request.model not in model_map:
        raise HTTPException(status_code=400, detail="Model not supported.")
    model_func = model_map[request.model]
    for symbol in request.symbols:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=730)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not fetch data for {symbol}.")
        data = resp.json()
        if "results" not in data:
            raise HTTPException(status_code=400, detail=f"No data found for {symbol}.")
        results = data["results"]
        prices = [r["c"] for r in results]
        interval_results = {}
        for interval_name, days in PREDICTION_INTERVALS:
            try:
                preds = model_func(prices, days)
                interval_results[interval_name] = float(preds[-1]) if preds else None
            except Exception as e:
                interval_results[interval_name] = None
        # Add full 2-year curve
        try:
            curve_preds = model_func(prices, 504)
            if not curve_preds or len(curve_preds) < 504:
                # If model returns a single value or too short, repeat last value
                last_val = curve_preds[-1] if curve_preds else (prices[-1] if prices else 0)
                curve_preds = [last_val] * 504
        except Exception as e:
            last_val = prices[-1] if prices else 0
            curve_preds = [last_val] * 504
        summary[symbol] = {
            request.model: interval_results,
            'curves_2_years': {request.model: curve_preds}
        }
        if prices:
            summary[symbol]['actual'] = prices[-1]
    return {"summary": summary} 