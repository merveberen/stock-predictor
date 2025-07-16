import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def predict_linear_regression(prices, days, window=10, spx_prices=None):
    """Linear regression with improved stability and bounds checking"""
    if len(prices) < window + 1:
        return [prices[-1]] * days
    
    # Prepare lagged features
    X = []
    y = []
    for i in range(window, len(prices)):
        features = prices[i-window:i]
        if spx_prices is not None and len(spx_prices) == len(prices):
            features = features + spx_prices[i-window:i]
        X.append(features)
        y.append(prices[i])
    
    X = np.array(X)
    y = np.array(y)
    
    model = LinearRegression()
    model.fit(X, y)
    
    preds = []
    history = list(prices[-window:])
    spx_history = list(spx_prices[-window:]) if spx_prices is not None and len(spx_prices) == len(prices) else None
    
    for _ in range(days):
        features = history[-window:]
        if spx_history is not None:
            features = features + spx_history[-window:]
        
        next_pred = model.predict([features])[0]
        
        # Apply bounds checking - prevent extreme predictions
        last_price = history[-1]
        max_change = 0.1  # 10% maximum change per day
        next_pred = max(last_price * (1 - max_change), 
                       min(last_price * (1 + max_change), next_pred))
        
        preds.append(next_pred)
        history.append(next_pred)
        
        if spx_history is not None:
            # Use a more realistic projection for SPX
            spx_change = 0.001  # Small daily change
            spx_next = spx_history[-1] * (1 + np.random.normal(0, spx_change))
            spx_history.append(spx_next)
    
    return preds

def predict_moving_average(prices, days, window=5):
    """Moving average with trend adjustment"""
    if len(prices) < 2:
        return [prices[-1]] * days
    
    # Calculate recent trend
    recent_prices = prices[-min(20, len(prices)):]
    if len(recent_prices) >= 2:
        trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
    else:
        trend = 0
    
    # Dampen trend for long predictions
    trend_damping = 0.95  # Reduce trend impact over time
    
    preds = []
    history = list(prices)
    
    for day in range(days):
        if len(history) < window:
            avg = np.mean(history)
        else:
            avg = np.mean(history[-window:])
        
        # Apply dampened trend
        damped_trend = trend * (trend_damping ** day)
        prediction = avg + damped_trend
        
        # Ensure positive prices
        prediction = max(0.01, prediction)
        
        preds.append(prediction)
        history.append(prediction)
    
    return preds

def predict_random_forest(prices, days, n_estimators=100, window=10, spx_prices=None):
    """Random Forest with improved feature engineering and bounds"""
    if len(prices) < window + 1:
        return [prices[-1]] * days
    
    # Prepare features with additional technical indicators
    X = []
    y = []
    
    for i in range(window, len(prices)):
        features = list(prices[i-window:i])
        
        # Add technical indicators
        if i >= window:
            # Recent volatility
            volatility = np.std(prices[i-window:i])
            features.append(volatility)
            
            # Price momentum
            momentum = (prices[i-1] - prices[i-window]) / prices[i-window]
            features.append(momentum)
        
        if spx_prices is not None and len(spx_prices) == len(prices):
            features.extend(spx_prices[i-window:i])
        
        X.append(features)
        y.append(prices[i])
    
    X = np.array(X)
    y = np.array(y)
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    preds = []
    history = list(prices[-window:])
    spx_history = list(spx_prices[-window:]) if spx_prices is not None and len(spx_prices) == len(prices) else None
    
    for _ in range(days):
        features = list(history[-window:])
        
        # Add technical indicators
        volatility = np.std(history[-window:])
        features.append(volatility)
        
        momentum = (history[-1] - history[-window]) / history[-window]
        features.append(momentum)
        
        if spx_history is not None:
            features.extend(spx_history[-window:])
        
        next_pred = model.predict([features])[0]
        
        # Apply bounds checking
        last_price = history[-1]
        max_change = 0.08  # 8% maximum change per day
        next_pred = max(last_price * (1 - max_change), 
                       min(last_price * (1 + max_change), next_pred))
        
        # Ensure positive prices
        next_pred = max(0.01, next_pred)
        
        preds.append(next_pred)
        history.append(next_pred)
        
        if spx_history is not None:
            spx_change = 0.001
            spx_next = spx_history[-1] * (1 + np.random.normal(0, spx_change))
            spx_history.append(spx_next)
    
    return preds

def predict_prophet(prices, days, cap_multiplier=2.0, floor_multiplier=0.3):
    """Fixed Prophet model with proper configuration for stock prices"""
    
    if len(prices) < 30:  # Need sufficient data
        return [prices[-1]] * days
    
    # Create DataFrame with proper date index
    df = pd.DataFrame({
        'ds': pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq='D'),
        'y': prices
    })
    
    # Set cap and floor for logistic growth (prevents extreme predictions)
    recent_max = max(prices[-min(252, len(prices)):])  # Last year max
    recent_min = min(prices[-min(252, len(prices)):])  # Last year min
    
    df['cap'] = recent_max * cap_multiplier
    df['floor'] = recent_min * floor_multiplier
    
    # Configure Prophet for stock prices
    model = Prophet(
        growth='logistic',  # Use logistic growth instead of linear
        daily_seasonality=False,  # Turn off daily seasonality
        weekly_seasonality=False,  # Turn off weekly seasonality  
        yearly_seasonality=False,  # Turn off yearly seasonality
        seasonality_mode='multiplicative',  # Use multiplicative seasonality
        changepoint_prior_scale=0.05,  # Reduce trend flexibility
        seasonality_prior_scale=0.1,   # Reduce seasonality strength
        holidays_prior_scale=0.1,      # Reduce holiday effects
        interval_width=0.8,            # Confidence interval
        n_changepoints=25,             # Limit changepoints
        changepoint_range=0.8          # Only consider changepoints in first 80% of data
    )
    
    # Fit the model
    model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=days, freq='D')
    future['cap'] = recent_max * cap_multiplier
    future['floor'] = recent_min * floor_multiplier
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract predictions and apply additional bounds checking
    preds = forecast['yhat'][-days:].tolist()
    
    # Apply additional conservative bounds
    last_price = prices[-1]
    conservative_preds = []
    
    for i, pred in enumerate(preds):
        # Gradually reduce maximum allowed change over time
        max_daily_change = 0.05 * (0.98 ** i)  # Decreasing volatility over time
        
        if i == 0:
            reference_price = last_price
        else:
            reference_price = conservative_preds[-1]
        
        # Apply bounds
        bounded_pred = max(reference_price * (1 - max_daily_change),
                          min(reference_price * (1 + max_daily_change), pred))
        
        # Ensure positive prices
        bounded_pred = max(0.01, bounded_pred)
        
        conservative_preds.append(bounded_pred)
    
    return conservative_preds