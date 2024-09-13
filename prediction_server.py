from flask import Flask, request, jsonify
import pickle
import numpy as np
import yfinance as yf
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold models
model_close = None
model_high = None
model_low = None

def load_models():
    global model_close, model_high, model_low
    try:
        with open('model_close.pkl', 'rb') as f:
            model_close = pickle.load(f)
        with open('model_high.pkl', 'rb') as f:
            model_high = pickle.load(f)
        with open('model_low.pkl', 'rb') as f:
            model_low = pickle.load(f)
        logger.info("All models loaded successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"Error loading models: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading models: {e}")
        return False

# Try to load models at startup
models_loaded = load_models()

def get_data(symbol, period="5d", interval="1m"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def create_features(data, lookback=5):
    if len(data) <= lookback:
        return np.array([])
    
    features = np.concatenate([
        data['Close'].iloc[-lookback:].values,
        data['Volume'].iloc[-lookback:].values,
        data['Close'].iloc[-lookback:].pct_change().values[1:],
        (data['Close'].iloc[-lookback:] - data['Close'].iloc[-lookback:].rolling(window=5).mean()).values
    ]).reshape(1, -1)
    
    return features

@app.route('/predict', methods=['POST'])
def predict():
    global models_loaded
    if not models_loaded:
        models_loaded = load_models()
    
    if not models_loaded:
        return jsonify({'error': 'Models not available. Please try again later.'})

    symbol = request.json['symbol']
    data = get_data(symbol)
    
    if data is None or len(data) < 6:
        return jsonify({'error': 'Not enough data'})
    
    features = create_features(data)
    
    prediction_close = model_close.predict(features)[0]
    prediction_high = model_high.predict(features)[0]
    prediction_low = model_low.predict(features)[0]
    
    return jsonify({
        'predicted_close': float(prediction_close),
        'predicted_high': float(prediction_high),
        'predicted_low': float(prediction_low)
    })

@app.route('/')
def home():
    return "Welcome to the prediction server!"

if __name__ == '__main__':
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir()}")
    app.run(host='0.0.0.0', port=5000)