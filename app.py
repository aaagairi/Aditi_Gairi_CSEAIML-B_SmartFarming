from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import joblib
import requests
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# ==================== WEATHER FORECAST MODEL ====================

def get_coordinates(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    response = requests.get(geo_url)
    data = response.json()

    if "results" in data and data["results"]:
        latitude = data["results"][0]["latitude"]
        longitude = data["results"][0]["longitude"]
        return latitude, longitude
    else:
        raise ValueError("City not found. Please try another city.")

def fetch_weather_forecast(lat, lon):
    api_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,"
        f"precipitation_sum,windspeed_10m_max&forecast_days=7&timezone=Asia/Kolkata"
    )
    response = requests.get(api_url)
    weather = response.json()["daily"]

    df = pd.DataFrame({
        "date": weather["time"],
        "temperature": weather["temperature_2m_max"],
        "precipitation": weather["precipitation_sum"],
        "wind_speed": weather["windspeed_10m_max"]
    })
    return df

def generate_alerts(row):
    alerts = []
    if row["predicted_temperature"] > 35:
        alerts.append("High temperature – risk of crop heat stress")
    if row["precipitation"] > 15:
        alerts.append("Heavy rain – possible flooding or waterlogging")
    if row["wind_speed"] > 30:
        alerts.append("High wind – risk of crop damage")
    return alerts

def train_weather_model(df):
    X = df[["precipitation", "wind_speed"]]
    y = df["temperature"]

    model = SVR()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, os.path.join(model_dir, "weather_model.pkl"))
    return model

def predict_weather(model, df):
    df["predicted_temperature"] = model.predict(df[["precipitation", "wind_speed"]])
    df["alerts"] = df.apply(generate_alerts, axis=1)
    return df

# ==================== CROP YIELD MODEL ====================

# Load or create crop yield model
def load_or_train_crop_yield_model():
    crop_model_path = os.path.join(model_dir, "crop_yield_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    label_encoders_path = os.path.join(model_dir, "label_encoders.pkl")
    
    if os.path.exists(crop_model_path) and os.path.exists(scaler_path) and os.path.exists(label_encoders_path):
        # Load existing model
        model = joblib.load(crop_model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(label_encoders_path)
    else:
        # For demo purposes, we'll create a dummy model
        # In a real application, you would load your actual data and train the model
        print("Training crop yield model on sample data...")
        
        # Create sample data (in production, replace with your real data)
        # This is just for demo purposes - you should use your actual CSV data
        sample_data = {
            'Crop': ['Rice', 'Wheat', 'Maize', 'Potato'] * 25,
            'Season': ['Kharif', 'Rabi', 'Whole Year', 'Summer'] * 25,
            'State': ['Maharashtra', 'Punjab', 'Karnataka', 'West Bengal'] * 25,
            'Area': np.random.uniform(10, 100, 100),
            'Annual_Rainfall': np.random.uniform(500, 2000, 100),
            'Fertilizer': np.random.uniform(100, 500, 100),
            'Pesticide': np.random.uniform(5, 30, 100),
            'Yield': np.random.uniform(20, 50, 100)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Define expected seasons
        expected_seasons = ["Kharif", "Rabi", "Whole Year", "Summer", "Autumn", "Winter"]
        
        # Process categorical variables
        label_encoders = {}
        categorical_cols = ["Crop", "Season", "State"]
        
        for col in categorical_cols:
            le = LabelEncoder()
            if col == "Season":
                le.fit(expected_seasons)
                df["Season"] = df["Season"].str.strip()
            else:
                le.fit(df[col])
            df[f"{col}_encoded"] = le.transform(df[col])
            label_encoders[col] = le
        
        # Feature selection
        feature_cols = [
            "Area", "Annual_Rainfall", "Fertilizer", "Pesticide",
            "Crop_encoded", "Season_encoded", "State_encoded"
        ]
        
        X = df[feature_cols].copy()
        y = df["Yield"]
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model and preprocessing objects
        joblib.dump(model, crop_model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoders, label_encoders_path)
    
    return model, scaler, label_encoders, ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]

# Function to predict crop yield
def predict_crop_yield(model, scaler, label_encoders, numerical_cols, input_data):
    # Create a copy of the input
    example = input_data.copy()
    
    # Encode categorical variables
    for col in ["Crop", "Season", "State"]:
        le = label_encoders[col]
        if example[col] in le.classes_:
            example[f"{col}_encoded"] = le.transform([example[col]])[0]
        else:
            return {"error": f"Unknown {col}: {example[col]}. Available options: {', '.join(le.classes_)}"}
    
    # Create feature dataframe
    feature_cols = numerical_cols + ["Crop_encoded", "Season_encoded", "State_encoded"]
    features_df = pd.DataFrame([[
        example["Area"],
        example["Annual_Rainfall"],
        example["Fertilizer"],
        example["Pesticide"],
        example["Crop_encoded"],
        example["Season_encoded"],
        example["State_encoded"]
    ]], columns=feature_cols)
    
    # Scale numerical columns
    features_df[numerical_cols] = scaler.transform(features_df[numerical_cols])
    
    # Predict
    prediction = model.predict(features_df)[0]
    
    return {
        "crop": example["Crop"],
        "estimated_yield": round(prediction, 2),
        "unit": "tonnes per hectare",
        "date": datetime.now().strftime('%Y-%m-%d')
    }

# ==================== Initialize models ====================

# Load or create weather model
if os.path.exists(os.path.join(model_dir, "weather_model.pkl")):
    weather_model = joblib.load(os.path.join(model_dir, "weather_model.pkl"))
else:
    # Create a dummy model initially
    dummy_df = pd.DataFrame({
        "date": ["2025-04-01", "2025-04-02", "2025-04-03"],
        "temperature": [25, 26, 24],
        "precipitation": [0, 5, 10],
        "wind_speed": [10, 15, 20]
    })
    weather_model = train_weather_model(dummy_df)

# Load or create crop yield model and related objects
crop_yield_model, scaler, label_encoders, numerical_cols = load_or_train_crop_yield_model()

# ==================== FLASK ROUTES ====================

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Weather prediction page
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    forecast_data = None
    error = None
    
    if request.method == 'POST':
        try:
            city = request.form['city']
            
            # Get coordinates and fetch weather forecast
            latitude, longitude = get_coordinates(city)
            df = fetch_weather_forecast(latitude, longitude)
            
            # Get predictions and alerts
            forecast_df = predict_weather(weather_model, df)
            
            # Convert to dictionary for template
            forecast_data = []
            for _, row in forecast_df.iterrows():
                day_data = {
                    'date': row['date'],
                    'temperature': round(row['temperature'], 1),
                    'predicted_temperature': round(row['predicted_temperature'], 1),
                    'precipitation': row['precipitation'],
                    'wind_speed': row['wind_speed'],
                    'alerts': row['alerts']
                }
                forecast_data.append(day_data)
            
        except Exception as e:
            error = str(e)
    
    return render_template('weather.html', forecast=forecast_data, error=error)

# Crop yield prediction page
@app.route('/crop_yield', methods=['GET', 'POST'])
def crop_yield():
    prediction = None
    error = None
    
    # Get available options for dropdowns
    crop_options = list(label_encoders['Crop'].classes_)
    season_options = list(label_encoders['Season'].classes_)
    state_options = list(label_encoders['State'].classes_)
    
    if request.method == 'POST':
        try:
            # Get input features for crop yield prediction
            input_data = {
                'Crop': request.form['crop_type'],
                'Season': request.form['season'],
                'State': request.form['state'],
                'Area': float(request.form['area']),
                'Annual_Rainfall': float(request.form['rainfall']),
                'Fertilizer': float(request.form['fertilizer']),
                'Pesticide': float(request.form['pesticide'])
            }
            
            # Make prediction
            prediction = predict_crop_yield(
                crop_yield_model, 
                scaler, 
                label_encoders, 
                numerical_cols, 
                input_data
            )
            
            if 'error' in prediction:
                error = prediction['error']
                prediction = None
                
        except Exception as e:
            error = str(e)
    
    return render_template(
        'crop_yield.html', 
        prediction=prediction, 
        error=error, 
        crop_options=crop_options,
        season_options=season_options,
        state_options=state_options
    )

# Dashboard route for visualizing farm data
@app.route('/dashboard')
def dashboard():
    # For demo purposes, we're using static data
    # You could replace this with real predictions or database data
    weather_data = {
        'dates': ['2025-04-01', '2025-04-02', '2025-04-03', '2025-04-04', '2025-04-05'],
        'temperatures': [25, 26, 24, 27, 28],
        'rainfall': [0, 0, 5, 2, 0]
    }
    
    crop_data = {
        'crops': ['Wheat', 'Rice', 'Maize', 'Soybeans'],
        'yields': [3.5, 4.2, 5.8, 2.7]
    }
    
    return render_template('dashboard.html', weather_data=weather_data, crop_data=crop_data)

# API endpoints for mobile applications or front-end frameworks
@app.route('/api/weather', methods=['POST'])
def weather_api():
    try:
        data = request.get_json()
        city = data.get('city')
        
        if not city:
            return jsonify({
                'success': False,
                'error': 'City name is required'
            })
            
        # Get coordinates and fetch weather forecast
        latitude, longitude = get_coordinates(city)
        df = fetch_weather_forecast(latitude, longitude)
        
        # Get predictions and alerts
        forecast_df = predict_weather(weather_model, df)
        
        # Convert to list for JSON response
        forecast_data = []
        for _, row in forecast_df.iterrows():
            day_data = {
                'date': row['date'],
                'temperature': round(row['temperature'], 1),
                'predicted_temperature': round(row['predicted_temperature'], 1),
                'precipitation': row['precipitation'],
                'wind_speed': row['wind_speed'],
                'alerts': row['alerts']
            }
            forecast_data.append(day_data)
            
        return jsonify({
            'success': True,
            'forecast': forecast_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/crop_yield', methods=['POST'])
def crop_yield_api():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['crop_type', 'season', 'state', 'area', 'rainfall', 'fertilizer', 'pesticide']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                })
        
        # Get input features for crop yield prediction
        input_data = {
            'Crop': data['crop_type'],
            'Season': data['season'],
            'State': data['state'],
            'Area': float(data['area']),
            'Annual_Rainfall': float(data['rainfall']),
            'Fertilizer': float(data['fertilizer']),
            'Pesticide': float(data['pesticide'])
        }
        
        # Make prediction
        prediction = predict_crop_yield(
            crop_yield_model, 
            scaler, 
            label_encoders, 
            numerical_cols, 
            input_data
        )
        
        if 'error' in prediction:
            return jsonify({
                'success': False,
                'error': prediction['error']
            })
            
        return jsonify({
            'success': True,
            **prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)