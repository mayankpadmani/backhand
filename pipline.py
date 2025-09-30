# app.py
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import (
    load_data, build_preprocessor, prepare_train_test,
    train_models, save_model, load_model,
    NUM_FEATURES, CAT_FEATURES, TARGET
)
from joblib import dump, load
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# --- MySQL setup ---
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "team12345")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "agrisense_ai")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# First connect to MySQL server (create DB if not exists)
root_engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/")
with root_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}"))
    conn.commit()

# Now connect to the specific DB
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- Define table ---
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_used = Column(String(10))
    state = Column(String(100))
    district = Column(String(100))
    crop_type = Column(String(100))
    season = Column(String(50))
    soil_type = Column(String(50))
    irrigation_method = Column(String(50))
    area_hectares = Column(Float)
    temperature_c = Column(Float)
    rainfall_mm = Column(Float)
    ph_level = Column(Float)
    nitrogen_kg_ha = Column(Float)
    phosphorus_kg_ha = Column(Float)
    potassium_kg_ha = Column(Float)
    predicted_yield = Column(Float)
    recommendations = Column(Text)

# Create table if not exists
Base.metadata.create_all(bind=engine)

MODEL_DIR = "models"
RF_PATH = os.path.join(MODEL_DIR, "rf_pipeline.joblib")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_pipeline.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
DATA_PATH = "/mnt/data/dataread.csv"   # your uploaded CSV

os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# --- Columns for prediction ---
NUM_CAT = [
    'Area_Hectares','Temperature_C','Rainfall_mm','pH_Level',
    'Nitrogen_kg_ha','Phosphorus_kg_ha','Potassium_kg_ha',
    'State','District','Crop_Type','Season','Soil_Type','Irrigation_Method'
]

# --- Utility: Recommendation rules ---
def generate_recommendations(input_row, predicted_yield, dataset_stats=None):
    recs = []
    N = input_row.get('Nitrogen_kg_ha', None)
    P = input_row.get('Phosphorus_kg_ha', None)
    K = input_row.get('Potassium_kg_ha', None)
    ph = input_row.get('pH_Level', None)
    rainfall = input_row.get('Rainfall_mm', None)
    irrigation = input_row.get('Irrigation_Method', None)

    if N is not None and N < 60:
        recs.append("Increase nitrogen application (N) — consider +20-40 kg/ha as per crop stage.")
    if P is not None and P < 30:
        recs.append("Low phosphorus detected — apply P fertilizer (DAP) as recommended.")
    if K is not None and K < 40:
        recs.append("Potassium low — apply potash based on crop requirement.")
    if ph is not None and ph < 5.5:
        recs.append("Soil acidic — consider liming to increase pH toward neutral.")
    if rainfall is not None and rainfall < 200 and irrigation in ['Rainfed','None','Low']:
        recs.append("Rainfall low — schedule supplementary irrigation or adopt micro-irrigation.")
    if dataset_stats:
        med = dataset_stats.get('median_yield', None)
        if med and predicted_yield < med:
            recs.append("Predicted yield below regional median — optimize fertilization and irrigation.")
    recs.append("Monitor for pests if humidity > 80% or after extended rainfall.")
    return recs

# --- Endpoint: train models ---
@app.route('/train', methods=['POST'])
def train():
    params = request.get_json(force=True, silent=True) or {}
    rf_params = params.get('rf_params', None)
    xgb_params = params.get('xgb_params', None)
    try:
        df = load_data(DATA_PATH)
        preprocessor = build_preprocessor()
        X_train, X_test, y_train, y_test = prepare_train_test(df)
        rf_pipe, xgb_pipe = train_models(X_train, y_train, preprocessor, rf_params, xgb_params)
        save_model(rf_pipe, RF_PATH)
        save_model(xgb_pipe, XGB_PATH)

        from sklearn.metrics import mean_squared_error, r2_score
        rf_preds = rf_pipe.predict(X_test)
        xgb_preds = xgb_pipe.predict(X_test)
        return jsonify({
            "status": "trained",
            "rf_mse": float(mean_squared_error(y_test, rf_preds)),
            "rf_r2": float(r2_score(y_test, rf_preds)),
            "xgb_mse": float(mean_squared_error(y_test, xgb_preds)),
            "xgb_r2": float(r2_score(y_test, xgb_preds))
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# --- Endpoint: predict single ---
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    model_choice = request.args.get('model', 'rf')
    model_path = RF_PATH if model_choice == 'rf' else XGB_PATH
    if not os.path.exists(model_path):
        return jsonify({"status":"error","message":"Model not trained. Call /train first."}), 400
    try:
        model = load_model(model_path)
        X_single = pd.DataFrame([payload])
        for col in NUM_CAT:
            if col not in X_single.columns:
                X_single[col] = np.nan
        pred = model.predict(X_single)[0]

        df = pd.read_csv(DATA_PATH)
        median_yield = float(df[TARGET].median()) if TARGET in df.columns else None
        recs = generate_recommendations(X_single.iloc[0].to_dict(), float(pred), {'median_yield': median_yield})

        # Store in MySQL
        db = SessionLocal()
        new_pred = Prediction(
            model_used=model_choice,
            state=payload.get("State"),
            district=payload.get("District"),
            crop_type=payload.get("Crop_Type"),
            season=payload.get("Season"),
            soil_type=payload.get("Soil_Type"),
            irrigation_method=payload.get("Irrigation_Method"),
            area_hectares=payload.get("Area_Hectares"),
            temperature_c=payload.get("Temperature_C"),
            rainfall_mm=payload.get("Rainfall_mm"),
            ph_level=payload.get("pH_Level"),
            nitrogen_kg_ha=payload.get("Nitrogen_kg_ha"),
            phosphorus_kg_ha=payload.get("Phosphorus_kg_ha"),
            potassium_kg_ha=payload.get("Potassium_kg_ha"),
            predicted_yield=float(pred),
            recommendations=json.dumps(recs)
        )
        db.add(new_pred)
        db.commit()
        db.close()

        return jsonify({
            "predicted_yield_tonnes_per_ha": float(pred),
            "recommendations": recs,
            "model": model_choice
        })
    except Exception as e:
        return jsonify({"status":"error","message": str(e)}), 400

# --- Endpoint: bulk predict ---
@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    file = request.files.get('file')
    model_choice = request.args.get('model', 'rf')
    model_path = RF_PATH if model_choice == 'rf' else XGB_PATH
    if not file:
        return jsonify({"status":"error","message":"No file uploaded"}), 400
    if not os.path.exists(model_path):
        return jsonify({"status":"error","message":"Model not trained. Call /train first."}), 400
    try:
        df = pd.read_csv(file)
        model = load_model(model_path)
        for col in NUM_CAT:
            if col not in df.columns:
                df[col] = np.nan
        preds = model.predict(df[NUM_CAT])
        df['Predicted_Yield_tonnes_per_ha'] = preds
        df['Low_vs_median'] = df['Predicted_Yield_tonnes_per_ha'] < df['Predicted_Yield_tonnes_per_ha'].median()
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({"status":"error","message": str(e)}), 400

# --- Health check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
