import os
from groq import Groq
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Groq API Key
# ----------------------------
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ----------------------------
# Config
# ----------------------------
CSV_FILE = "dataread.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_FEATURES = [
    "Area_Hectares", "Temperature_C", "Rainfall_mm", "pH_Level",
    "Nitrogen_kg_ha", "Phosphorus_kg_ha", "Potassium_kg_ha"
]
CAT_FEATURES = ["State", "District", "Crop_Type", "Season", "Soil_Type", "Irrigation_Method"]
TARGET = "Yield_Tonnes_per_Ha"

# ----------------------------
# Load data
# ----------------------------
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    print("✅ Data loaded:", df.shape)
    return df

# ----------------------------
# Preprocessor
# ----------------------------
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

# ----------------------------
# Train/test split
# ----------------------------
def prepare_train_test(df):
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Train models
# ----------------------------
def train_models(X_train, y_train, preprocessor):
    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    xgb_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    xgb_pipe.fit(X_train, y_train)
    return rf_pipe, xgb_pipe

# ----------------------------
# Evaluate models
# ----------------------------
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"📊 {name} → MSE: {mse:.4f}, R²: {r2:.4f}")
    return mse, r2

# ----------------------------
# Recommendations
# ----------------------------
# ----------------------------
# Recommendations 
# ----------------------------
def generate_recommendations(input_row, predicted_yield, capital):
    recs = []

    # Fertilizer balance
    if input_row["Nitrogen_kg_ha"] < 60:
        recs.append("Increase Nitrogen (Urea/Ammonium Sulphate). Cost ~₹2000/acre")
    if input_row["Phosphorus_kg_ha"] < 30:
        recs.append("Add Phosphorus (DAP/SSP). Cost ~₹2500/acre")
    if input_row["Potassium_kg_ha"] < 40:
        recs.append("Apply Potash (MOP/SOP). Cost ~₹1800/acre")
    if 6.0 <= input_row["pH_Level"] <= 7.5:
        recs.append("Soil pH optimal — maintain organic content.")

    # Soil correction
    if input_row["pH_Level"] < 5.5:
        recs.append("Soil acidic — apply lime/gypsum as per soil test.")
    elif input_row["pH_Level"] > 8:
        recs.append("Soil alkaline — add organic manure/green manure crops.")

    # Water management
    if input_row["Rainfall_mm"] < 200 and input_row["Irrigation_Method"] in ["Rainfed", "None", "Low"]:
        recs.append("Low rainfall — plan irrigation (drip/sprinkler recommended).")
    elif input_row["Rainfall_mm"] > 800:
        recs.append("Excess rainfall — improve drainage to avoid root damage.")

    # General pest & disease
    recs.append("Monitor pests and fungal diseases after heavy rains.")

    # 🔹 Crop-specific recommendations via Groq API
    crop_type = input_row["Crop_Type"]
    try:
        prompt = (
            f"Provide 3 short and practical recommendations for farmers growing '{crop_type}' in India. "
            f"Cover pest management, irrigation, and nutrient management. Keep it concise."
        )
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert Indian agricultural advisor."},
                {"role": "user", "content": prompt}
            ],
        )
        crop_recs = chat_completion.choices[0].message.content.split("\n")
        for rec in crop_recs:
            if rec.strip():
                recs.append(rec.strip())
    except Exception as e:
        recs.append(f"(⚠ Could not fetch AI-based crop-specific advice: {e})")

    # Season-specific
    if input_row["Season"].lower() == "rabi":
        recs.append("Rabi season — ensure timely sowing to maximize yield.")
    if input_row["Season"].lower() == "kharif":
        recs.append("Kharif season — weed control crucial during first 30 days.")

    # Budget-based
    if capital < 5000:
        recs.append("⚠ Limited budget — prioritize NPK balance, use compost/organic manure.")
    elif capital > 20000:
        recs.append("💰 High capital — invest in micro-irrigation and soil testing for precision farming.")
    try:
        prompt = (
            f"provide government schemes for the '{crop_type}' in India only like about 2 to 3 maximum 5 government schemes not more than that."
        )
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert Indian agricultural advisor."},
                {"role": "user", "content": prompt}
            ],
        )
        crop_recs = chat_completion.choices[0].message.content.split("\n")
        for rec in crop_recs:
            if rec.strip():
                recs.append(rec.strip())
    except Exception as e:
        recs.append(f"(⚠ Could not fetch AI-based crop-specific advice: {e})")
    return recs


# ----------------------------
# Groq crop recommendation
# ----------------------------
def get_groq_crop_recommendation(district, area_hectares, capital):
    prompt = (
        f"Recommend the best crop for district in one word {district}. "
        f"Land size {area_hectares:.2f} ha, capital ₹{capital}. "
        f"Return crop name and expected yield in two types just this three in exactly three words with one of them in kgs per acre(kg/ac) and another type units=(kgs per acre) divide by 20 output should be in (units/ac)."
    )
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert agricultural assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return chat_completion.choices[0].message.content

# ----------------------------
# Best crop by ML
# ----------------------------
def suggest_best_crop(df, model, sample_input):
    district = sample_input.get("District")
    area_hectares = sample_input.get("Area_Hectares", 1)

    district_df = df[df["District"] == district] if district in df["District"].values else df
    numeric_means = district_df[NUM_FEATURES].mean().to_dict()
    categorical_modes = district_df[CAT_FEATURES].mode().iloc[0].to_dict()

    avg_yields = df.groupby('Crop_Type')['Yield_Tonnes_per_Ha'].mean().to_dict()
    best_crop, best_score = None, 0

    print(f"\n🌱 Yield predictions for crops in {district}:")
    for crop in district_df["Crop_Type"].unique():
        input_copy = {**numeric_means, **categorical_modes, **sample_input, "Crop_Type": crop}
        sample_df = pd.DataFrame([input_copy])
        pred = model.predict(sample_df[NUM_FEATURES + CAT_FEATURES])[0]
        norm_score = pred / avg_yields[crop]
        print(f" - {crop}: {pred:.2f} t/ha (norm: {norm_score:.2f})")
        if norm_score > best_score:
            best_score, best_crop = norm_score, crop

    return best_crop, best_score


#conversion part
# ----------------------------
# Yield-unit conversions
# ----------------------------
TONNES_PER_HA_TO_KG_PER_AC = 404.69   # 1 t/ha → 404.69 kg/ac
UNIT_WEIGHT_KG = 20                  # 1 “unit” = 20 kg

def yield_to_kg_ac_and_units(t_per_ha: float):
    """Convert tonnes/ha → kg/acre and 20-kg units."""
    kg_per_ac = t_per_ha * TONNES_PER_HA_TO_KG_PER_AC
    units = kg_per_ac / UNIT_WEIGHT_KG
    return kg_per_ac, units




# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    df = load_data(CSV_FILE)
    preprocessor = build_preprocessor()
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    print("🚀 Training models...")
    rf_pipe, xgb_pipe = train_models(X_train, y_train, preprocessor)

    print("\n✅ Model Performance:")
    evaluate(rf_pipe, X_test, y_test, "RandomForest")
    evaluate(xgb_pipe, X_test, y_test, "XGBoost")

    # Save models
    dump(rf_pipe, os.path.join(MODEL_DIR, "rf_pipeline.joblib"))
    dump(xgb_pipe, os.path.join(MODEL_DIR, "xgb_pipeline.joblib"))

    # ---------------- User Input ----------------
    print("\n🌾 Enter farm details for prediction:")
    district = input("District: ")
    acres = float(input("Land size (in acres): "))
    capital = float(input("Capital (₹): "))

    area_hectares = acres * 0.4047
    defaults = df.mean(numeric_only=True).to_dict()

    # Ask farmer if they want to fix a crop
    fixed_crop = input("Do you want to grow a specific crop? (yes/no): ").strip().lower()
    chosen_crop = None
    if fixed_crop == "yes":
        chosen_crop = input("Enter the crop name: ").strip()

    # ---------------- Sample Input ----------------
    sample_input = {
        "Area_Hectares": area_hectares,
        "Temperature_C": defaults.get("Temperature_C", 28),
        "Rainfall_mm": defaults.get("Rainfall_mm", 200),
        "pH_Level": defaults.get("pH_Level", 6.5),
        "Nitrogen_kg_ha": defaults.get("Nitrogen_kg_ha", 50),
        "Phosphorus_kg_ha": defaults.get("Phosphorus_kg_ha", 25),
        "Potassium_kg_ha": defaults.get("Potassium_kg_ha", 35),
        "State": "Gujarat",
        "District": district,
        "Crop_Type": chosen_crop if chosen_crop else "Wheat",  # farmer's crop OR default
        "Season": "Rabi",
        "Soil_Type": "Loamy",
        "Irrigation_Method": "Rainfed"
    }

    sample_df = pd.DataFrame([sample_input])
    pred = rf_pipe.predict(sample_df[NUM_FEATURES + CAT_FEATURES])[0]

    # ---------------- Output ----------------
    if chosen_crop:  # Case 1: Farmer fixed a crop
        print(f"\n🌱 Yield Prediction for chosen crop: {chosen_crop}")
        print(f" - Expected yield: {pred:.2f} t/ha")
        kg_ac, units = yield_to_kg_ac_and_units(pred)
        print(f" - In kg per acre: {kg_ac:,.0f} kg/ac")
        print(f" - In 20 kg units: {units:,.0f} units")
        print("\n🌱 Suggested Crops")
        best_crop, best_crop_score = suggest_best_crop(df, rf_pipe, sample_input)
        print("\nML Model Suggestion and yield prediction:")
        print(f" - ML Model Suggestion: {best_crop} (score {best_crop_score:.2f})")
        print(f" - Expected yield for given inputs: {pred:.2f} t/ha")
        kg_ac, units = yield_to_kg_ac_and_units(pred)
        print(f" - In kg per acre: {kg_ac:,.0f} kg/ac")
        print(f" - In 20 kg units: {units:,.0f} units")

        groq_response = get_groq_crop_recommendation(district, area_hectares, capital)
        print("\n🌱 Another Suggestion :")
        print(f" - {groq_response}")


    else:  # Case 2: Suggest best crop
        print("\n🌱 Suggested Crops")
        best_crop, best_crop_score = suggest_best_crop(df, rf_pipe, sample_input)
        print("\nML Model Suggestion and yield prediction:")
        print(f" - ML Model Suggestion: {best_crop} (score {best_crop_score:.2f})")
        print(f" - Expected yield for given inputs: {pred:.2f} t/ha")
        kg_ac, units = yield_to_kg_ac_and_units(pred)
        print(f" - In kg per acre: {kg_ac:,.0f} kg/ac")
        print(f" - In 20 kg units: {units:,.0f} units")

        groq_response = get_groq_crop_recommendation(district, area_hectares, capital)
        print("\n🌱 Another Suggestion (Groq AI):")
        print(f" - {groq_response}")

    

    # Recommendations (applies in both cases)
    print("\n💡 Recommendations : \n")
    recommendations = generate_recommendations(sample_input, pred, capital)
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            print(f" {idx}. {rec}")
    else:
        print(" ✅ No additional recommendations. Conditions look good!")



    
