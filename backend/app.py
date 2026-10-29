import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the serialized pipeline (preprocessing + model)
model = joblib.load("superkart_model.joblib")

# Feature columns expected by the model, in training order
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]


@superkart_api.get("/")
def home():
    """Health-check endpoint."""
    return "Welcome to the SuperKart Sales Forecasting API!"


@superkart_api.post("/v1/predict")
def predict_sales():
    """Online inference: predict sales for a single product-store record (JSON body)."""
    record = request.get_json()
    missing = [f for f in FEATURES if f not in record]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_df = pd.DataFrame([{f: record[f] for f in FEATURES}])
    prediction = float(model.predict(input_df)[0])
    return jsonify({"Sales": round(prediction, 2)})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Batch inference: predict sales for every row of an uploaded CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in the request"}), 400

    input_df = pd.read_csv(request.files["file"])
    missing = [f for f in FEATURES if f not in input_df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    predictions = model.predict(input_df[FEATURES]).round(2)
    # Key = row index, value = predicted sales
    return jsonify({str(i): float(p) for i, p in zip(input_df.index, predictions)})


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
