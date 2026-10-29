import io
import os

import pandas as pd
import requests
import streamlit as st

# Backend URL: on the shared Docker network the backend is reachable by its container name
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
st.title("SuperKart Sales Forecast")
st.write("Predict the total sales revenue of a product in a store.")

# ---------------- Online prediction ----------------
st.subheader("Single prediction")

col1, col2 = st.columns(2)
with col1:
    product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.66, step=0.01)
    sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
    mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=117.08, step=0.01)
    id_char = st.selectbox("Product ID Prefix", ["FD", "NC", "DR"],
                           help="FD = Food, NC = Non-consumable, DR = Drinks")
with col2:
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
    city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2",
                                             "Departmental Store", "Food Mart"], index=1)
    store_age = st.number_input("Store Age (years)", min_value=0, max_value=100, value=16, step=1)
    category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

if st.button("Predict"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": area,
        "Product_MRP": mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": id_char,
        "Store_Age_Years": int(store_age),
        "Product_Type_Category": category,
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        resp.raise_for_status()
        st.success(f"Predicted sales: {resp.json()['Sales']:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- Batch prediction ----------------
st.subheader("Batch prediction")
uploaded = st.file_uploader("Upload a CSV with the model's feature columns", type="csv")

if uploaded is not None and st.button("Predict batch"):
    try:
        resp = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                             files={"file": uploaded.getvalue()}, timeout=60)
        resp.raise_for_status()
        result = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        result["Predicted_Sales"] = [resp.json()[str(i)] for i in result.index]
        st.dataframe(result)
        st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
