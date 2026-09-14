import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="💊")


@st.cache_resource
def load_artifacts():
    """Load the model + preprocessing objects saved by the training notebook.

    Expects these four files to sit next to app.py:
      - model_metadata.pkl   (tells us which file/format the winning model uses)
      - scaler.pkl           (the StandardScaler fit on the TRAINING data)
      - model_columns.pkl    (exact training column order)
      - best_insurance_model.json  OR  best_insurance_model.pkl
    """
    meta = joblib.load("model_metadata.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")

    if meta["is_xgboost"]:
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(meta["model_filename"])
    else:
        model = joblib.load(meta["model_filename"])

    return model, scaler, columns, meta


model, scaler, columns, meta = load_artifacts()

st.title("💊 Medical Insurance Cost Predictor")
st.caption(
    f"Powered by a **{meta['best_name']}** model trained on the Kaggle "
    "Medical Cost Personal Datasets."
)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Number of children", 0, 5, 0)
with col2:
    sex = st.radio("Sex", ["Male", "Female"])
    smoker = st.radio("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

if st.button("Predict my cost", type="primary"):
    # Build one row in the exact same shape/encoding the model was trained on.
    # "Northeast" was the dropped baseline category during one-hot encoding,
    # so it's represented as all three region_* columns being 0.
    row = {
        "age": age,
        "sex": 0 if sex == "Male" else 1,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "Yes" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
    }
    new_data = pd.DataFrame([row])[columns]  # reorder to match training columns
    new_data[["age", "bmi", "children"]] = scaler.transform(
        new_data[["age", "bmi", "children"]]
    )

    prediction = model.predict(new_data)[0]
    st.success(f"Estimated annual insurance cost: **${prediction:,.2f}**")

st.caption("Educational project — not a real insurance quote.")
