# Medical Insurance Cost Predictor

Predicts estimated annual medical insurance charges from age, BMI, smoker
status, and other factors — trained on the Kaggle Medical Cost Personal
Datasets.

**🔗 Try it live:** [Insurance Cost Calculator]([https://YOUR-APP-NAME.streamlit.app](https://xenospro-mby6albdvmq8kk9zddmyej.streamlit.app)

## How it works
- Model: XGBoost regressor (R² ≈ 0.89 on held-out test data)
- Built with pandas, scikit-learn, XGBoost, and Streamlit

**Dataset:** [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle, by mirichoi0218)

## Files
- [`ui_app/app.py`](ui_app/app.py) — the Streamlit app
- [`ui_app/insurance_cost_prediction.ipynb`](ui_app/insurance_cost_prediction.ipynb) — training notebook (EDA, cleaning, model comparison)
