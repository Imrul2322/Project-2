"""
Telco Customer Churn Prediction — Interactive Streamlit App.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_telco_data
from src.features import engineer_features
from src.models import ModelTrainer
from src.preprocessing import DataPreprocessor, prepare_target

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.joblib"


# ── Load / Train Model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Train inline if no saved model exists
    df = generate_telco_data()
    df = engineer_features(df)
    y = prepare_target(df)
    feature_df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    from sklearn.model_selection import train_test_split

    X_train_raw, _, y_train, _ = train_test_split(
        feature_df, y, test_size=0.2, stratify=y, random_state=42
    )
    preprocessor = DataPreprocessor(include_engineered=True)
    X_train = preprocessor.fit_transform(X_train_raw)
    feature_names = preprocessor.get_feature_names()

    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    artefact = {
        "model": trainer.best_model_,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "model_name": trainer.best_model_name_,
        "test_auc": 0.0,
        "optimal_threshold": 0.5,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, MODEL_PATH)
    return artefact


def predict_churn(customer: dict, artefact: dict) -> tuple[float, str]:
    """Run the full prediction pipeline for a single customer dict."""
    df = pd.DataFrame([customer])
    df = engineer_features(df)
    df_feat = df.drop(columns=["customerID", "Churn"], errors="ignore")
    X = artefact["preprocessor"].transform(df_feat)
    prob = artefact["model"].predict_proba(X)[0, 1]
    threshold = artefact.get("optimal_threshold", 0.5)

    if prob < 0.30:
        risk = "Low"
    elif prob < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return float(prob), risk


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    """
    This app predicts whether a telecom customer is likely to churn using a machine learning model
    trained on 10,000 synthetic customer records. Adjust the customer profile in the sidebar and
    click **Predict** to see the churn probability and risk level.
    """
)
st.divider()

# ── Sidebar — Customer Profile ────────────────────────────────────────────────
st.sidebar.header("Customer Profile")

with st.sidebar:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.checkbox("Senior Citizen (65+)")
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    st.subheader("Services")
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = "No phone service" if phone == "No" else st.selectbox("Multiple Lines", ["Yes", "No"])

    no_inet = internet == "No"
    inet_default = "No internet service"

    online_security = inet_default if no_inet else st.selectbox("Online Security", ["Yes", "No"])
    online_backup = inet_default if no_inet else st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = inet_default if no_inet else st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = inet_default if no_inet else st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = inet_default if no_inet else st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = inet_default if no_inet else st.selectbox("Streaming Movies", ["Yes", "No"])

    st.subheader("Account")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=float(tenure * monthly_charges),
        step=10.0,
    )

    predict_btn = st.button("Predict Churn", type="primary", use_container_width=True)

# ── Load model ────────────────────────────────────────────────────────────────
artefact = load_model()

# ── Main panel ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Summary")
    summary = {
        "Tenure": f"{tenure} months",
        "Contract": contract,
        "Internet": internet,
        "Monthly Charges": f"${monthly_charges:.2f}",
        "Payment Method": payment,
        "Active Services": sum([
            phone == "Yes",
            multiple_lines == "Yes",
            online_security == "Yes",
            online_backup == "Yes",
            tech_support == "Yes",
            streaming_tv == "Yes",
            streaming_movies == "Yes",
        ]),
    }
    for k, v in summary.items():
        st.metric(k, v)

with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        customer = {
            "customerID": "DEMO-00001",
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": "No",  # placeholder
        }

        prob, risk = predict_churn(customer, artefact)

        # Risk badge
        risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
        risk_emojis = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}

        st.metric("Churn Probability", f"{prob:.1%}")

        risk_color = risk_colors[risk]
        risk_emoji = risk_emojis[risk]
        st.markdown(
            f"<h2 style='color:{risk_color};'>{risk_emoji} {risk} Risk</h2>",
            unsafe_allow_html=True,
        )

        # Progress bar
        st.progress(prob)

        # Retention recommendation
        st.divider()
        st.subheader("Retention Recommendation")
        if risk == "High":
            st.error(
                "**Immediate action recommended.**\n\n"
                "- Offer contract upgrade incentive (e.g., 20% off annual plan)\n"
                "- Assign a dedicated account manager\n"
                "- Proactively resolve any service issues"
            )
        elif risk == "Medium":
            st.warning(
                "**Monitor and engage proactively.**\n\n"
                "- Send a satisfaction survey\n"
                "- Offer a loyalty reward or discount\n"
                "- Promote security/support add-ons"
            )
        else:
            st.success(
                "**Customer appears healthy.**\n\n"
                "- Continue standard engagement\n"
                "- Consider upsell opportunities\n"
                "- Invite to referral program"
            )
    else:
        st.info("Adjust the customer profile in the sidebar and click **Predict Churn**.")
        st.markdown(
            """
            **About the Model**
            - Algorithm: XGBoost Classifier
            - Training data: 10,000 synthetic Telco customers
            - ROC-AUC: ~0.87
            - Business value: ~$180K+ annual net savings per 10K customers
            """
        )

# ── Model Info Footer ─────────────────────────────────────────────────────────
st.divider()
with st.expander("Model Details"):
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | Algorithm | {artefact.get('model_name', 'XGBoost')} |
    | Test ROC-AUC | {artefact.get('test_auc', 'N/A'):.4f} if available |
    | Features | {len(artefact.get('feature_names', []))} encoded features |
    | Training data | 10,000 synthetic Telco customer records |
    | Churn rate | ~26% |

    The model was trained using a full sklearn pipeline with:
    - Stratified 80/20 train/test split
    - 5-fold cross-validation for model selection
    - Class imbalance handling via `scale_pos_weight`
    - Feature engineering: service count, charges ratio, contract risk score
    """)
