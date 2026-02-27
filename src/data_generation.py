"""
Synthetic Telco Customer Churn Data Generation.

Generates a realistic 10,000-row dataset that mirrors the IBM Telco Churn
dataset structure, with industry-accurate correlations between features
and the churn target.
"""

import numpy as np
import pandas as pd


def generate_telco_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic telecom customer dataset with realistic churn patterns.

    Key correlations encoded:
    - Month-to-month contracts churn at ~45% vs ~3% for two-year contracts
    - Fiber optic users churn more than DSL users
    - Short tenure (<12 months) customers churn at higher rates
    - Lack of support/security services correlates with higher churn
    - High monthly charges (>$70) slightly elevate churn risk

    Args:
        n_samples: Number of customer records to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with 21 columns matching the IBM Telco Churn schema.
    """
    rng = np.random.default_rng(random_state)

    # ── Demographics ──────────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # ── Contract & Tenure ─────────────────────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.21, 0.24],
    )
    # Tenure distribution conditioned on contract type
    tenure = np.zeros(n_samples, dtype=int)
    mask_m2m = contract == "Month-to-month"
    mask_1yr = contract == "One year"
    mask_2yr = contract == "Two year"
    tenure[mask_m2m] = rng.integers(1, 25, size=mask_m2m.sum())
    tenure[mask_1yr] = rng.integers(12, 50, size=mask_1yr.sum())
    tenure[mask_2yr] = rng.integers(24, 73, size=mask_2yr.sum())

    # ── Phone Services ────────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]),
    )

    # ── Internet Services ─────────────────────────────────────────────────────
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.34, 0.44, 0.22],
    )
    no_internet = internet_service == "No"

    def internet_addon(yes_p: float = 0.45) -> np.ndarray:
        vals = rng.choice(["Yes", "No"], size=n_samples, p=[yes_p, 1 - yes_p])
        vals[no_internet] = "No internet service"
        return vals

    online_security = internet_addon(0.29)
    online_backup = internet_addon(0.34)
    device_protection = internet_addon(0.34)
    tech_support = internet_addon(0.29)
    streaming_tv = internet_addon(0.38)
    streaming_movies = internet_addon(0.39)

    # ── Account Info ──────────────────────────────────────────────────────────
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges: fiber > DSL > no internet
    monthly_charges = np.where(
        internet_service == "Fiber optic",
        rng.uniform(65, 110, size=n_samples),
        np.where(
            internet_service == "DSL",
            rng.uniform(25, 75, size=n_samples),
            rng.uniform(18, 30, size=n_samples),
        ),
    )
    # Add noise for extra services
    service_bonus = (
        (multiple_lines == "Yes").astype(float) * rng.uniform(5, 15, n_samples)
        + (online_security == "Yes").astype(float) * rng.uniform(3, 8, n_samples)
        + (online_backup == "Yes").astype(float) * rng.uniform(3, 8, n_samples)
        + (streaming_tv == "Yes").astype(float) * rng.uniform(4, 10, n_samples)
        + (streaming_movies == "Yes").astype(float) * rng.uniform(4, 10, n_samples)
    )
    monthly_charges = np.clip(monthly_charges + service_bonus, 18, 120).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 5, n_samples)).clip(0).round(2)

    # ── Churn Probability ─────────────────────────────────────────────────────
    # Build a latent churn score from known risk factors
    churn_score = np.zeros(n_samples)

    # Contract type is the strongest predictor
    churn_score += np.where(contract == "Month-to-month", 1.8, 0)
    churn_score += np.where(contract == "One year", 0.5, 0)

    # Short tenure → higher risk
    churn_score += np.where(tenure < 12, 1.0, 0)
    churn_score += np.where(tenure < 6, 0.5, 0)
    churn_score -= np.where(tenure > 36, 0.6, 0)

    # Internet type
    churn_score += np.where(internet_service == "Fiber optic", 0.5, 0)

    # Lack of security / support services
    churn_score += np.where(online_security == "No", 0.4, 0)
    churn_score += np.where(tech_support == "No", 0.4, 0)

    # Payment method
    churn_score += np.where(payment_method == "Electronic check", 0.4, 0)

    # High charges
    churn_score += np.where(monthly_charges > 70, 0.3, 0)

    # Senior citizens churn slightly more
    churn_score += (senior_citizen == 1).astype(float) * 0.2

    # No partner / no dependents → slightly higher
    churn_score += np.where(partner == "No", 0.1, 0)

    # Convert to probability via sigmoid and add noise
    # Offset of 3.6 calibrates overall churn rate to ~26% (industry average)
    churn_prob = 1 / (1 + np.exp(-churn_score + 3.6))
    churn_prob = np.clip(churn_prob + rng.normal(0, 0.05, n_samples), 0.01, 0.99)
    churn = np.where(rng.uniform(size=n_samples) < churn_prob, "Yes", "No")

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_telco_data()
    print(f"Generated {len(df):,} rows | Churn rate: {(df['Churn']=='Yes').mean():.1%}")
    print(df.head())
