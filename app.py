import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "notebook" / "customer_churn_ann.pkl"
SCALER_PATH = BASE_DIR / "notebook" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "notebook" / "feature_columns.pkl"


# ============================================================
# 3. LOAD MODEL AND PREPROCESSING FILES
# ============================================================

try:

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

except FileNotFoundError as e:

    st.error("Required model file was not found.")
    st.write("Please check that these files exist inside the notebook folder:")

    st.code(
        """
notebook/
│
├── customer_churn_ann.pkl
├── scaler.pkl
└── feature_columns.pkl
        """
    )

    st.error(f"Missing file: {e}")
    st.stop()


# ============================================================
# 4. TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")
st.markdown(
    "### Artificial Neural Network (ANN) based customer churn prediction"
)

st.write(
    "Enter the customer's information below to predict whether "
    "the customer is likely to churn."
)


# ============================================================
# 5. CUSTOMER INFORMATION
# ============================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )


with col2:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


with col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=monthly_charges * tenure
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


# ============================================================
# 6. SERVICES
# ============================================================

st.subheader("🌐 Internet & Services")

col1, col2, col3 = st.columns(3)


with col1:

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# ============================================================
# 7. PAYMENT METHOD
# ============================================================

st.subheader("💳 Payment Information")

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# ============================================================
# 8. CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe():

    data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,

        "tenure": tenure,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges,

        "gender_Male": 1 if gender == "Male" else 0,

        "Partner_Yes": 1 if partner == "Yes" else 0,

        "Dependents_Yes": 1 if dependents == "Yes" else 0,

        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

        "MultipleLines_No phone service":
            1 if multiple_lines == "No phone service" else 0,

        "MultipleLines_Yes":
            1 if multiple_lines == "Yes" else 0,

        "InternetService_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "InternetService_No":
            1 if internet_service == "No" else 0,

        "OnlineSecurity_No internet service":
            1 if online_security == "No internet service" else 0,

        "OnlineSecurity_Yes":
            1 if online_security == "Yes" else 0,

        "OnlineBackup_No internet service":
            1 if online_backup == "No internet service" else 0,

        "OnlineBackup_Yes":
            1 if online_backup == "Yes" else 0,

        "DeviceProtection_No internet service":
            1 if device_protection == "No internet service" else 0,

        "DeviceProtection_Yes":
            1 if device_protection == "Yes" else 0,

        "TechSupport_No internet service":
            1 if tech_support == "No internet service" else 0,

        "TechSupport_Yes":
            1 if tech_support == "Yes" else 0,

        "StreamingTV_No internet service":
            1 if streaming_tv == "No internet service" else 0,

        "StreamingTV_Yes":
            1 if streaming_tv == "Yes" else 0,

        "StreamingMovies_No internet service":
            1 if streaming_movies == "No internet service" else 0,

        "StreamingMovies_Yes":
            1 if streaming_movies == "Yes" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "PaperlessBilling_Yes":
            1 if paperless_billing == "Yes" else 0,

        "PaymentMethod_Credit card (automatic)":
            1 if payment_method == "Credit card (automatic)" else 0,

        "PaymentMethod_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "PaymentMethod_Mailed check":
            1 if payment_method == "Mailed check" else 0,
    }

    df = pd.DataFrame([data])

    return df


# ============================================================
# 9. PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    try:

        # Create input
        input_df = create_input_dataframe()

        # Make sure columns are in exactly the same order
        # as used during model training
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Get probability
        prediction_probability = model.predict(
            input_scaled,
            verbose=0
        )

        # Extract churn probability
        churn_probability = float(
            np.asarray(prediction_probability).ravel()[0]
        )

        # Convert probability into class
        prediction = 1 if churn_probability >= 0.5 else 0


        # ====================================================
        # 10. DISPLAY RESULT
        # ====================================================

        st.subheader("📈 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to CHURN"
                )

            else:

                st.success(
                    "✅ Customer is likely to STAY"
                )


        with col2:

            st.metric(
                "Churn Probability",
                f"{churn_probability:.2%}"
            )


        # ====================================================
        # 11. PROBABILITY BAR
        # ====================================================

        st.write("### Churn Probability")

        st.progress(
            min(max(churn_probability, 0.0), 1.0)
        )


        # ====================================================
        # 12. RECOMMENDATION
        # ====================================================

        if prediction == 1:

            st.warning(
                "💡 Recommendation: "
                "This customer has a high probability of leaving. "
                "Consider offering personalized discounts, "
                "loyalty benefits, or a suitable contract upgrade."
            )

        else:

            st.info(
                "💡 Recommendation: "
                "This customer currently has a lower probability "
                "of churn. Continue providing good service and "
                "customer engagement."
            )


        # ====================================================
        # 13. INPUT SUMMARY
        # ====================================================

        with st.expander("🔎 View Processed Input"):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)