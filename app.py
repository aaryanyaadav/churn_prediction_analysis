import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from preprocessing import feature_engineering
from predict import preprocess_data, model


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "Upload customer data to predict churn probability "
    "and identify high-risk customers."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # =====================================================
    # READ CSV
    # =====================================================

    df = pd.read_csv(uploaded_file)

    st.success(
        f"Successfully uploaded {len(df):,} customers."
    )


    # =====================================================
    # SHOW RAW DATA
    # =====================================================

    with st.expander("View Uploaded Data"):

        st.dataframe(
            df,
            use_container_width=True
        )


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df_model = feature_engineering(df)


    # =====================================================
    # PREPROCESSING
    # =====================================================

    try:

        X = preprocess_data(df_model)

    except Exception as e:

        st.error(
            f"Error during preprocessing: {e}"
        )

        st.stop()


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        churn_probability = model.predict_proba(X)[:, 1]

        prediction = (
            churn_probability >= 0.50
        ).astype(int)

    except Exception as e:

        st.error(
            f"Error during prediction: {e}"
        )

        st.stop()


    # =====================================================
    # ADD RESULTS TO DATAFRAME
    # =====================================================

    results = df.copy()

    results["Churn Probability"] = churn_probability

    results["Prediction"] = prediction


    # =====================================================
    # RISK LEVEL
    # =====================================================

    def risk_level(probability):

        if probability >= 0.75:
            return "High"

        elif probability >= 0.50:
            return "Medium"

        else:
            return "Low"


    results["Risk"] = results[
        "Churn Probability"
    ].apply(risk_level)


    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    total_customers = len(results)

    churned_customers = (
        results["Prediction"] == 1
    ).sum()

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    high_risk_customers = (
        results["Risk"] == "High"
    ).sum()

    high_risk_rate = (
        high_risk_customers /
        total_customers
    ) * 100


    # =====================================================
    # KPI DASHBOARD
    # =====================================================

    st.subheader("📌 Customer Overview")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "Predicted Churn",
            f"{churned_customers:,}"
        )


    with col3:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )


    with col4:

        st.metric(
            "High Risk Customers",
            f"{high_risk_customers:,}"
        )


    # =====================================================
    # SECOND ROW
    # =====================================================

    col5, col6 = st.columns(2)


    with col5:

        st.metric(
            "High Risk Rate",
            f"{high_risk_rate:.2f}%"
        )


    with col6:

        average_probability = (
            results["Churn Probability"].mean()
            * 100
        )

        st.metric(
            "Average Churn Probability",
            f"{average_probability:.2f}%"
        )


    # =====================================================
    # CHARTS
    # =====================================================

    st.subheader("📈 Churn Analysis")


    chart_col1, chart_col2 = st.columns(2)


    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    with chart_col1:

        churn_counts = results[
            "Prediction"
        ].value_counts()

        labels = [
            "Stayed",
            "Churned"
        ]

        values = [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]

        fig, ax = plt.subplots()

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Predicted Churn Distribution"
        )

        st.pyplot(fig)


    # -----------------------------------------------------
    # RISK DISTRIBUTION
    # -----------------------------------------------------

    with chart_col2:

        risk_counts = results[
            "Risk"
        ].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            risk_counts.index,
            risk_counts.values
        )

        ax.set_title(
            "Customer Risk Distribution"
        )

        ax.set_xlabel("Risk Level")

        ax.set_ylabel("Number of Customers")

        st.pyplot(fig)


    # =====================================================
    # HIGH RISK CUSTOMERS
    # =====================================================

    st.subheader("🚨 High-Risk Customers")


    high_risk = results[
        results["Risk"] == "High"
    ].copy()


    high_risk = high_risk.sort_values(
        by="Churn Probability",
        ascending=False
    )


    display_columns = []


    # Keep columns only if they exist
    for column in [
        "CustomerId",
        "Surname",
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "Churn Probability",
        "Risk"
    ]:

        if column in high_risk.columns:

            display_columns.append(column)


    st.dataframe(
        high_risk[display_columns],
        use_container_width=True
    )


    # =====================================================
    # DOWNLOAD RESULTS
    # =====================================================

    csv = results.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )


    # =====================================================
    # INDIVIDUAL CUSTOMER PREDICTION
    # =====================================================

    st.subheader(
        "🔍 Individual Customer Analysis"
    )


    if "CustomerId" in results.columns:

        customer_ids = results[
            "CustomerId"
        ].astype(str).tolist()


        selected_customer = st.selectbox(
            "Select Customer",
            customer_ids
        )


        customer_index = results[
            results["CustomerId"].astype(str)
            == selected_customer
        ].index[0]


        customer = results.loc[
            customer_index
        ]


        # -------------------------------------------------
        # CUSTOMER DETAILS
        # -------------------------------------------------

        st.write("### Customer Information")


        info_columns = []

        for column in [
            "CustomerId",
            "Surname",
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]:

            if column in results.columns:

                info_columns.append(column)


        st.dataframe(
            customer[info_columns]
            .to_frame()
            .T,
            use_container_width=True
        )


        # -------------------------------------------------
        # CUSTOMER PREDICTION
        # -------------------------------------------------

        probability = customer[
            "Churn Probability"
        ]


        prediction_value = customer[
            "Prediction"
        ]


        risk = customer[
            "Risk"
        ]


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Prediction",
                "Will Churn"
                if prediction_value == 1
                else "Will Stay"
            )


        with col3:

            st.metric(
                "Risk Level",
                risk
            )


        # -------------------------------------------------
        # SHAP ANALYSIS
        # -------------------------------------------------

        st.write(
            "### 🔎 Prediction Explanation"
        )

        try:

            import shap
            import numpy as np

            # ---------------------------------------------
            # Create TreeExplainer
            # IMPORTANT:
            # Do NOT pass X as background data.
            # ---------------------------------------------

            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent"
            )

            # ---------------------------------------------
            # Explain only the selected customer
            # ---------------------------------------------

            customer_X = X.loc[
                [customer_index]
            ]

            customer_shap = explainer(
                customer_X
            )

            values = customer_shap.values

            # ---------------------------------------------
            # Handle possible multi-output shape
            # ---------------------------------------------

            if len(values.shape) == 3:

                # Binary classification
                values = values[0, :, 1]

            else:

                values = values[0]


            # ---------------------------------------------
            # Feature names
            # ---------------------------------------------

            feature_names = list(
                X.columns
            )


            # ---------------------------------------------
            # Create SHAP dataframe
            # ---------------------------------------------

            shap_df = pd.DataFrame({

                "Feature": feature_names,

                "SHAP Value": values

            })


            shap_df["Absolute SHAP"] = (
                shap_df["SHAP Value"].abs()
            )


            shap_df = shap_df.sort_values(
                "Absolute SHAP",
                ascending=False
            )


            # ---------------------------------------------
            # Strongest factor
            # ---------------------------------------------

            strongest_feature = shap_df.iloc[0]

            feature_name = (
                strongest_feature["Feature"]
            )

            shap_value = (
                strongest_feature["SHAP Value"]
            )


            if shap_value > 0:

                direction = (
                    "increased the model's "
                    "churn prediction"
                )

            else:

                direction = (
                    "decreased the model's "
                    "churn prediction"
                )


            st.info(
                f"**Strongest factor:** "
                f"{feature_name} "
                f"({direction})"
            )


            # ---------------------------------------------
            # TOP 5 FEATURES
            # ---------------------------------------------

            top_5 = shap_df.head(5)


            fig, ax = plt.subplots()


            ax.barh(
                top_5["Feature"][::-1],
                top_5["SHAP Value"][::-1]
            )


            ax.set_xlabel(
                "SHAP Value"
            )


            ax.set_title(
                "Top 5 Factors Affecting This Customer"
            )


            plt.tight_layout()


            st.pyplot(fig)


            # ---------------------------------------------
            # SHAP TABLE
            # ---------------------------------------------

            st.dataframe(
                top_5[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ],
                use_container_width=True
            )


        except Exception as e:

            st.warning(
                f"SHAP explanation unavailable: {e}"
            )
    else:

        st.warning(
            "CustomerId column not found."
        )