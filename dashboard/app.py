
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Energy Management System",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    energy_model = joblib.load(
        "models/improved_energy_prediction_model.pkl"
    )

    theft_model = joblib.load(
        "models/final_electricity_theft_model.pkl"
    )

    return energy_model, theft_model


energy_model, theft_model = load_models()


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("⚡ Smart Energy Management System")

st.markdown(
    "### Energy Consumption Prediction & Electricity Theft Detection"
)

st.write(
    "An AI-powered system for monitoring electricity consumption "
    "and identifying potential electricity theft risks."
)

st.divider()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:15px 5px;">
        <h1 style="color:#00D4FF; font-size:28px;">⚡ SEMS</h1>
        <p style="color:#94A3B8; font-size:12px;">
            SMART ENERGY MANAGEMENT
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.markdown(
    "### 🧭 Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Energy Consumption Prediction",
        "Electricity Theft Detection"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="padding:10px; color:#94A3B8; font-size:12px;">
        <p>🤖 AI Powered System</p>
        <p>⚡ Energy Analytics</p>
        <p>🔍 Theft Detection</p>
        <p style="margin-top:20px;">
            Version 1.0.0
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



# --------------------------------------------------
# LOAD DASHBOARD DATA
# --------------------------------------------------

@st.cache_data
def load_dashboard_data():

    return pd.read_csv(
        "data/processed/sgcc_features.csv"
    )


@st.cache_data
def load_energy_data():

    data = pd.read_csv(
        "data/processed/energy_data.csv"
    )

    data["Datetime"] = pd.to_datetime(
        data["Datetime"]
    )

    return data


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.subheader("📊 Energy Management Analytics")

    df = load_dashboard_data()

    total_consumers = len(df)

    theft_cases = int(
        (df["FLAG"] == 1).sum()
    )

    normal_cases = int(
        (df["FLAG"] == 0).sum()
    )

    theft_percentage = (
        theft_cases / total_consumers
    ) * 100

    # Dashboard metrics

    
    # Dashboard KPI Cards

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #151F32, #1B2B45);
            border: 1px solid #263B58;
            padding: 20px 15px;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8;
        }

        div[data-testid="stMetricValue"] {
            color: #00D4FF;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total Consumers", f"{total_consumers:,}")
    col2.metric("✅ Normal Consumers", f"{normal_cases:,}")
    col3.metric("⚠️ Theft-labelled Consumers", f"{theft_cases:,}")
    col4.metric("📊 Theft-labelled Share", f"{theft_percentage:.2f}%")

    # Consumer label distribution

    st.subheader("📊 Consumer Label Distribution")

    label_data = pd.DataFrame({
        "Category": ["Normal", "Theft-labelled"],
        "Consumers": [normal_cases, theft_cases]
    })

    st.bar_chart(
        label_data.set_index("Category"),
        color="#00D4FF",
        height=300
    )

    # Energy consumption overview
    

    # Energy consumption trend
    energy_df = load_energy_data()

    st.subheader("📈 Energy Consumption Trend")

    chart_data = energy_df.set_index("Datetime")

    daily_consumption = (
        chart_data["Global_active_power"]
        .resample("D")
        .mean()
    )

    st.line_chart(
        daily_consumption,
        color="#00D4FF",
        height=350
    )

    # Recent 7 Days Summary

    st.subheader("📅 Recent 7 Days Energy Summary")

    recent_7_days = daily_consumption.dropna().tail(7)

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Daily Power",
        f"{recent_7_days.mean():.3f} kW"
    )

    col2.metric(
        "Highest Daily Power",
        f"{recent_7_days.max():.3f} kW"
    )

    st.dataframe(
        recent_7_days.rename("Average Power (kW)").to_frame(),
        use_container_width=True
    )

    st.divider()


    # Model performance

    st.subheader("🤖 Model Performance")

    st.markdown(
        """
        <style>
        .model-card {
            background: linear-gradient(135deg, #151F32, #1B2B45);
            border: 1px solid #263B58;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 15px;
        }

        .model-title {
            color: #00D4FF;
            font-size: 19px;
            font-weight: bold;
            margin-bottom: 15px;
        }

        .model-label {
            color: #94A3B8;
            font-size: 13px;
        }

        .model-value {
            color: #F1F5F9;
            font-size: 25px;
            font-weight: bold;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="model-card">
                <div class="model-title">⚡ Energy Prediction</div>
                <div class="model-label">R² Score</div>
                <div class="model-value">58.90%</div>
                <div class="model-label">RMSE</div>
                <div class="model-value">0.4861</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="model-card">
                <div class="model-title">🔍 Electricity Theft Detection</div>
                <div class="model-label">Accuracy</div>
                <div class="model-value">72.04%</div>
                <div class="model-label">Recall</div>
                <div class="model-value">63.43%</div>
                <div class="model-label">Precision</div>
                <div class="model-value">17.87%</div>
                <div class="model-label">F1 Score</div>
                <div class="model-value">27.88%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Note: Energy consumption and SGCC theft detection "
        "are based on two different datasets."
    )


# --------------------------------------------------
# ENERGY CONSUMPTION PREDICTION PAGE
# --------------------------------------------------

elif page == "Energy Consumption Prediction":

    st.subheader("⚡ Energy Consumption Prediction")

    st.write(
        "Enter recent electricity consumption details "
        "to predict the next hour's energy consumption."
    )

    st.info(
        "Enter all consumption values in kW."
    )

    with st.form("energy_prediction_form"):

        st.markdown("### Recent Consumption")

        col1, col2, col3 = st.columns(3)

        with col1:

            current_power = st.number_input(
                "Current hour consumption",
                min_value=0.0,
                value=1.0
            )

            lag_1 = st.number_input(
                "Previous hour",
                min_value=0.0,
                value=1.0
            )

            lag_2 = st.number_input(
                "Consumption 2 hours ago",
                min_value=0.0,
                value=1.0
            )

            lag_3 = st.number_input(
                "Consumption 3 hours ago",
                min_value=0.0,
                value=1.0
            )

        with col2:

            lag_24 = st.number_input(
                "Consumption 24 hours ago",
                min_value=0.0,
                value=1.0
            )

            rolling_3h = st.number_input(
                "3-hour rolling average",
                min_value=0.0,
                value=1.0
            )

            rolling_24h = st.number_input(
                "24-hour rolling average",
                min_value=0.0,
                value=1.0
            )

            rolling_7d = st.number_input(
                "7-day rolling average",
                min_value=0.0,
                value=1.0
            )

        with col3:

            hour = st.number_input(
                "Current hour (0-23)",
                min_value=0,
                max_value=23,
                value=12
            )

            day_of_week = st.number_input(
                "Day of week (Monday=0, Sunday=6)",
                min_value=0,
                max_value=6,
                value=0
            )

            month = st.number_input(
                "Month (1-12)",
                min_value=1,
                max_value=12,
                value=1
            )

        submitted = st.form_submit_button(
            "Predict Next Hour Consumption"
        )

    if submitted:

        input_data = pd.DataFrame([{

            "current_power": current_power,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_24": lag_24,
            "rolling_3h": rolling_3h,
            "rolling_24h": rolling_24h,
            "rolling_7d": rolling_7d,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,

            "hour_sin": np.sin(
                2 * np.pi * hour / 24
            ),

            "hour_cos": np.cos(
                2 * np.pi * hour / 24
            ),

            "day_sin": np.sin(
                2 * np.pi * day_of_week / 7
            ),

            "day_cos": np.cos(
                2 * np.pi * day_of_week / 7
            )

        }])

        prediction = energy_model.predict(
            input_data
        )[0]

        st.divider()

        st.success(
            "Energy prediction completed!"
        )

        st.metric(
            "Predicted Next-Hour Consumption",
            f"{max(0, prediction):.3f} kW"
        )

        st.caption(
            "Prediction generated using the trained "
            "Random Forest model."
        )


# --------------------------------------------------
# ELECTRICITY THEFT DETECTION PAGE
# --------------------------------------------------

elif page == "Electricity Theft Detection":

    st.subheader(
        "🔍 Consumer-wise Electricity Theft Detection"
    )

    st.write(
        "Select a consumer ID to automatically analyze "
        "their historical electricity consumption."
    )

    consumer_df = load_dashboard_data()

    feature_cols = [
        "average_consumption",
        "total_consumption",
        "std_consumption",
        "min_consumption",
        "max_consumption",
        "median_consumption",
        "zero_consumption_days",
        "high_consumption_days",
        "consumption_variation"
    ]

    st.divider()

    # Individual consumer analysis

    st.markdown("### Select Consumer")

    consumer_ids = (
        consumer_df["CONS_NO"]
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "Choose Consumer ID",
        consumer_ids
    )

    if st.button("Analyze Consumer"):

        consumer_row = consumer_df[
            consumer_df["CONS_NO"].astype(str)
            == selected_id
        ]

        if not consumer_row.empty:

            input_data = consumer_row[feature_cols]

            probability = theft_model.predict_proba(
                input_data
            )[0][1]

            prediction = int(
                probability >= 0.50
            )

            st.divider()

            st.subheader(
                "Consumer Analysis Results"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Potential Theft Risk",
                    f"{probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Consumer ID",
                    selected_id
                )

            if prediction == 1:

                st.warning(
                    "Potential theft risk detected. "
                    "Further investigation is recommended."
                )

            else:

                st.success(
                    "Consumer classified as Normal by the model."
                )

            st.markdown(
                "### Consumption Statistics"
            )

            st.dataframe(
                consumer_row[feature_cols].T.rename(
                    columns={
                        consumer_row.index[0]: "Value"
                    }
                ),
                use_container_width=True
            )

            st.caption(
                "Experimental ML result based on historical "
                "SGCC data. This is not confirmation of "
                "actual electricity theft."
            )

        else:

            st.error(
                "Consumer ID not found."
            )

    # Bulk consumer analysis

    st.divider()

    st.subheader(
        "📂 Bulk Consumer Analysis"
    )

    st.write(
        "Upload a CSV file containing multiple consumers' "
        "electricity consumption features."
    )

    uploaded_file = st.file_uploader(
        "Upload Consumer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            bulk_df = pd.read_csv(
                uploaded_file
            )

            missing_cols = [
                col for col in feature_cols
                if col not in bulk_df.columns
            ]

            if missing_cols:

                st.error(
                    f"Missing required columns: {missing_cols}"
                )

            elif bulk_df.empty:

                st.warning(
                    "The uploaded CSV file is empty."
                )

            else:

                st.success(
                    f"Uploaded {len(bulk_df)} consumer records."
                )

                if st.button("Analyze All Consumers"):

                    input_features = bulk_df[
                        feature_cols
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    if input_features.isnull().any().any():

                        st.error(
                            "CSV contains missing or non-numeric "
                            "feature values. Please clean the data."
                        )

                    else:

                        probabilities = (
                            theft_model.predict_proba(
                                input_features
                            )[:, 1]
                        )

                        predictions = (
                            probabilities >= 0.50
                        ).astype(int)

                        results_df = bulk_df.copy()

                        results_df[
                            "Theft_Risk_Percentage"
                        ] = (
                            probabilities * 100
                        ).round(2)

                        results_df["Prediction"] = [
                            "Potential Theft"
                            if p == 1
                            else "Normal"
                            for p in predictions
                        ]

                        st.subheader(
                            "Bulk Analysis Results"
                        )

                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )

                        theft_count = int(
                            predictions.sum()
                        )

                        normal_count = (
                            len(predictions) - theft_count
                        )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Potential Theft Cases",
                            theft_count
                        )

                        col2.metric(
                            "Normal Cases",
                            normal_count
                        )

                        csv_data = results_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Analysis Report",
                            data=csv_data,
                            file_name="electricity_theft_analysis.csv",
                            mime="text/csv"
                        )

        except Exception as e:

            st.error(
                f"Error processing CSV: {e}"
            )
