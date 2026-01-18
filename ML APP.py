import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="EDA & ML Demo", layout="wide")

# ---------------- TITLE ----------------
st.title("📊 Weather & Health Data: EDA + ML Demo")
st.markdown("**Exploratory Data Analysis and Machine Learning App**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("1️⃣ Data Input")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# ---------------- DATA LOAD ----------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if file:
    df = load_data(file)
    st.success("✅ Dataset Loaded Successfully")

    # ---------------- DATA CLEANING ----------------
    st.sidebar.markdown("---")
    st.sidebar.header("2️⃣ Data Cleaning")
    if st.sidebar.checkbox("Drop Missing Values"):
        df = df.dropna()
    if st.sidebar.checkbox("Fill Missing Values (Mean)"):
        df = df.fillna(df.mean(numeric_only=True))

    # ---------------- SESSION STATE ----------------
    if "active_chart" not in st.session_state:
        st.session_state.active_chart = None

    st.sidebar.markdown("---")
    st.sidebar.header("3️⃣ Chart Controls")

    if st.sidebar.button("📈 Line Chart"):
        st.session_state.active_chart = "Line"
    if st.sidebar.button("📊 Histogram"):
        st.session_state.active_chart = "Hist"
    if st.sidebar.button("📦 Boxplot"):
        st.session_state.active_chart = "Box"
    if st.sidebar.button("🎯 Scatter Plot"):
        st.session_state.active_chart = "Scatter"
    if st.sidebar.button("🔥 Correlation Heatmap"):
        st.session_state.active_chart = "Heatmap"

    # ---------------- DATA PREVIEW ----------------
    with st.expander("🔍 Dataset Preview & Statistics"):
        st.subheader("Preview")
        st.dataframe(df.head())
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

    # ---------------- COLUMN TYPES ----------------
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ---------------- EDA SECTION ----------------
    st.markdown("---")
    st.subheader("📊 Exploratory Data Analysis")

    if st.session_state.active_chart == "Line":
        col_x = st.selectbox("X-axis", all_cols)
        col_y = st.selectbox("Y-axis", num_cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df[col_x], df[col_y], marker="o")
        ax.set_title(f"{col_y} vs {col_x}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif st.session_state.active_chart == "Hist":
        col = st.selectbox("Select Column", num_cols)
        bins = st.slider("Bins", 5, 50, 20)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df[col], bins=bins, edgecolor="black")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    elif st.session_state.active_chart == "Box":
        col = st.selectbox("Select Column", num_cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    elif st.session_state.active_chart == "Scatter":
        col_x = st.selectbox("X-axis", num_cols)
        col_y = st.selectbox("Y-axis", num_cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.regplot(data=df, x=col_x, y=col_y, ax=ax)
        ax.set_title(f"{col_x} vs {col_y}")
        st.pyplot(fig)

    elif st.session_state.active_chart == "Heatmap":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    else:
        st.info("👈 Choose a visualization from the sidebar")

    # ---------------- ML SECTION ----------------
    st.markdown("---")
    st.subheader("🤖 Machine Learning")

    st.markdown("### Model Configuration")

    features = st.multiselect("Select Feature Columns", num_cols)
    target = st.selectbox("Select Target Column", num_cols)

    model_type = st.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest"]
    )

    if st.button("🚀 Train Model"):

        if len(features) == 0:
            st.error("❌ Please select at least one feature.")
        else:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            with st.spinner("Training model..."):

                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(
                        n_estimators=200, random_state=42
                    )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("✅ Training Complete")

            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("R² Score", f"{r2:.2f}")

            # Prediction Plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "--r"
            )
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # Feature Importance
            if model_type == "Random Forest":
                st.subheader("🌟 Feature Importance")
                imp_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                st.bar_chart(imp_df.set_index("Feature"))

# ---------------- NO FILE ----------------
else:
    st.info("👈 Upload a dataset to begin")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Developed by <b>DANIYAL ALVI</b> 🚀 | Streamlit EDA & ML Application"
    "</div>",
    unsafe_allow_html=True
)
