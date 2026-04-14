import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🏠 House Price AI", layout="wide")

st.title("🏠 AI House Price Predictor")
st.markdown("### Smart real estate valuation powered by Machine Learning")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=150)
    model.fit(X, y)
    return model

model = train_model()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Customize House")

def user_input():
    data = {}
    for col in X.columns:
        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    return pd.DataFrame(data, index=[0])

df = user_input()

prediction = model.predict(df)[0]
price = prediction * 100000

# ---------------------------
# TOP METRICS
# ---------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Predicted Price", f"${price:,.0f}")
col2.metric("📊 Dataset Avg", f"${y.mean()*100000:,.0f}")
col3.metric("📈 Difference", f"${price - y.mean()*100000:,.0f}")

st.divider()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "📊 Analytics", "🗺️ Map", "🧠 Model"])

# ---------------------------
# TAB 1 INPUT
# ---------------------------
with tab1:
    st.subheader("Your House Parameters")
    st.dataframe(df)

# ---------------------------
# TAB 2 ANALYTICS
# ---------------------------
with tab2:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(y, bins=30)
        ax.axvline(prediction, linestyle="--")
        st.pyplot(fig)

    with colB:
        st.subheader("Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance["Feature"], importance["Importance"])
        st.pyplot(fig2)

# ---------------------------
# TAB 3 MAP
# ---------------------------
with tab3:
    st.subheader("Geographic Visualization")

    map_data = X.copy()
    map_data["price"] = y

    st.map(map_data.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon"
    }))

# ---------------------------
# TAB 4 MODEL INFO
# ---------------------------
with tab4:
    st.subheader("Model Explanation")

    st.write("""
    This model uses **Random Forest Regression**.

    ✔ Combines multiple decision trees  
    ✔ Captures complex relationships  
    ✔ Works well for real estate data  

    ### Key Factors:
    - Median Income (most important)
    - Location (Latitude/Longitude)
    - Average Rooms
    """)

    st.subheader("Correlation Matrix")

    corr = X.corr()
    fig3, ax3 = plt.subplots()
    cax = ax3.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig3.colorbar(cax)

    st.pyplot(fig3)

# ---------------------------
# BONUS: COMPARE WITH DATASET
# ---------------------------
st.divider()
st.subheader("📊 Compare Your House to Dataset")

comparison = pd.DataFrame({
    "Your House": df.iloc[0],
    "Average": X.mean()
})

st.bar_chart(comparison)
