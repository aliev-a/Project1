import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="House Price App", layout="wide")

st.title("🏠 California House Price Prediction App")
st.markdown("Predict house prices using Machine Learning")

# ---------------------------
# LOAD DATA (with caching)
# ---------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("⚙️ Input Parameters")

def user_input_features():
    data = {}
    for col in X.columns:
        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# ---------------------------
# MODEL (cached)
# ---------------------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model()

prediction = model.predict(df)[0]

# ---------------------------
# MAIN LAYOUT
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Data")
    st.write(df)

with col2:
    st.subheader("💰 Prediction")

    price_usd = prediction * 100000
    st.success(f"Estimated price: ${price_usd:,.2f}")

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------
st.subheader("📊 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig1, ax1 = plt.subplots()
ax1.barh(importance["Feature"], importance["Importance"])
ax1.invert_yaxis()
st.pyplot(fig1)

# ---------------------------
# DISTRIBUTION OF TARGET
# ---------------------------
st.subheader("📈 Price Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(y, bins=30)
ax2.set_title("Distribution of House Prices")
st.pyplot(fig2)

# ---------------------------
# CORRELATION HEATMAP
# ---------------------------
st.subheader("🔗 Correlation Matrix")

corr = X.corr()

fig3, ax3 = plt.subplots()
cax = ax3.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig3.colorbar(cax)

st.pyplot(fig3)

# ---------------------------
# RAW DATA VIEW
# ---------------------------
if st.checkbox("Show raw dataset"):
    st.subheader("Dataset")
    st.write(X.head())

# ---------------------------
# MODEL INSIGHT
# ---------------------------
st.subheader("🧠 Model Info")

st.write(f"""
- Model: RandomForestRegressor  
- Features: {X.shape[1]}  
- Dataset size: {X.shape[0]} rows  
""")
