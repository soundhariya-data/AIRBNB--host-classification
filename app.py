import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Airbnb Host Predictor", layout="wide")
st.title("🏠 Airbnb New Host Predictor")
st.markdown("""
### 📌 What This App Does
This application helps identify whether an Airbnb host is **New or Experienced** by analyzing listing, host, and location data using a machine learning–based classification approach.  
It enables **better onboarding, pricing strategy, and risk assessment** for short-term rental platforms.

**Business Value**
- Better host onboarding
- Improved pricing strategies
- Risk and trust assessment

---
### 🧭 How the App Works (Step-by-Step)

**1️⃣ Upload Airbnb CSV**  

**2️⃣ Automatic Data Processing**  

**3️⃣ Model Training & Evaluation**  

**4️⃣ Feature Importance Analysis**  

**5️⃣ Predict Host Type**  

---
""")

st.markdown("Upload Airbnb CSV → Auto Preprocess → Train → Predict")

# ---------------- CSV READER ----------------
@st.cache_data
def read_csv_safe(file):
    for enc in ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1']:
        try:
            return pd.read_csv(file, encoding=enc, low_memory=False)
        except:
            continue
    return pd.read_csv(file, encoding='latin1', errors='replace')

# ---------------- PREPROCESS ----------------
def preprocess_data(df):
    st.info("🔄 Preprocessing data...")

    df = df.drop_duplicates()

    drop_patterns = ['id', 'name', 'description', 'summary', 'url']
    df = df.drop(
        columns=[c for c in df.columns if any(p in c.lower() for p in drop_patterns)],
        errors='ignore'
    )

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    if 'is_new_host' not in df.columns:
        st.error("❌ Column 'is_new_host' not found")
        st.stop()

    X = df.drop(columns=['is_new_host'])
    y = df['is_new_host'].astype(int)

    st.success(f"✅ Dataset ready: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y

# ---------------- FEATURE IMPORTANCE ----------------
def show_feature_importance(model, feature_names):
    st.subheader("📈 Feature Importance")

    coef = model.coef_[0]
    imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.abs(coef)
    }).sort_values("Importance", ascending=False).head(15)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(imp, use_container_width=True)
    with col2:
        st.bar_chart(imp.set_index("Feature"))

# ---------------- MAIN APP ----------------
uploaded_file = st.file_uploader("📁 Upload Airbnb CSV", type="csv")

if uploaded_file:
    st.info("📖 Reading the dataset...")
    df_raw = read_csv_safe(uploaded_file)
    st.success("✅ Dataset loaded successfully")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df_raw.shape[0])
        st.metric("Columns", df_raw.shape[1])
    with col2:
        st.dataframe(df_raw.head(), use_container_width=True)

    if st.button("🚀 PREPROCESS & TRAIN MODEL", type="primary", use_container_width=True):
        X, y = preprocess_data(df_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y_train)

        acc = accuracy_score(y_test, model.predict(X_test_scaled))

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_names = X.columns.tolist()
        st.session_state.X_train_raw = X_train

        st.success("🎉 Model trained successfully")
        st.metric("Accuracy", f"{acc * 100:.2f}%")

        show_feature_importance(model, X.columns.tolist())

# ---------------- PREDICTION ----------------
if 'model' in st.session_state:
    st.subheader("🔮 Predict Host Type")

    tab1, tab2, tab3 = st.tabs(["📊 Core Features", "✅ Host Flags", "📍 Location"])
    input_data = {}

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['price'] = st.number_input("💰 Price ($)", 10.0, 5000.0, 100.0)
            input_data['accommodates'] = st.selectbox("👥 Accommodates", [1,2,3,4,5,6,8,10], 1)
        with col2:
            input_data['bedrooms'] = st.selectbox("🛏️ Bedrooms", [0,1,2,3,4,5], 1)
            input_data['review_scores_rating'] = st.slider("⭐ Review Score", 0.0, 100.0, 90.0)
        with col3:
            input_data['minimum_nights'] = st.selectbox("📅 Min Nights", [1,2,3,5,7,30], 1)
            input_data['amenities'] = st.selectbox("🛋️ Amenities", [0,5,10,15,20,30,50], 3)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            input_data['host_is_superhost_t'] = st.selectbox("⭐ Superhost", ["No","Yes"])
            input_data['host_has_profile_pic_1'] = st.selectbox("📸 Profile Pic", ["No","Yes"], 1)
            input_data['host_identity_verified'] = st.selectbox("🔐 ID Verified", ["No","Yes"], 1)
            input_data['instant_bookable_t'] = st.selectbox("⚡ Instant Book", ["No","Yes"], 1)
        with col2:
            input_data['host_response_rate'] = st.slider("📞 Response Rate", -1.0, 1.0, 1.0)
            input_data['host_acceptance_rate'] = st.slider("✅ Acceptance Rate", -1.0, 1.0, 1.0)
            input_data['host_response_time_encoded'] = st.selectbox(
                "⏱️ Response Time", ["None","Fast","Medium","Slow"], 1
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            input_data['latitude'] = st.slider("📍 Latitude", 30.0, 50.0, 40.75)
            input_data['longitude'] = st.slider("🌐 Longitude", -80.0, -70.0, -74.0)
        with col2:
            input_data['host_since_days'] = st.selectbox(
                "📅 Host Experience", ["New (<1yr)","1yr","2yrs","3yrs","5yrs","10yrs+"], 3
            )
            input_data['property_group_freq'] = st.slider("🏘️ Property Rarity", 0.0, 1.0, 0.5)

    # ---------- ADVANCED FEATURES ----------
    final_input = {}
    handled = set(input_data.keys())

    with st.expander("⚙️ Advanced / Auto Features"):
        cols = st.columns(3)
        auto_inputs = {}
        for i, f in enumerate(st.session_state.feature_names):
            if f not in handled:
                with cols[i % 3]:
                    auto_inputs[f] = st.number_input(
                        f, value=float(st.session_state.X_train_raw[f].median())
                    )

    for feature in st.session_state.feature_names:
        if feature in input_data:
            val = input_data[feature]
            if val == "Yes": val = 1
            if val == "No": val = 0
            if feature == "host_since_days":
                val = {"New (<1yr)":180,"1yr":365,"2yrs":730,"3yrs":1095,"5yrs":1825,"10yrs+":3650}[val]
            if feature == "host_response_time_encoded":
                val = {"None":0,"Fast":3,"Medium":2,"Slow":1}[val]
            final_input[feature] = val
        elif feature in auto_inputs:
            final_input[feature] = auto_inputs[feature]
        else:
            final_input[feature] = float(st.session_state.X_train_raw[feature].median())

    # ---------- RESULT PRESENTATION ----------
    if st.button("🎯 PREDICT", type="primary", use_container_width=True):
        df_input = pd.DataFrame([final_input])
        df_scaled = st.session_state.scaler.transform(df_input)

        pred = st.session_state.model.predict(df_scaled)[0]
        prob = st.session_state.model.predict_proba(df_scaled)[0]

        st.markdown("## 🎯 Prediction Result")

        if pred == 1:
            st.success("🆕 **Predicted Host Type: NEW HOST**")
        else:
            st.info("👨‍💼 **Predicted Host Type: EXPERIENCED HOST**")

        col1, col2, col3 = st.columns(3)
        col1.metric("🏷️ Host Classification", "NEW HOST" if pred else "EXPERIENCED HOST")
        col2.metric("🆕 New Host Probability", f"{prob[1]*100:.2f}%")
        col3.metric("👨‍💼 Experienced Host Probability", f"{prob[0]*100:.2f}%")

        st.markdown("### 📊 Confidence Level")
        st.markdown("**New Host Likelihood**")
        st.progress(float(prob[1]))

        st.markdown("**Experienced Host Likelihood**")
        st.progress(float(prob[0]))

        st.markdown("""
        ### 🧠 Interpretation
        - Higher probability means stronger confidence  
        - Close probabilities indicate borderline hosts  
        - Useful for onboarding, pricing & risk analysis
        """)

st.markdown("---")