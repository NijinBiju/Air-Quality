import streamlit as st
import pandas as pd
import pickle
import requests
from streamlit_lottie import st_lottie

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ---------------- LOAD LOTTIE ----------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_good = load_lottie_url(
    "https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json"
)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style="text-align:center;">🌍 Air Quality Prediction System</h1>
<p style="text-align:center;">ML-Based Pollution Monitoring & Alerts</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Project Info")
st.sidebar.info("""
**Model:** Random Forest  
**Scaling:** Standard Scaler  
**Output:** AQI Group  
**Type:** Classification
""")
st.sidebar.caption("Developed using Streamlit & Python")
st.sidebar.image("image.png", use_container_width=True)

# ---------------- INPUT SECTION ----------------
st.subheader("🧪 Enter Environmental Values")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(
        label=feature,
        value=0.0,
        step=0.1
    )

# ---------------- PREDICT BUTTON ----------------
st.markdown("###")
predict_btn = st.button("🚀 Predict Air Quality", use_container_width=True)

# ---------------- RESULT SECTION ----------------
if predict_btn:
    with st.spinner("🔍 Analyzing air quality..."):
        df = pd.DataFrame([user_input])
        df = df[features]   # correct feature order
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]

    st.markdown("---")

    # -------- GOOD (0) --------
    if prediction == 0:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg,#a8ff78,#78ffd6);
            padding:25px;
            border-radius:20px;
            text-align:center;
        ">
            <h2>🌱 GOOD AIR QUALITY</h2>
            <p>Clean & Healthy Environment</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(30)
        st_lottie(lottie_good, height=180)
        st.success("✅ Safe for outdoor activities & exercise")

    # -------- MODERATE (1) --------
    elif prediction == 1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg,#ffe29f,#ffa751);
            padding:25px;
            border-radius:20px;
            text-align:center;
        ">
            <h2>⚠️ MODERATE AIR QUALITY</h2>
            <p>Sensitive groups should be cautious</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(60)
        st.warning("⚠️ Reduce prolonged outdoor exertion")

    # -------- POOR (2) --------
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg,#ff416c,#ff4b2b);
            padding:25px;
            border-radius:20px;
            text-align:center;
            color:white;
        ">
            <h2>🚨 POOR AIR QUALITY</h2>
            <p>High health risk detected</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(90)
        st.error("❌ Avoid outdoor exposure | 😷 Wear protective mask")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style="text-align:center;">
🌍 Air Quality Prediction System | Built with ❤️ using Streamlit & Machine Learning
</p>
""", unsafe_allow_html=True)
