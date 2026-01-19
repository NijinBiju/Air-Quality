import streamlit as st
import pandas as pd
import pickle


st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)

model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

feature_info = {
    "CO(GT)": {
        "label": "Carbon Monoxide (CO)",
        "unit": "mg/m³",
        "desc": "Colorless toxic gas from vehicles. Normal range: 0–10 mg/m³."
    },
    "PT08.S1(CO)": {
        "label": "CO Sensor (Tin Oxide)",
        "unit": "Sensor Units",
        "desc": "Electronic sensor response related to carbon monoxide levels."
    },
    "NMHC(GT)": {
        "label": "Non-Methane Hydrocarbons (NMHC)",
        "unit": "µg/m³",
        "desc": "Pollutants released from fuel combustion and industries."
    },
    "C6H6(GT)": {
        "label": "Benzene (C₆H₆)",
        "unit": "µg/m³",
        "desc": "Highly toxic compound from vehicle exhausts."
    },
    "PT08.S2(NMHC)": {
        "label": "NMHC Sensor",
        "unit": "Sensor Units",
        "desc": "Sensor response detecting hydrocarbons in the air."
    },
    "NOx(GT)": {
        "label": "Nitrogen Oxides (NOx)",
        "unit": "ppb",
        "desc": "Harmful gases produced from engines and power plants."
    },
    "PT08.S3(NOx)": {
        "label": "NOx Sensor",
        "unit": "Sensor Units",
        "desc": "Sensor response related to nitrogen oxide concentration."
    },
    "NO2(GT)": {
        "label": "Nitrogen Dioxide (NO₂)",
        "unit": "µg/m³",
        "desc": "Irritating gas affecting lungs and respiratory system."
    },
    "PT08.S4(NO2)": {
        "label": "NO₂ Sensor",
        "unit": "Sensor Units",
        "desc": "Sensor response measuring nitrogen dioxide presence."
    },
    "PT08.S5(O3)": {
        "label": "Ozone Sensor (O₃)",
        "unit": "Sensor Units",
        "desc": "Sensor related to ground-level ozone concentration."
    },
    "T": {
        "label": "Temperature",
        "unit": "°C",
        "desc": "Ambient air temperature."
    },
    "RH": {
        "label": "Relative Humidity",
        "unit": "%",
        "desc": "Amount of moisture present in the air."
    },
    "AH": {
        "label": "Absolute Humidity",
        "unit": "g/m³",
        "desc": "Actual quantity of water vapor in the air."
    }
}


st.markdown("""
<style>
.header-card {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
.result-low {
    background-color: #e6fffa;
    padding: 20px;
    border-radius: 12px;
    color: #065f46;
    font-size: 20px;
}
.result-mid {
    background-color: #fff7ed;
    padding: 20px;
    border-radius: 12px;
    color: #92400e;
    font-size: 20px;
}
.result-high {
    background-color: #fee2e2;
    padding: 20px;
    border-radius: 12px;
    color: #7f1d1d;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header-card">
    <h1>🌍 Air Quality Prediction System</h1>
    <p>Beginner-friendly air pollution prediction using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


st.sidebar.header("⚙️ About Project")
st.sidebar.info("""
**Model:** Logistic Regression  
**Scaling:** Standard Scaler  
**Output:** Pollution Level  
**Type:** ML Classification
""")
st.sidebar.markdown("—")
st.sidebar.caption("Developed using Streamlit & Python")
st.sidebar.image("image.png", use_container_width=True)


col1, col2 = st.columns([2, 1])


with col1:
    st.subheader("🧪 Enter Environmental Values")

    user_input = {}

    for feature in features:
        info = feature_info.get(feature, {
            "label": feature,
            "unit": "",
            "desc": "Environmental sensor value used for air quality prediction."
        })

        user_input[feature] = st.number_input(
            label=f"{info['label']} ({info['unit']})",
            value=0.0,
            step=0.1,
            help=info["desc"]
        )


with col2:
    st.subheader("Air Quality Insight")

    if st.button("Predict Air Quality", use_container_width=True):
        df = pd.DataFrame([user_input])
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]

        if prediction == 0:
            st.markdown("""
            <div class="pred-card good">
                <div class="big-text">Good Air Quality</div>
                <p>Air is clean and healthy.</p>
                <ul>
                    <li>Outdoor activities are safe</li>
                    <li>No health risk for any age group</li>
                    <li>Ideal conditions for exercise</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        elif prediction == 1:
            st.markdown("""
            <div class="pred-card moderate">
                <div class="big-text">Moderate Air Quality</div>
                <p>Sensitive individuals may feel mild discomfort.</p>
                <ul>
                    <li>People with asthma should limit outdoor exertion</li>
                    <li>Children and elderly should be cautious</li>
                    <li>Short outdoor activities are generally safe</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="pred-card bad">
                <div class="big-text">Poor Air Quality</div>
                <p>Air pollution level is harmful.</p>
                <ul>
                    <li>Avoid outdoor activities</li>
                    <li>Wear masks if going outside</li>
                    <li>High risk for lung and heart patients</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)