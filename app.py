import streamlit as st
import requests

st.set_page_config(page_title="Customer Propensity Predictor", layout="wide")

st.title("🛒 Customer Purchasing Intent Predictor")
st.markdown("Adjust customer web session metrics to generate real-time purchase propensity predictions.")

# Form inputs layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Session Activity")
    admin = st.number_input("Administrative Pages Visited", min_value=0, value=2)
    admin_dur = st.number_input("Admin Duration (s)", min_value=0.0, value=50.0)
    info = st.number_input("Informational Pages Visited", min_value=0, value=0)
    info_dur = st.number_input("Informational Pages Duration (s)", min_value=0.0, value=0.0)
    product = st.number_input("Product Related Pages Visited", min_value=0, value=10)
    product_dur = st.number_input("Product Pages Duration (s)", min_value=0.0, value=300.0)


with col2:
    st.subheader("Web Traffic Metrics")
    bounce = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.0)
    exit_rate = st.number_input("Exit Rate", min_value=0.0, max_value=1.0, value=0.02)
    page_val = st.number_input("Page Value", min_value=0.0, value=15.5)
    special = st.number_input("Special Day Proximity", min_value=0.0, max_value=1.0, value=0.0)

with col3:
    st.subheader("User Environment")
    month = st.selectbox("Month", ["Jan", "Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                         index=9)
    visitor = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], index=None, \
                           placeholder="Select one option")
    weekend = st.checkbox("Weekend Visit", value=False)
    os_sys = st.number_input("OS ID", min_value=1, max_value=8, value=2)
    browser = st.number_input("Browser ID", min_value=1, max_value=13, value=2)
    region = st.number_input("Region ID", min_value=1, max_value=9, value=1)
    traffic = st.number_input("Traffic Type ID", min_value=1, max_value=20, value=2)

st.markdown("---")

# Predict button logic
if st.button("🚀 Predict Purchase Intent", type="primary", use_container_width=True):
    # Construct JSON payload matching CustomerFeaturesInput schema
    payload = {
        "Administrative": admin,
        "Administrative_Duration": admin_dur,
        "Informational": info,
        "Informational_Duration": info_dur,
        "ProductRelated": product,
        "ProductRelated_Duration": product_dur,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_val,
        "SpecialDay": special,
        "Month": month,
        "OperatingSystems": os_sys,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "VisitorType": visitor,
        "Weekend": weekend
    }

    # Point to the FastAPI service container inside docker network
    API_URL = "http://api:8000/predict"

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            if result["high_value_flag"]:
                st.success("🎉 High Intent to Purchase detected!")
            else:
                st.info("ℹ️ Low Intent to Purchase detected.")
            st.json(result)
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI server: {str(e)}")