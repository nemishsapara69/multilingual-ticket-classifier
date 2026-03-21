import os

import requests
import streamlit as st


st.set_page_config(page_title="Smart Router Demo", page_icon="Ticket", layout="centered")

st.title("End-to-End Multilingual Ticket Classifier")
st.write("Type a customer message in English, Spanish, or German, then classify it.")

api_url = os.getenv("API_URL", "http://localhost:8000/predict")

user_text = st.text_area("Customer message", placeholder="Mi luz inteligente no funciona")


def confidence_band(conf: float) -> tuple[str, str]:
    if conf >= 0.75:
        return "High", "green"
    if conf >= 0.5:
        return "Medium", "orange"
    return "Low", "red"

if st.button("Classify Ticket"):
    if not user_text.strip():
        st.warning("Please enter a message.")
    else:
        try:
            response = requests.post(api_url, json={"text": user_text}, timeout=20)
            if response.ok:
                result = response.json()
                band, color = confidence_band(float(result["confidence"]))

                st.success("Prediction complete")
                st.metric("Predicted Category", result["category"])
                st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
                st.markdown(
                    f"Confidence Band: :{color}[{band}]"
                )
                if band == "Low":
                    st.info("Low confidence: consider manual review by a support agent.")
                with st.expander("Cleaned text"):
                    st.write(result["cleaned_text"])
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")

st.caption("Tip: run API first, then launch this app.")
