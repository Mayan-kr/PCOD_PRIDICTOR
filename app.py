import streamlit as st
import pandas as pd
import joblib

# 💾 Load trained model
model = joblib.load("pcod_model.pkl")

# 🔁 Y/N or R/I encoder
def encode(val):
    return 1 if val in ["Y", "I"] else 0

# 📥 User input function
def get_user_input():
    age = st.number_input("Age (yrs)", min_value=10, max_value=60)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
    pulse = st.number_input("Pulse rate (bpm)", min_value=40, max_value=150)
    hb = st.number_input("Hb (g/dl)", min_value=5.0, max_value=20.0)
    cycle_type = st.selectbox("Cycle Type (R=Regular / I=Irregular)", ["R", "I"])
    cycle_length = st.number_input("Cycle length (days)", min_value=10, max_value=50)
    pregnant = st.selectbox("Currently Pregnant? (Y/N)", ["Y", "N"])
    abortions = st.number_input("No. of abortions", min_value=0, max_value=10)
    waist_hip_ratio = st.number_input("Waist:Hip Ratio", min_value=0.5, max_value=2.0)
    weight_gain = st.selectbox("Weight gain (Y/N)", ["Y", "N"])
    hair_growth = st.selectbox("Hair growth (Y/N)", ["Y", "N"])
    skin_darkening = st.selectbox("Skin darkening (Y/N)", ["Y", "N"])
    hair_loss = st.selectbox("Hair loss (Y/N)", ["Y", "N"])
    pimples = st.selectbox("Pimples (Y/N)", ["Y", "N"])
    fast_food = st.selectbox("Fast food (Y/N)", ["Y", "N"])
    exercise = st.selectbox("Regular Exercise? (Y/N)", ["Y", "N"])

    features = pd.DataFrame({
        'Age (yrs)': [age],
        'BMI': [bmi],
        'Pulse rate(bpm)': [pulse],
        'Hb(g/dl)': [hb],
        'Cycle(R/I)': [encode(cycle_type)],
        'Cycle length(days)': [cycle_length],
        'Pregnant(Y/N)': [encode(pregnant)],
        'No. of abortions': [abortions],
        'Waist:Hip Ratio': [waist_hip_ratio],
        'Weight gain(Y/N)': [encode(weight_gain)],
        'hair growth(Y/N)': [encode(hair_growth)],
        'Skin darkening (Y/N)': [encode(skin_darkening)],
        'Hair loss(Y/N)': [encode(hair_loss)],
        'Pimples(Y/N)': [encode(pimples)],
        'Fast food (Y/N)': [encode(fast_food)],
        'Reg.Exercise(Y/N)': [encode(exercise)],
    })

    return features

# 🖼️ App UI
st.set_page_config(page_title="PCOD Prediction App 💖", layout="centered")

st.title("💊 PCOD Prediction System")
st.write("Predict whether someone might be at risk of PCOD based on health features.")

menu = st.sidebar.selectbox("Menu", ["Predict", "Help Corner"])

if menu == "Predict":
    st.subheader("🧪 Enter Health Information:")
    user_input = get_user_input()

    if st.button("Predict PCOD Status"):
        prediction = model.predict(user_input)[0]
        if prediction == 1:
            st.error("⚠️ High Risk of PCOD detected. Please consult a doctor.")
        else:
            st.success("🎉 No PCOD detected. Keep up the healthy habits!")

elif menu == "Help Corner":
    st.subheader("🩺 Feature Descriptions")
    st.markdown("""
    - **Age (yrs)**: Age of the individual.
    - **BMI**: Body Mass Index, used to assess weight category.
    - **Pulse rate (bpm)**: Number of heartbeats per minute.
    - **Hb (g/dl)**: Hemoglobin level in the blood.
    - **Cycle(R/I)**: Regular or Irregular menstrual cycle.
    - **Cycle length (days)**: Duration of one menstrual cycle.
    - **Pregnant(Y/N)**: Is the person currently pregnant?
    - **No. of abortions**: Number of past abortions.
    - **Waist:Hip Ratio**: Waist measurement divided by hip measurement.
    - **Weight gain (Y/N)**: Has there been any unexplained weight gain?
    - **Hair growth (Y/N)**: Abnormal body/facial hair growth.
    - **Skin darkening (Y/N)**: Skin pigmentation in certain areas.
    - **Hair loss (Y/N)**: Thinning or loss of scalp hair.
    - **Pimples (Y/N)**: Acne severity.
    - **Fast food (Y/N)**: Frequent consumption of junk food.
    - **Reg.Exercise (Y/N)**: Regular exercise habits.
    """)
