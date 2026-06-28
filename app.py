import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Чи піде завтра дощ?", page_icon="🌧️", layout="centered")

MODEL_PATH = "models/aussie_rain.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


aussie_rain = load_model()
model = aussie_rain["model"]
imputer = aussie_rain["imputer"]
scaler = aussie_rain["scaler"]
encoder = aussie_rain["encoder"]
input_cols = aussie_rain["input_cols"]
numeric_cols = aussie_rain["numeric_cols"]
categorical_cols = aussie_rain["categorical_cols"]
encoded_cols = aussie_rain["encoded_cols"]


def predict_rain(user_input: dict):

    input_df = pd.DataFrame([user_input], columns=input_cols)

    # 1-2. Числові ознаки: імпутація + масштабування
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # 3. Категоріальні ознаки: one-hot кодування
    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=input_df.index)

    # 4. Формуємо фінальний набір ознак і прогнозуємо
    X = pd.concat([input_df[numeric_cols], encoded_df], axis=1)[numeric_cols + encoded_cols]
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Ймовірність саме класу "Yes" (буде дощ)
    classes = list(model.classes_)
    prob_yes = probability[classes.index("Yes")]
    return prediction, prob_yes


# ---------------------------------------------------------------------------
# Інтерфейс
# ---------------------------------------------------------------------------
st.title("🌦️ Прогноз дощу в Австралії")
st.markdown(
    "Цей застосунок передбачає, **чи піде завтра дощ** (`RainTomorrow`), "
    "на основі сьогоднішніх метеоспостережень. "
    "Модель — логістична регресія, навчена на датасеті *weatherAUS*."
)
st.divider()

# Категорії беремо напряму з навченого енкодера
locations = [c for c in encoder.categories_[categorical_cols.index("Location")] if c != "nan"]
wind_dirs = [c for c in encoder.categories_[categorical_cols.index("WindGustDir")] if c != "nan"]

st.subheader("📍 Локація та погода сьогодні")
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Метеостанція (Location)", locations, index=locations.index("Sydney") if "Sydney" in locations else 0)
    min_temp = st.number_input("Мін. температура, °C (MinTemp)", -10.0, 35.0, 12.0, 0.1)
    max_temp = st.number_input("Макс. температура, °C (MaxTemp)", -5.0, 50.0, 23.0, 0.1)
    rainfall = st.number_input("Опади сьогодні, мм (Rainfall)", 0.0, 400.0, 0.0, 0.1)
    evaporation = st.number_input("Випаровування, мм (Evaporation)", 0.0, 85.0, 5.0, 0.1)
    sunshine = st.number_input("Сонячне сяйво, год (Sunshine)", 0.0, 15.0, 8.0, 0.1)
    rain_today = st.selectbox("Чи був дощ сьогодні? (RainToday)", ["No", "Yes"])

with col2:
    wind_gust_dir = st.selectbox("Напрям пориву вітру (WindGustDir)", wind_dirs)
    wind_gust_speed = st.number_input("Швидкість пориву вітру, км/год (WindGustSpeed)", 6.0, 135.0, 40.0, 1.0)
    wind_dir_9am = st.selectbox("Напрям вітру о 9:00 (WindDir9am)", wind_dirs)
    wind_dir_3pm = st.selectbox("Напрям вітру о 15:00 (WindDir3pm)", wind_dirs)
    wind_speed_9am = st.number_input("Швидкість вітру о 9:00, км/год (WindSpeed9am)", 0.0, 90.0, 14.0, 1.0)
    wind_speed_3pm = st.number_input("Швидкість вітру о 15:00, км/год (WindSpeed3pm)", 0.0, 90.0, 20.0, 1.0)

st.subheader("💧 Вологість, тиск, хмарність, температура")
col3, col4 = st.columns(2)

with col3:
    humidity_9am = st.slider("Вологість о 9:00, % (Humidity9am)", 0, 100, 70)
    humidity_3pm = st.slider("Вологість о 15:00, % (Humidity3pm)", 0, 100, 50)
    pressure_9am = st.number_input("Тиск о 9:00, гПа (Pressure9am)", 980.0, 1042.0, 1015.0, 0.1)
    pressure_3pm = st.number_input("Тиск о 15:00, гПа (Pressure3pm)", 979.0, 1040.0, 1013.0, 0.1)

with col4:
    cloud_9am = st.slider("Хмарність о 9:00, октанти (Cloud9am)", 0, 9, 4)
    cloud_3pm = st.slider("Хмарність о 15:00, октанти (Cloud3pm)", 0, 9, 4)
    temp_9am = st.number_input("Температура о 9:00, °C (Temp9am)", -6.0, 41.0, 17.0, 0.1)
    temp_3pm = st.number_input("Температура о 15:00, °C (Temp3pm)", -6.0, 47.0, 22.0, 0.1)

st.divider()

if st.button("Спрогнозувати погоду на завтра", type="primary", use_container_width=True):
    user_input = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
    }

    prediction, prob_yes = predict_rain(user_input)

    if prediction == "Yes":
        st.error(f"☔ Завтра, ймовірно, **піде дощ**.")
    else:
        st.success(f"☀️ Завтра дощу, ймовірно, **не буде**.")

    st.metric("Ймовірність дощу завтра", f"{prob_yes * 100:.1f}%")
    st.progress(float(prob_yes))
