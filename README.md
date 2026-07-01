# 🌦️ Will It Rain Tomorrow?

A Streamlit web app that predicts whether it will **rain tomorrow** in Australia
(`RainTomorrow`) based on today's weather observations.

The model is a **logistic regression** trained on the
[weatherAUS dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
(~10 years of daily observations from numerous Australian weather stations).

## How it works

The file `models/aussie_rain.joblib` is not just a model — it's a dictionary that
bundles the trained estimator together with every preprocessing component needed
for inference:

| Key | Purpose |
|-----|---------|
| `model` | trained `LogisticRegression` |
| `imputer` | `SimpleImputer` — fills missing numeric values (mean strategy) |
| `scaler` | `MinMaxScaler` — scales numeric features |
| `encoder` | `OneHotEncoder` — encodes categorical features |
| `input_cols`, `numeric_cols`, `categorical_cols`, `encoded_cols` | column metadata |

When the user clicks the predict button, the raw input goes through the full
preprocessing pipeline before reaching the model:

1. impute missing numeric values (`imputer`);
2. scale numeric features (`scaler`);
3. one-hot encode categorical features (`encoder`);
4. predict the class and probability (`predict` / `predict_proba`).

The app then shows whether rain is expected tomorrow (Yes/No) along with the
predicted probability.

## Project structure

```
weather_streamlit_app/
├── app.py                      # Streamlit application
├── models/
│   └── aussie_rain.joblib      # model + preprocessing components
├── requirements.txt            # dependencies
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this project to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → select the repository, branch, and set the main file to `app.py`.
4. Open **Advanced settings** and pin the **Python version to 3.13** (see the note below).
5. **Deploy** — you'll get a public URL in a couple of minutes.

### Python version

The Python version is pinned through **Advanced settings** in the Streamlit deploy
UI — this app runs on **Python 3.13**.

## Tech stack

- **Python**
- **scikit-learn** — model & preprocessing
- **Streamlit** — web UI
- **pandas / numpy** — data handling
- **joblib** — model persistence
