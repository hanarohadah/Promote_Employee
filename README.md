# Employee Promotion Prediction

A machine learning application that predicts whether an employee is eligible for promotion based on their performance and work-related factors.

## Overview

This project uses a **RandomForest classifier** with preprocessing pipelines to predict employee promotion eligibility. The web interface is built with **Streamlit**, allowing easy interaction without coding knowledge.

## Features

- 🎯 **Accurate Predictions** — Machine learning model trained on employee data
- 📊 **User-Friendly Interface** — Simple web UI built with Streamlit
- 📈 **Performance Metrics** — Displays prediction confidence scores
- 🔧 **Easy Deployment** — Lightweight and fast prediction inference

## Dataset

The model is trained on the `Kinerja_Karyawan_Sintetik.csv` dataset containing:
- **lama_kerja_tahun** — Years of employment
- **rata_rata_rating** — Average performance rating (1-5)
- **jumlah_pelatihan** — Number of trainings attended
- **gaji_kenaikan_usd** — Salary increase (USD)
- **departemen** — Department (Tech, Sales, HR)
- **promosi_ya_tidak** — Promotion status (Yes/No)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
ProyekKinerjaKaryawan/
├── app.py                           # Streamlit web application
├── training.py                      # Model training script
├── model_kinerja_karyawan           # Trained model (binary)
├── Kinerja_Karyawan_Sintetik.csv   # Training dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## How to Use

1. Open the Streamlit app: `streamlit run app.py`
2. Enter employee data using the sliders:
   - Years of Employment (0-40)
   - Performance Rating (1-5)
   - Number of Trainings (0-40)
   - Department (Tech / Sales / HR)
   - Salary Increase (0-20,000 USD)
3. Click **"Prediksi Promosi Karyawan"** button
4. View the prediction result with confidence percentage

## Model Details

- **Algorithm** — Random Forest Classifier
- **Preprocessing** — StandardScaler (numeric) + OneHotEncoder (categorical)
- **Pipeline** — scikit-learn ColumnTransformer with Pipeline
- **Test Accuracy** — Evaluated on 20% test split

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib
- streamlit
- matplotlib
- numpy

See `requirements.txt` for exact versions.

## Future Improvements

- [ ] Add more features (age, work experience, etc.)
- [ ] Implement model explainability (SHAP values)
- [ ] Add data visualization and analytics dashboard
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
- [ ] Multi-language support

## Author

**Hana Rohadah** — 2025

## License

This project is provided as-is for educational purposes.
Link -> https://promote-employee.streamlit.app/

---

**Need Help?** — Check the error message in the Streamlit interface or review the `training.py` file to understand the model pipeline.
