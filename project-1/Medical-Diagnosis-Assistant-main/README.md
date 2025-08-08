# 🏥 Medical Diagnosis Assistant

## 📌 What is Medical Diagnosis Assistant?

**Medical Diagnosis Assistant** is an intelligent, multi-featured desktop application built using Python. It aids users in predicting diseases by analyzing user-provided symptoms or medical data. Using machine learning models, it provides accurate predictions and generates detailed PDF reports to help users understand their health condition better.

The application covers:
- General disease prediction based on symptoms.
- Diabetes risk prediction based on health metrics.
- Heart disease prediction from medical indicators.
- Auto-generated PDF reports summarizing each prediction.

---

## 📂 Dataset Explanation (`Datasets/`)

| Filename                                             | Description |
|------------------------------------------------------|-------------|
| `dataset.csv`                                        | A raw disease-symptom dataset used for preliminary cleaning and understanding common symptom associations with diseases. |
| `diabetes.csv`                                       | Contains health metrics (glucose level, BMI, age, etc.) for diabetes prediction. |
| `Disease_symptom_and_patient_profile_dataset.csv`    | A larger combined dataset featuring diseases, symptoms, and patient profiles for potential future integration. |
| `heart.csv`                                          | Dataset with patient data (age, cholesterol, resting BP, etc.) for heart disease prediction. |
| `Symptom-severity.csv`                               | Maps symptoms to their severity on a scale of 1–5. Used for enhanced report generation and feature encoding. |
| `symptom_Description.csv`                            | Contains detailed descriptions for each symptom used in user explanation and report insights. |
| `symptom_precaution.csv`                             | Maps diseases to a list of recommended precautions for user safety and post-diagnosis care. |

---

## 📒 Jupyter Folder (`Jupyter/`)

This folder includes all **EDA (Exploratory Data Analysis)**, **data cleaning**, and **model training** work.

- `Model1.ipynb`: Handles EDA and model development for **disease prediction** using the cleaned dataset.
- `Model2.ipynb`: Contains the **diabetes prediction** model development including correlation analysis and feature scaling.
- `Model(heart_disease).ipynb`: Used for building and evaluating the **heart disease prediction** model.
- EDA Steps:
  - Null value detection and handling.
  - Label encoding and one-hot encoding for symptoms.
  - Feature importance visualization.
  - Performance metrics like accuracy, confusion matrix, and classification report.

All intermediate cleaned datasets (`cleaned_dataset.csv`) and other copies used during development are stored here.

---

## 🧠 Model Structure (`Models/`)

This folder contains the trained models and scripts for each prediction feature.

---

### 📌 Feature 1: Disease Prediction and Report Generation
- **Path**: `Models/Feature 1 ( Disease prediction and report generation)/`
- **Files**:
  - `Disease_prediction.py`: Accepts user symptoms, preprocesses them, and predicts the most likely disease.
  - `rf_model.joblib`: Random Forest model trained on cleaned multi-label symptom-disease data.
  - `mlb.joblib`: MultiLabelBinarizer to convert symptom lists to numerical arrays.
  - `symptom_Description.csv`, `symptom_precaution.csv`, `Symptom-severity.csv`: Used for enhancing prediction output and PDF reports.

---

### 📌 Feature 2: Diabetes Prediction
- **Path**: `Models/Feature 2 (Diabetes prediction)/`
- **Files**:
  - `Diabetes_prediction.py`: Takes input like Glucose, BMI, Age, etc., and uses a trained Random Forest model.
  - `diabetes_rf_model.joblib`: Trained ML model for predicting diabetes.
  - `scaler.joblib`: Scaler used to normalize input values.

---

### 📌 Feature 3: Heart Disease Prediction
- **Path**: `Models/Feature 3 (Heart Disease Prediction)/`
- **Files**:
  - `Heart_Disease_prediction.py`: Predicts likelihood of heart disease using a logistic regression model.
  - `heart_model.pkl`: Pre-trained model for classification.
  - `scaler.pkl`: Feature scaler used during model training.

---

## 📄 Report Generation (`Reports/`)

After a successful prediction:
- The app generates a **customized PDF report** using the `reportlab` library.
- Report contains:
  - User-entered symptoms or metrics.
  - Predicted disease or risk status.
  - Description of symptoms.
  - Recommended precautions.
  - Severity analysis (for general diseases).

The reports are saved in the `Reports/` directory with timestamped filenames for easy access.

---

## 🧩 App.py - Main Application Logic

This is the **central script** that ties all features together. It serves as the user interface and orchestrates all backend model calls.

### Key Functions:
- Imports models and scalers from respective folders.
- Prompts users via terminal or GUI to:
  - Enter symptoms (Feature 1).
  - Input health metrics for diabetes (Feature 2).
  - Input heart health indicators (Feature 3).
- Calls the correct prediction script and retrieves the output.
- Displays results and invokes **PDF report generation** with all relevant info.

### Highlights:
- Modular code for easy maintenance.
- Dynamically loads trained models using `joblib` or `pickle`.
- Fully functional offline (no internet/API dependency).

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/medical-diagnosis-assistant.git
cd medical-diagnosis-assistant
pip install -r Requirements.txt.txt
streamlit run App.py
