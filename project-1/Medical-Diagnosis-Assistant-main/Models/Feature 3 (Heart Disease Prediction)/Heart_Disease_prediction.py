import pandas as pd
import numpy as np
import os
import random
import string
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# === File paths ===
MODEL_PATH = "Models/Feature 3 (Heart Disease Prediction)/heart_model.pkl"
SCALER_PATH = "Models/Feature 3 (Heart Disease Prediction)/scaler.pkl"
DATA_PATH = "Models/Feature 3 (Heart Disease Prediction)/heart.csv"

# === Model training and saving ===
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model and scaler saved.")

# === Load model and scaler ===
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    train_and_save_model()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Utility Functions ===

def generate_user_id(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_pdf_report(user_name, user_input, prediction_result):
    user_id = generate_user_id()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_dir = "Reports"
    os.makedirs(report_dir, exist_ok=True)

    file_name = f"HeartReport_{user_name}_{user_id}.pdf"
    file_path = os.path.join(report_dir, file_name)

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.darkred)
    c.drawCentredString(width / 2, y, "❤️ Heart Disease Prediction Report")
    y -= 40

    # Separator
    c.setLineWidth(1)
    c.setStrokeColor(colors.grey)
    c.line(50, y, width - 50, y)
    y -= 30

    # User Info
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawString(50, y, "Patient Information")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Name: {user_name}")
    y -= 20
    c.drawString(60, y, f"User ID: {user_id}")
    y -= 20
    c.drawString(60, y, f"Timestamp: {timestamp}")
    y -= 30

    # Input Data
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Input Parameters")
    y -= 20
    c.setFont("Helvetica", 12)
    for key, value in user_input.items():
        c.drawString(60, y, f"{key}: {value}")
        y -= 20
        if y < 100:
            c.showPage()
            y = height - 50

    y -= 10
    c.setLineWidth(0.5)
    c.line(50, y, width - 50, y)
    y -= 30

    # Prediction Result
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.red if "Detected" in prediction_result else colors.green)
    c.drawString(50, y, f"Prediction: {prediction_result}")
    c.setFillColor(colors.black)
    y -= 30

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, y, "Report generated using Python, Scikit-learn, and ReportLab")

    c.save()
    print(f"\n✅ Styled report saved to: {file_path}")

# === Prediction ===
def predict_heart_disease(user_name, user_input):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
    generate_pdf_report(user_name, user_input, result)
    return result

# === Main Program ===
if __name__ == "__main__":
    print("=== Heart Disease Prediction ===")
    user_name = input("Enter your name: ")

    print("\nPlease enter the following medical information:")
    user_input = {
        'age': int(input("Age: ")),
        'sex': int(input("Sex (1=Male, 0=Female): ")),
        'cp': int(input("Chest Pain Type (0-3): ")),
        'trestbps': int(input("Resting Blood Pressure: ")),
        'chol': int(input("Serum Cholesterol (mg/dl): ")),
        'fbs': int(input("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False): ")),
        'restecg': int(input("Resting ECG Results (0-2): ")),
        'thalach': int(input("Maximum Heart Rate Achieved: ")),
        'exang': int(input("Exercise Induced Angina (1=Yes, 0=No): ")),
        'oldpeak': float(input("ST Depression Induced by Exercise: ")),
        'slope': int(input("Slope of the ST Segment (0-2): ")),
        'ca': int(input("Number of Major Vessels (0-3): ")),
        'thal': int(input("Thal (1=Normal, 2=Fixed Defect, 3=Reversible Defect): "))
    }

    prediction_result = predict_heart_disease(user_name, user_input)
    print(f"\nPrediction: {prediction_result}")
    print("PDF report generated successfully.")