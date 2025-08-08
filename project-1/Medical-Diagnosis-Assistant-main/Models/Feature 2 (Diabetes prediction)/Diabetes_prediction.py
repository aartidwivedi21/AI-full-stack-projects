import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.units import inch
import joblib

# Paths to save/load model and scaler
MODEL_PATH = "Models/Feature 2 (Diabetes prediction)/diabetes_rf_model.joblib"
SCALER_PATH = "Models/Feature 2 (Diabetes prediction)/scaler.joblib"

# Load the dataset
data = pd.read_csv('Models/Feature 2 (Diabetes prediction)/diabetes.csv')

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if model and scaler exist, else train and save them
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded existing model and scaler.")
else:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Trained and saved model and scaler.")

# Prediction function
def predict_diabetes_from_input(Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DiabetesPedigreeFunction, Age]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

# PDF generation function (unchanged)
def generate_pdf_report(username, user_input, prediction_result):
    os.makedirs("Reports", exist_ok=True)
    file_name = f"{username} ({prediction_result}).pdf"
    file_path = os.path.join("Reports", file_name)

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.darkblue)
    c.drawCentredString(width / 2, y, "🩺 Diabetes Prediction Report")
    c.setFillColor(colors.black)
    y -= 40

    # Line separator
    c.setLineWidth(1)
    c.line(50, y, width - 50, y)
    y -= 30

    # User Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Patient Information")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Name: {username}")
    y -= 20
    c.drawString(60, y, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    c.setFillColor(colors.red if prediction_result == "Diabetic" else colors.green)
    c.drawString(50, y, f"Prediction: {prediction_result}")
    c.setFillColor(colors.black)
    y -= 30

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, y, "Report generated using Python, Scikit-learn, and ReportLab")

    c.save()
    print(f"\n✅ Styled report saved to: {file_path}")

# Main program
if __name__ == "__main__":
    print("=== Diabetes Prediction ===")
    username = input("Enter your name: ")

    print("\nEnter the following medical details:")
    user_input = {
        'Pregnancies': int(input("Pregnancies: ")),
        'Glucose': float(input("Glucose: ")),
        'BloodPressure': float(input("Blood Pressure: ")),
        'SkinThickness': float(input("Skin Thickness: ")),
        'Insulin': float(input("Insulin: ")),
        'BMI': float(input("BMI: ")),
        'DiabetesPedigreeFunction': float(input("Diabetes Pedigree Function: ")),
        'Age': int(input("Age: "))
    }

    result = predict_diabetes_from_input(**user_input)
    print(f"\nPrediction: {result}")
    
    generate_pdf_report(username, user_input, result)

    # Optional: Print accuracy
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    print("Model accuracy on test data:", accuracy_score(y_test, y_pred))
