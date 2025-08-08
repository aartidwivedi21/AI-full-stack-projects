import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart

# === Load cleaned dataset ===
df = pd.read_csv("Models/Feature 1 ( Disease prediction and report generation)/cleaned_dataset.csv")

# Combine and clean symptoms
symptom_columns = df.columns[1:]  # Assuming first column is 'Disease'
df['all_symptoms'] = df[symptom_columns].values.tolist()
df['all_symptoms'] = df['all_symptoms'].apply(
    lambda x: [sym.strip().lower() for sym in x if isinstance(sym, str)]
)
df = df[df['all_symptoms'].apply(lambda x: len(x) > 0)]

# Prepare features and labels
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['all_symptoms'])
y = df['Disease'].str.strip().str.lower()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and MultiLabelBinarizer using joblib
model_path = "Models/Feature 1 ( Disease prediction and report generation)/rf_model.joblib"
mlb_path = "Models/Feature 1 ( Disease prediction and report generation)/mlb.joblib"
joblib.dump(model, model_path)
joblib.dump(mlb, mlb_path)

# === Load additional datasets ===
severity_df = pd.read_csv("Models/Feature 1 ( Disease prediction and report generation)/Symptom-severity.csv")
description_df = pd.read_csv("Models/Feature 1 ( Disease prediction and report generation)/symptom_Description.csv")
precaution_df = pd.read_csv("Models/Feature 1 ( Disease prediction and report generation)/symptom_precaution.csv")

# Normalize and create lookup dictionaries
severity_dict = {k.strip().lower(): v for k, v in zip(severity_df['Symptom'], severity_df['weight'])}
description_dict = {k.strip().lower(): v for k, v in zip(description_df['Disease'], description_df['Description'])}

# Fix for precaution dictionary: filter out NaNs and empty strings
precaution_dict = {
    k.strip().lower(): [str(p).strip() for p in v if isinstance(p, str) and p.strip() != '']
    for k, v in precaution_df.set_index('Disease').T.to_dict('list').items()
}

def predict_disease(user_symptoms):
    cleaned = [sym.strip().lower() for sym in user_symptoms]
    user_input_vector = mlb.transform([cleaned])
    prediction = model.predict(user_input_vector)
    return prediction[0].strip().lower()

def create_severity_chart(symptoms):
    drawing = Drawing(400, 150)
    
    # Extract severity values and labels
    severity_values = [severity_dict.get(sym, 0) for sym in symptoms]
    labels = [sym.title() for sym in symptoms]
    
    # Bar chart
    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 30
    bc.height = 100
    bc.width = 300
    bc.data = [severity_values]
    bc.barWidth = 18
    bc.groupSpacing = 10
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(severity_values) + 1 if severity_values else 5
    bc.valueAxis.valueStep = 1
    bc.categoryAxis.categoryNames = labels
    bc.barLabels.nudge = 7
    bc.barLabels.dy = -10
    bc.barLabels.fontSize = 8
    bc.barLabels.fillColor = colors.black
    bc.bars[0].fillColor = colors.HexColor("#4B8BBE")
    
    drawing.add(bc)
    
    # Title label replaced with String
    title = String(200, 140, 'Symptom Severity Levels')  # x=200 for centered approx.
    title.textAnchor = 'middle'  # horizontally centered
    title.fontSize = 12
    title.fillColor = colors.darkblue
    drawing.add(title)
    
    return drawing

def create_pdf_report(disease, symptoms, precautions):
    if not os.path.exists("Reports"):
        os.makedirs("Reports")

    filename = f"Reports/{disease.replace(' ', '_')}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.HexColor("#2E4053"))
    c.drawCentredString(width / 2, height - inch, "Medical Diagnosis Report")

    # Horizontal line
    c.setStrokeColor(colors.HexColor("#566573"))
    c.setLineWidth(1.5)
    c.line(inch, height - 1.1*inch, width - inch, height - 1.1*inch)

    # Disease Name
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.black)
    c.drawString(inch, height - 1.6*inch, f"Disease Predicted: {disease.title()}")

    # Symptoms Section Title
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#2874A6"))
    c.drawString(inch, height - 2.1*inch, "Symptoms Provided:")

    c.setFont("Helvetica", 12)
    y = height - 2.4*inch
    line_height = 16

    # Symptoms with severity and description
    for sym in symptoms:
        sym_clean = sym.strip().lower()
        sev = severity_dict.get(sym_clean, "N/A")
        desc = description_dict.get(sym_clean, "No description available.")
        text = f"- {sym_clean.title()} (Severity: {sev}): {desc}"
        text_lines = split_text(text, 80)
        for line in text_lines:
            c.drawString(inch + 20, y, line)
            y -= line_height
            if y < inch + 120:
                c.showPage()
                y = height - inch

    # Add severity chart
    drawing = create_severity_chart([sym.strip().lower() for sym in symptoms])
    drawing.wrapOn(c, width, height)
    drawing.drawOn(c, inch, y - 130)
    y -= 160

    # Precautions Section Title
    if y < inch + 80:
        c.showPage()
        y = height - inch
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#2874A6"))
    c.drawString(inch, y, "Recommended Precautions:")
    y -= 24

    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)

    if precautions:
        for i, p in enumerate(precautions, 1):
            precaution_lines = split_text(f"{i}. {p}", 90)
            for line in precaution_lines:
                c.drawString(inch + 20, y, line)
                y -= line_height
                if y < inch:
                    c.showPage()
                    y = height - inch
    else:
        c.drawString(inch + 20, y, "No specific precautions found.")

    c.save()
    print(f"\nPDF report generated and saved as '{filename}'.")

def split_text(text, max_chars):
    """Helper function to split long text into multiple lines for PDF."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def generate_report(user_symptoms):
    prediction = predict_disease(user_symptoms)
    print(f"\n🩺 Predicted Disease: {prediction.title()}\n")

    print("Matched Symptoms and Details:")
    for symptom in user_symptoms:
        s = symptom.strip().lower()
        desc = description_dict.get(s, "No description available.")
        sev = severity_dict.get(s, "N/A")
        print(f"- {s.title()} (Severity: {sev}): {desc}")

    precautions = precaution_dict.get(prediction, [])
    print("\nRecommended Precautions:")
    if precautions:
        for i, p in enumerate(precautions, 1):
            print(f"{i}. {p}")
    else:
        print("No specific precautions found.")
    
    create_pdf_report(prediction, user_symptoms, precautions)

def show_symptoms():
    print("\nAvailable Symptoms:")
    print(", ".join(sorted([s.strip() for s in mlb.classes_])))

def main():
    show_symptoms()
    user_input = input("\nEnter symptoms separated by commas: ").split(',')
    user_symptoms = [sym.strip().lower() for sym in user_input if sym.strip().lower() in mlb.classes_]

    if not user_symptoms:
        print("No valid symptoms entered. Please enter symptoms exactly as shown above.")
        return

    generate_report(user_symptoms)

if __name__ == "__main__":
    main()
