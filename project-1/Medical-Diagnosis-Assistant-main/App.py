import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
import os
import base64


def generate_pretty_pdf_report(username, selected_symptoms, disease, description, precautions):
    if not os.path.exists("reports"):
        os.makedirs("reports")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{username}_{timestamp}.pdf"
    file_path = f"reports/{filename}"

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("🩺 Medical Diagnosis Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Patient Name:</b> {username}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Selected Symptoms:</b>", styles['Heading2']))
    story.append(ListFlowable(
        [ListItem(Paragraph(sym, styles['Normal'])) for sym in selected_symptoms],
        bulletType='bullet'
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Predicted Disease:</b> {disease}", styles['Heading2']))
    story.append(Spacer(1, 8))

    if description:
        story.append(Paragraph("<b>Disease Description:</b>", styles['Heading2']))
        story.append(Paragraph(description, styles['Normal']))
        story.append(Spacer(1, 12))

    if precautions:
        story.append(Paragraph("<b>Precautions:</b>", styles['Heading2']))
        story.append(ListFlowable(
            [ListItem(Paragraph(p, styles['Normal'])) for p in precautions],
            bulletType='bullet'
        ))
        story.append(Spacer(1, 12))

    doc.build(story)
    return file_path

# Load the datasets
desc_df = pd.read_csv("Datasets/symptom_Description.csv")
prec_df = pd.read_csv("Datasets/symptom_precaution.csv")

# Load data
df_cleaned = pd.read_csv('Models/Feature 1 ( Disease prediction and report generation)/cleaned_dataset.csv')
df_desc = pd.read_csv('Models/Feature 1 ( Disease prediction and report generation)/symptom_Description.csv')
df_prec = pd.read_csv('Models/Feature 1 ( Disease prediction and report generation)/symptom_precaution.csv')

# Load model and MultiLabelBinarizer
rf_model = joblib.load("Models/Feature 1 ( Disease prediction and report generation)/rf_model.joblib")
mlb = joblib.load("Models/Feature 1 ( Disease prediction and report generation)/mlb.joblib")


# Get all possible symptoms from cleaned dataset
# Extract all unique symptoms from the dataset
symptom_columns = df_cleaned.columns[df_cleaned.columns != 'Disease']
all_symptoms = pd.unique(df_cleaned[symptom_columns].values.ravel())
all_symptoms = sorted([sym for sym in all_symptoms if pd.notna(sym)])

@st.cache_data
def load_prediction_datasets():
    df_cleaned = pd.read_csv('Models/Feature 1 ( Disease prediction and report generation)/cleaned_dataset.csv')
    df_desc = pd.read_csv('Datasets/symptom_Description.csv')
    df_severity = pd.read_csv('Datasets/symptom-severity.csv')
    df_prec = pd.read_csv('Datasets/symptom_precaution.csv')
    return df_cleaned, df_desc, df_severity, df_prec

df_cleaned, df_desc, df_severity, df_prec = load_prediction_datasets()

# Load disease description and precautions
@st.cache_data
def load_disease_info():
    desc_df = pd.read_csv('Datasets/symptom_Description.csv')
    prec_df = pd.read_csv('Datasets/symptom_precaution.csv')
    return desc_df, prec_df
   
desc_df, prec_df = load_disease_info()

# Function to fetch news
def fetch_news(api_key, query, language='en', page_size=10):
    url = f'https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}'
    response = requests.get(url)
    return response.json()

# Your News API key
api_key = 'cd52220d58f2490fb62f64dfc0435e9c'

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a section:", 
                          ["Home", "Disease Prediction", "Diabetes Prediction", "Heart Disease Prediction", "Reports", "Disease Info"])

# Header and greeting
st.markdown("# Medical Diagnosis Assistant")

now = datetime.now()
current_hour = now.hour
if 5 <= current_hour < 12:
    greeting = "Good Morning"
elif 12 <= current_hour < 17:
    greeting = "Good Afternoon"
elif 17 <= current_hour < 21:
    greeting = "Good Evening"
else:
    greeting = "Good Night"

st.write(f"## {greeting}!")
st.write(f"**Date:** {now.strftime('%Y-%m-%d')}")
st.write(f"**Time:** {now.strftime('%H:%M:%S')}")

# Main Sections
if option == "Home":
    # Fetch healthcare and AI news
    healthcare_news = fetch_news(api_key, 'healthcare')
    ai_news = fetch_news(api_key, 'artificial intelligence')

    # Display healthcare news
    st.markdown("### 🏥 Healthcare News")
    for article in healthcare_news.get('articles', []):
        st.subheader(article['title'])
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
        st.write("---")

    # Display AI news
    st.markdown("### 🤖 AI News")
    for article in ai_news.get('articles', []):
        st.subheader(article['title'])
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
        st.write("---")

elif option == "Disease Prediction":
    
    st.markdown("## 📊 Disease Prediction Dashboard")

    # Summary Metrics
    st.write("### Summary")
    st.metric("Total Diseases", df_desc['Disease'].nunique())
    st.metric("Total Symptoms", df_cleaned.columns[:-1].nunique())  # Exclude target column

    # Charts
    st.write("### Symptom Frequency")
    symptom_counts = df_cleaned.drop('Disease', axis=1).apply(pd.Series.value_counts).sum(axis=1)
    top_symptoms = symptom_counts.sort_values(ascending=False).head(15)

    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_symptoms.values, y=top_symptoms.index, ax=ax1, palette="viridis")
    ax1.set_title("Top 15 Frequent Symptoms")
    st.pyplot(fig1)

    st.write("### Symptom Severity Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_severity['weight'], bins=10, kde=True, color="skyblue", ax=ax2)
    ax2.set_title("Symptom Severity (Weight)")
    st.pyplot(fig2)

    st.write("### Precaution Coverage")
    precaution_count = df_prec.set_index('Disease').notna().sum(axis=1)
    top_precaution = precaution_count.sort_values(ascending=False).head(15)

    fig3, ax3 = plt.subplots()
    top_precaution.plot(kind='barh', ax=ax3, color='salmon')
    ax3.set_title("Top 15 Diseases with Most Precautions")
    st.pyplot(fig3)

    # Dataset samples
    st.write("### Sample from Datasets")
    st.subheader("Cleaned Dataset")
    st.dataframe(df_cleaned.head(5))

    st.subheader("Symptom Description")
    st.dataframe(df_desc.head(5))

    st.subheader("Symptom Severity")
    st.dataframe(df_severity.head(5))

    st.subheader("Symptom Precaution")
    st.dataframe(df_prec.head(5))

    st.markdown("## 🩺 Predict Disease from Symptoms")
    selected_symptoms = st.multiselect("Select Symptoms:", all_symptoms)
    st.write("You can select multiple symptoms. The model will predict the disease based on the selected symptoms.")
    username = st.text_input("Enter your name for the report:")

    if st.button("Predict Disease"):
     if selected_symptoms and username.strip():
        input_transformed = mlb.transform([selected_symptoms])
        prediction = rf_model.predict(input_transformed)[0]

        st.success(f"### 🧬 Predicted Disease: `{prediction}`")

        description_row = desc_df[desc_df['Disease'].str.lower() == prediction.lower()]
        description = description_row['Description'].values[0] if not description_row.empty else "No description available."

        # Fetch precautions (if available)
        precaution_row = prec_df[prec_df['Disease'].str.lower() == prediction.lower()]
        precautions = []

        if not precaution_row.empty:
            for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                val = precaution_row[col].values[0]
                if pd.notna(val):
                    precautions.append(val)
        else:
            precautions = ["No precautions available."]

        # Generate PDF report
        pdf_path = generate_pretty_pdf_report(username, selected_symptoms, prediction, description, precautions)
        st.success(f"PDF report saved: `{pdf_path}`")

    else:
        st.warning("Please enter your name and select at least one symptom.")

elif option == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction")
    st.markdown("## 🧪 Diabetes Prediction")

    # Load dataset
    diabetes_df = pd.read_csv("Datasets/diabetes.csv")
    st.subheader("📊 Dataset Overview")
    st.dataframe(diabetes_df.head())

    # Charts and dashboard
    st.subheader("🔍 Exploratory Data Analysis")

    # Outcome distribution
    fig1, ax1 = plt.subplots()
    diabetes_df['Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=["No Diabetes", "Diabetes"], colors=['#66b3ff','#ff9999'], ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title("Diabetes Outcome Distribution")
    st.pyplot(fig1)

    # Correlation heatmap
    st.subheader("📈 Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(diabetes_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    st.pyplot(fig2)

    # Feature distributions
    st.subheader("📌 Feature Distributions")
    selected_feature = st.selectbox("Choose feature to visualize", diabetes_df.columns[:-1])
    fig3, ax3 = plt.subplots()
    sns.histplot(diabetes_df[selected_feature], kde=True, ax=ax3, color='skyblue')
    st.pyplot(fig3)

    st.markdown("---")

    # Prediction input
    st.subheader("🔮 Predict Diabetes")
    username = st.text_input("Enter your name:")

    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0)

    if st.button("📌 Predict Diabetes"):
        try:
            # Load model and scaler
            scaler = joblib.load("Models/Feature 2 (Diabetes prediction)/scaler.joblib")
            model = joblib.load("Models/Feature 2 (Diabetes prediction)/diabetes_rf_model.joblib")

            input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
            st.success(f"🩺 Prediction: {result}")

            # Build PDF content
            symptoms_used = [
                f"Pregnancies: {pregnancies}",
                f"Glucose: {glucose}",
                f"Blood Pressure: {bp}",
                f"Skin Thickness: {skin_thickness}",
                f"Insulin: {insulin}",
                f"BMI: {bmi}",
                f"Diabetes Pedigree Function: {dpf}",
                f"Age: {age}"
            ]

            # Optionally set suggestions
            precautions = []
            if prediction == 1:
                precautions = [
                    "Follow a healthy diet low in sugar.",
                    "Exercise regularly (at least 30 mins/day).",
                    "Monitor blood sugar regularly.",
                    "Consult your healthcare provider."
                ]
            else:
                precautions = [
                    "Maintain a balanced diet.",
                    "Stay physically active.",
                    "Have annual diabetes checkups."
                ]

            description = "Diabetes is a chronic condition that affects how your body turns food into energy."

            # Generate report
            pdf_path = generate_pretty_pdf_report(username, symptoms_used, result, description, precautions)

            st.success(f"📄 Report saved: `{os.path.basename(pdf_path)}`")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


elif option == "Heart Disease Prediction":
    st.markdown("## ❤️ Heart Disease Prediction")

    # Load dataset
    heart_df = pd.read_csv("Datasets/heart.csv")
    st.subheader("📊 Dataset Overview")
    st.dataframe(heart_df.head())

    # Charts and dashboard
    st.subheader("🔍 Exploratory Data Analysis")

    # Target distribution
    fig1, ax1 = plt.subplots()
    heart_df['target'].value_counts().plot(kind='pie', labels=["No Disease", "Disease"], autopct='%1.1f%%', startangle=90, colors=["#99ff99", "#ff6666"], ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title("Heart Disease Distribution")
    st.pyplot(fig1)

    # Correlation heatmap
    st.subheader("📈 Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heart_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # Feature distribution
    st.subheader("📌 Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize", heart_df.columns[:-1])
    fig3, ax3 = plt.subplots()
    sns.histplot(heart_df[selected_feature], kde=True, ax=ax3, color='lightcoral')
    st.pyplot(fig3)

    st.markdown("---")

    # Prediction Input
    st.subheader("🔮 Predict Heart Disease")

    username = st.text_input("Enter your name:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP", min_value=0)
        chol = st.number_input("Cholesterol", min_value=0)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", min_value=0)
        exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])

    with col3:
        oldpeak = st.number_input("Oldpeak", format="%.2f")
        slope = st.selectbox("Slope of ST segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

    if st.button("📌 Predict Heart Disease"):
        try:
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]])

            # Load model and scaler
            scaler = joblib.load("Models/Feature 3 (Heart Disease Prediction)/scaler.pkl")
            model = joblib.load("Models/Feature 3 (Heart Disease Prediction)/heart_model.pkl")

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            result = "At Risk of Heart Disease" if prediction == 1 else "Not at Risk"
            st.success(f"🩺 Prediction: {result}")

            # Build PDF report
            symptoms_used = [
                f"Age: {age}", f"Sex: {sex}", f"Chest Pain Type: {cp}", f"Resting BP: {trestbps}",
                f"Cholesterol: {chol}", f"Fasting Blood Sugar: {fbs}", f"Resting ECG: {restecg}",
                f"Max Heart Rate: {thalach}", f"Exercise Angina: {exang}", f"Oldpeak: {oldpeak}",
                f"Slope: {slope}", f"CA: {ca}", f"Thal: {thal}"
            ]

            description = "Heart disease refers to various types of heart conditions that can lead to heart attacks or other complications."

            precautions = []
            if prediction == 1:
                precautions = [
                    "Avoid high cholesterol/saturated fat.",
                    "Exercise regularly (30 mins/day).",
                    "Monitor blood pressure and heart rate.",
                    "Avoid smoking and alcohol.",
                    "Consult a cardiologist regularly."
                ]
            else:
                precautions = [
                    "Maintain a healthy diet and weight.",
                    "Stay physically active.",
                    "Get regular health checkups."
                ]

            pdf_path = generate_pretty_pdf_report(username, symptoms_used, result, description, precautions)
            st.success(f"📄 Report saved: `{os.path.basename(pdf_path)}`")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif option == "Reports":
    st.title("📁 Patient Reports")
    report_files = [f for f in os.listdir("reports") if f.endswith(".pdf")]

    if not report_files:
        st.info("No reports available.")
    else:
        for file in sorted(report_files, reverse=True):
            filepath = os.path.join("reports", file)
            with open(filepath, "rb") as f:
                PDFbyte = f.read()

            st.subheader(file.replace(".pdf", ""))
            st.download_button(
                label="⬇️ Download PDF",
                data=PDFbyte,
                file_name=file,
                mime='application/pdf',
                key=file
            )

            # Preview in app
            base64_pdf = base64.b64encode(PDFbyte).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.markdown("---")
    st.write("### Patient Reports")

elif option == "Disease Info":
    st.write("### Disease Information")

    # Get the intersection of diseases present in both datasets
    diseases = sorted(list(set(desc_df['Disease']) & set(prec_df['Disease'])))

    selected_disease = st.selectbox("Select a disease:", diseases)

    if selected_disease:
        st.markdown(f"## {selected_disease}")

        # Description
        desc_row = desc_df[desc_df['Disease'] == selected_disease]
        if not desc_row.empty:
            st.markdown(f"**Description:** {desc_row['Description'].values[0]}")

        # Precautions
        prec_row = prec_df[prec_df['Disease'] == selected_disease]
        if not prec_row.empty:
            st.markdown("**Precautions:**")
            for i in range(1, 5):  # Precaution_1 to Precaution_4
                precaution = prec_row[f'Precaution_{i}'].values[0]
                if pd.notna(precaution):
                    st.write(f"- {precaution}")
        else:
            st.write("No precautions available for this disease.")
