

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
chd_df = pd.read_csv("framingham.csv")
chd_df.drop(['education'], axis=1, inplace=True)
chd_df.rename(columns={'male': 'Sex_male'}, inplace=True)
chd_df.dropna(axis=0, inplace=True)

# Features and target
X = np.asarray(chd_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(chd_df['TenYearCHD'])

# Standardization
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=4)

# Train model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Streamlit UI
st.title("💓 10-Year Heart Disease Risk Predictor")

st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
cigsPerDay = st.sidebar.slider("Cigarettes Per Day", 0, 60, 10)
totChol = st.sidebar.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
sysBP = st.sidebar.slider("Systolic BP", 90, 250, 120)
glucose = st.sidebar.slider("Glucose (mg/dL)", 50, 300, 90)

# Prepare input
input_data = np.array([[age, 1 if sex == "Male" else 0, cigsPerDay, totChol, sysBP, glucose]])
input_scaled = scaler.transform(input_data)

# Predict
prediction = logreg.predict(input_scaled)
prediction_proba = logreg.predict_proba(input_scaled)

# Output
if prediction[0] == 1:
    st.error(f"⚠️ High risk of heart disease in 10 years ({round(prediction_proba[0][1]*100, 2)}%)")
else:
    st.success(f"✅ Low risk of heart disease in 10 years ({round(prediction_proba[0][1]*100, 2)}%)")

# Display model accuracy
st.write("### Model Accuracy on Test Set:")
y_pred = logreg.predict(X_test)
st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion matrix
st.write("### Confusion Matrix")
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
st.dataframe(cm)

# Classification Report
st.write("### Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
