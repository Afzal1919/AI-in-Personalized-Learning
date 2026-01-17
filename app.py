import streamlit as st
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Exam Readiness Predictor", layout="centered")

st.title("📘 AI-Based Exam Readiness Prediction System")

st.write("Enter your mock test details to check exam readiness.")

# Input fields
avg_score = st.slider("Average Mock Test Score", 0, 100, 60)
topic_accuracy = st.slider("Topic-wise Accuracy (%)", 0, 100, 65)
revision_freq = st.slider("Revision Frequency (per week)", 1, 10, 4)
consistency = st.slider("Consistency Score", 0, 100, 70)

# Dummy trained model (same logic)
model = DecisionTreeClassifier(max_depth=5)
X_dummy = np.array([
    [80, 85, 7, 80],
    [60, 65, 4, 60],
    [40, 45, 2, 40]
])
y_dummy = ["Ready", "Needs Revision", "High Risk"]
model.fit(X_dummy, y_dummy)

if st.button("Predict Readiness"):
    input_data = np.array([[avg_score, topic_accuracy, revision_freq, consistency]])
    prediction = model.predict(input_data)[0]

    st.subheader(f"📊 Readiness Level: {prediction}")

    st.write("### Why this result?")
    st.write("- Based on mock test performance")
    st.write("- Consistency across attempts")
    st.write("- Revision frequency")


    if prediction == "Ready":
        st.success("You are exam-ready! Focus on mock tests.")
    elif prediction == "Needs Revision":
        st.warning("Revise weak topics and practice regularly.")
    else:
        st.error("High risk detected. Strengthen fundamentals immediately.")
