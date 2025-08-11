import streamlit as st
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("Student Performance Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("Student.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # 1. Mean, Median, Mode
    mean_marks = df["Marks"].mean()
    median_marks = df["Marks"].median()
    mode_marks = stats.mode(df["Marks"], keepdims=True).mode[0]

    st.subheader("Statistics")
    st.write(f"**Mean Marks:** {mean_marks:.2f}")
    st.write(f"**Median Marks:** {median_marks}")
    st.write(f"**Mode Marks:** {mode_marks}")

    # 2. Correlation
    correlation = df["Study_Hours"].corr(df["Marks"])
    st.write(f"**Correlation between study hours & marks:** {correlation:.2f}")

    # 3. Top 5% Performers
    top_5_percent = df.nlargest(int(len(df) * 0.05), "Marks")
    st.subheader("Top 5% Performers")
    st.dataframe(top_5_percent)

    # 4. Linear Regression Prediction
    X = df[["Study_Hours"]]
    y = df["Marks"]

    model = LinearRegression()
    model.fit(X, y)

    study_hours_input = st.number_input("Enter study hours to predict marks:", min_value=0.0, max_value=15.0, step=0.5)
    if study_hours_input:
        predicted_score = model.predict([[study_hours_input]])[0]
        st.write(f"📌 Predicted score for {study_hours_input} hours/day: **{predicted_score:.2f}**")

    # 5. Visualization
    st.subheader("Scatter Plot with Regression Line")
    fig, ax = plt.subplots()
    sns.regplot(x="Study_Hours", y="Marks", data=df, ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start the analysis.")
