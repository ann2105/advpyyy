import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("🎓 Predict Student Performance")

# Load data
df = pd.read_csv("student_mat.csv")
st.write("### Dataset Sample", df.head())

# Visualize
st.write("### Study Time vs Final Grade")
sns_plot = sns.boxplot(x="study_time", y="G3", data=df)
st.pyplot(sns_plot.figure)

# Prepare data
X = df[["study_time", "failures", "absences", "G1", "G2"]]
y = df["passed"].map({"yes": 1, "no": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
st.write("### Accuracy:", accuracy_score(y_test, y_pred))

# Predict user input
st.write("### Try it Yourself:")
study_time = st.slider("Study Time", 1, 4, 2)
failures = st.slider("Failures", 0, 4, 0)
absences = st.slider("Absences", 0, 30, 5)
G1 = st.slider("G1 Grade", 0, 20, 10)
G2 = st.slider("G2 Grade", 0, 20, 10)

input_data = pd.DataFrame([[study_time, failures, absences, G1, G2]], columns=X.columns)
prediction = model.predict(input_data)

if prediction[0]:
    st.success("🎉 The student is likely to PASS.")
else:
    st.error("⚠️ The student is likely to FAIL.")
