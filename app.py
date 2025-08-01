import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px

# Title
st.title("🎓 Student Performance Predictor")

# Load dataset
df = pd.read_csv("student_mat.csv")
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Basic stats
st.subheader("📊 Feature Distribution")
numeric_cols = df.select_dtypes(include='number').columns
selected_col = st.selectbox("Select column to visualize", numeric_cols)
fig = px.histogram(df, x=selected_col)
st.plotly_chart(fig)

# Preprocessing
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
X = df.drop(['G3', 'pass'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['pass']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
st.subheader("📈 Model Performance")
st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Prediction
st.subheader("🔮 Predict Student Outcome")
input_data = {}
for col in X.columns:
    if df[col].dtype == 'int64':
        input_data[col] = st.number_input(f"{col}", value=int(df[col].mean()))
    else:
        input_data[col] = st.selectbox(f"{col}", options=df[col].unique())

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
result = "✅ Pass" if prediction == 1 else "❌ Fail"
st.success(f"Prediction: {result}")
