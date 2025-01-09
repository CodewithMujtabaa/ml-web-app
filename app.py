import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sidebar to select dataset
st.title("Machine Learning Web App by Mujtaba")
st.sidebar.header("User Options")
dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer"])

# Load dataset
def load_dataset(name):
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Breast Cancer":
        data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="Target")

X, y = load_dataset(dataset_name)
st.write(f"Dataset: {dataset_name}")
st.write("Shape of the dataset:", X.shape)
st.write("First 5 rows of the dataset:")
st.dataframe(X.head())

# Visualizations
if st.checkbox("Show Pairplot"):
    sns.pairplot(pd.concat([X, y], axis=1), hue="Target")
    st.pyplot()

if st.checkbox("Show Correlation Heatmap"):
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot()

# Model selection
model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])

# Get model
def get_model(model_name):
    if model_name == "Logistic Regression":
        C = st.sidebar.slider("C (Inverse of Regularization Strength)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C)
    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 200, 50)
        max_depth = st.sidebar.slider("max_depth", 1, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif model_name == "SVM":
        C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVC(C=C, kernel=kernel)
    return model

model = get_model(model_name)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write(f"Accuracy of {model_name}:", accuracy_score(y_test, y_pred))

# User input for prediction
st.sidebar.header("Input Features for Prediction")
input_data = [st.sidebar.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean())) for col in X.columns]

# Prediction
prediction = model.predict([input_data])
st.write("Prediction for input data:", prediction[0])
