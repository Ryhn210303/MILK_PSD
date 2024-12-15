import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load dataset
@st.cache_data
def load_data():
    # Replace 'milknew.csv' with the actual path to your dataset
    data = pd.read_csv('milknew.csv')
    data = data[data['Grade'].isin(['high', 'low'])]
    label_binarizer = LabelBinarizer()
    data['Grade'] = label_binarizer.fit_transform(data['Grade'])
    return data

data = load_data()

# Features and target separation
X = data.drop(columns=['Grade'])
y = data['Grade']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Streamlit App
def main():
    st.title("Milk Quality Prediction")
    st.write("Input milk data to predict whether the quality is High or Low.")

    # Input features
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, key="pH")
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=100, step=1, key="temperature")
    taste = st.selectbox("Taste", ["Good", "Bad"], key="taste")
    odor = st.selectbox("Odor", ["Pleasant", "Unpleasant"], key="odor")
    fat = st.selectbox("Fat Content", ["Present", "Absent"], key="fat")
    turbidity = st.selectbox("Turbidity", ["Clear", "Turbid"], key="turbidity")
    colour = st.slider("Colour Intensity", min_value=0, max_value=255, step=1, key="colour")

    # Encode categorical inputs
    taste = 1 if taste == "Good" else 0
    odor = 1 if odor == "Pleasant" else 0
    fat = 1 if fat == "Present" else 0
    turbidity = 1 if turbidity == "Clear" else 0

    # Prepare input for prediction
    input_features = np.array([[pH, temperature, taste, odor, fat, turbidity, colour]])

    # Prediction
    if st.button("Predict with Naive Bayes", key="nb_button"):
        prediction = nb_model.predict(input_features)[0]
        result = "High" if prediction == 1 else "Low"
        st.success(f"Naive Bayes Prediction: The milk quality is {result}.")

    if st.button("Predict with Decision Tree", key="dt_button"):
        prediction = dt_model.predict(input_features)[0]
        result = "High" if prediction == 1 else "Low"
        st.success(f"Decision Tree Prediction: The milk quality is {result}.")

if __name__ == "__main__":
    main()