import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Title for the web app
st.title('Rock and Mine Prediction App')

# Load dataset and model
@st.cache
def load_data():
    df = pd.read_csv('sonar data.csv')
    df.rename(columns={df.columns[60]: 'Target'}, inplace=True)
    return df

df = load_data()

# Show a sample of the data
if st.checkbox('Show Data Sample'):
    st.write(df.head())

# Separate features and target variable
X = df.drop(columns='Target')
Y = df['Target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, Y_train)

# Accuracy on the training and testing data
X_train_pred = log_reg_model.predict(X_train)
training_accuracy = accuracy_score(X_train_pred, Y_train)

X_test_pred = log_reg_model.predict(X_test)
test_accuracy = accuracy_score(X_test_pred, Y_test)

# Show the model accuracy on training and test data
st.write(f'Logistic Regression Model Accuracy on Training Data: {training_accuracy:.2f}')
st.write(f'Logistic Regression Model Accuracy on Test Data: {test_accuracy:.2f}')

# Train a Neural Network model
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(X_train, Y_train)

# Accuracy for MLP
mlp_pred = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(Y_test, mlp_pred)

st.write(f'MLP Model Accuracy on Test Data: {mlp_accuracy:.2f}')

# Create a form for users to input new data for prediction
st.sidebar.header('User Input Features')

def user_input_features():
    features = {}
    for col in X.columns:
        features[col] = st.sidebar.number_input(col, min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))
    return pd.DataFrame(features, index=[0])

user_input = user_input_features()

# Predict using Logistic Regression
log_reg_prediction = log_reg_model.predict(user_input)
st.sidebar.subheader('Logistic Regression Prediction')
st.sidebar.write(log_reg_prediction)

# Predict using MLP
mlp_prediction = mlp_model.predict(user_input)
st.sidebar.subheader('MLP Prediction')
st.sidebar.write(mlp_prediction)

