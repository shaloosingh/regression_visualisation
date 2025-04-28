
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title(" Multiple Linear Regression Tutorial")

# Introduction
st.write("""
## What is Multiple Linear Regression?
Multiple Linear Regression models the relationship between multiple independent variables (X1, X2, ...) and one dependent variable (y).
""")

st.write("---")

# Choose how to input data
st.header("1. Provide Your Data")

data_input_method = st.radio("Choose data input method:", ("Upload CSV file", "Manual Entry"))

if data_input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df)

        # Only keep numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.error("No numeric columns found in the file! Please upload a valid dataset.")
            st.stop()

        # Select columns for multiple features
        feature_columns = st.multiselect("Select feature columns (X):", numeric_df.columns)
        target_column = st.selectbox("Select target column (y):", numeric_df.columns)

        # Check if user selected enough columns
        if len(feature_columns) < 2:
            st.warning("Please select at least two feature columns for multiple regression.")
            st.stop()

        X = numeric_df[feature_columns].values
        y = numeric_df[target_column].values
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

else:
    st.subheader("Manual Entry")
    num_points = st.number_input("Enter the number of data points (rows):", min_value=2, max_value=100, value=5)

    num_features = st.number_input("Enter the number of features (independent variables):", min_value=2, max_value=10, value=2)

    feature_columns = [f"Feature {i+1}" for i in range(num_features)]
    X_values = []
    for i in range(num_features):
        feature_values = st.text_area(f"Enter values for {feature_columns[i]} (comma separated):", "1,2,3,4,5", key=f"feature_{i}")
        X_values.append(np.array([float(x.strip()) for x in feature_values.split(",")]))

    X = np.vstack(X_values).T
    y_values = st.text_area("Enter target values (y) separated by commas:", "2,4,6,8,10")
    y = np.array([float(y.strip()) for y in y_values.split(",")])

    st.write("Your input data:")
    st.write(pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=feature_columns + ['y']))

st.write("---")

# Visualize data (optional for 2D or 3D)
st.header("2. Visualize the Data")

fig, ax = plt.subplots()
if len(feature_columns) == 2:
    ax.scatter(X[:, 0], y, color='blue', label='Data Points')
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel('y')
    ax.set_title("Scatter Plot")
elif len(feature_columns) == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data Points')
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_zlabel('y')
    ax.set_title("3D Scatter Plot")
else:
    st.info("Visualization is only available for 2 or 3 features.")

st.pyplot(fig)
st.write("---")

# Train a Multiple Linear Regression model
st.header("3. Train Multiple Linear Regression Model")
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# Show coefficients
st.write("**Coefficients (Slopes)**:", model.coef_)
st.write(f"**Intercept (b₀):** {model.intercept_:.2f}")

# Plot data with predicted line (for 2D only)
if len(feature_columns) == 2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(X[:, 0], y, color='blue', label='Data Points')
    ax2.plot(X[:, 0], y_pred, color='red', label='Regression Line')
    ax2.set_xlabel(feature_columns[0])
    ax2.set_ylabel('y')
    ax2.set_title("Multiple Regression Fit (2D)")
    ax2.legend()
    st.pyplot(fig2)

st.write("---")

# Model performance
st.header("4. Model Evaluation")
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
st.metric(label="R-squared (R² Score)", value=f"{r2:.2f}")

st.write("""
- **MSE** measures the average squared difference between actual and predicted values.
- **R²** tells us how much of the variance in `y` is explained by `X`.
""")

st.write("---")

# Predict for new input (multiple features)
st.header("5. Predict with New Data")

new_input = []
for feature in feature_columns:
    feature_value = st.number_input(f"Enter a value for {feature}:", step=0.1)
    new_input.append(feature_value)

new_input = np.array(new_input).reshape(1, -1)
prediction = model.predict(new_input)
st.success(f"Predicted y value: {prediction[0]:.2f}")

st.write("---")



