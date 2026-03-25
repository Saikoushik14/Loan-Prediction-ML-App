import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("💼 Loan Prediction ML App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Raw Data")
    st.write(df.head())

    # Missing values
    st.subheader("🔍 Missing Values")
    st.write(df.isnull().sum())

    # Fill missing values
    for col in ['credit_score', 'income', 'loan_amount', 'annual_spend']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Outlier capping
    def cap_outlier(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower, upper)

    for col in ['credit_score', 'income', 'loan_amount', 'annual_spend']:
        if col in df.columns:
            cap_outlier(col)

    st.subheader("📦 Boxplot After Cleaning")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, ax=ax)
    st.pyplot(fig)

    # Encoding
    cat_cols = ['city', 'employment_type']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    if all(col in df.columns for col in cat_cols):
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)

    # Scaling
    if 'income' in df.columns and 'loan_amount' in df.columns:
        scaler = MinMaxScaler()
        df[['income', 'loan_amount']] = scaler.fit_transform(df[['income', 'loan_amount']])

    if 'credit_score' in df.columns:
        scaler_std = StandardScaler()
        df['credit_score'] = scaler_std.fit_transform(df[['credit_score']])

    st.subheader("✅ Processed Data")
    st.write(df.head())

    # Split
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
        y = df['target']

        X = X.select_dtypes(include='number')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.subheader("🤖 Model Training")

        model_option = st.selectbox(
            "Choose Model",
            ["KNN", "Linear Regression", "Decision Tree"]
        )

        if model_option == "KNN":
            k = st.slider("Select K value", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=k)

        elif model_option == "Linear Regression":
            model = LinearRegression()

        else:
            model = DecisionTreeRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success(f"📉 MSE: {mse}")
        st.success(f"📈 R2 Score: {r2}")

        # Elbow Plot (only for KNN)
        if model_option == "KNN":
            mse_values = []
            k_values = range(1, 21)

            for k in k_values:
                knn = KNeighborsRegressor(n_neighbors=k)
                knn.fit(X_train, y_train)
                pred = knn.predict(X_test)
                mse_values.append(mean_squared_error(y_test, pred))

            fig2, ax2 = plt.subplots()
            ax2.plot(k_values, mse_values, marker='o')
            ax2.set_xlabel("K Value")
            ax2.set_ylabel("MSE")
            ax2.set_title("Elbow Method")
            st.pyplot(fig2)

    else:
        st.warning("⚠️ 'target' column not found in dataset")