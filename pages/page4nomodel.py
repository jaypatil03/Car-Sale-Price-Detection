import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import zscore
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from keras.models import load_model

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>Model Selection</h1>", unsafe_allow_html=True)

# @st.cache(allow_output_mutation=True)
def load_models():
    nn_model = load_model('car_price_prediction_model_neural_network_new.h5')
    rf_model = joblib.load("car_price_prediction_model_random_forest.sav")
    xgb_model = joblib.load('car_price_prediction_model_xgboost_new.sav')
    return nn_model, rf_model, xgb_model

nn_model, rf_model, xgb_model = load_models()

def load_and_clean_data(file_path='./cardekho_dataset_unclean.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Data cleaning steps as before
        for column in ['model', 'seller_type', 'mileage', 'engine']:
            if df[column].dtype == 'object':  # Categorical column
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:  # Numerical column
                df[column].fillna(df[column].median(), inplace=True)
        df.drop_duplicates(inplace=True)
        
        # Removing outliers
        def remove_outliers(df, column, z_threshold=3):
            z_scores = zscore(df[column])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < z_threshold)
            return df[filtered_entries]
        
        df = remove_outliers(df, 'km_driven')
        df = remove_outliers(df, 'selling_price')
        df['brand'] = df['brand'].str.title()  # Capitalize first letter of each word
        df['fuel_type'] = df['fuel_type'].str.capitalize()  # Capitalize only the first letter
        return df
    else:
        return None

df = load_and_clean_data()

if df is not None:
    model_option = st.selectbox("Select an option", ["Neural Network", "Random Forest", "XGBoost"])

    if st.button('Perform training'):
        # Assuming 'clean_df' is your cleaned DataFrame
        def display_model_performance(model):
            predictions = model.predict(X_train_transformed)
            if model_option == 'Neural Network':  # NN model predicts in different shape
                predictions = predictions.flatten()
            mse = mean_squared_error(y_train, predictions)
            mae = mean_absolute_error(y_train, predictions)
            rmse = np.sqrt(mse)
            r2= r2_score(y_train, predictions)
            st.write(f'RMSE: {rmse}, MAE: {mae}, MSE: {mse}, R2 Score: {r2}')
            

        def preprocess_input():
            clean_df = df
            numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
            categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']

            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            X = clean_df.drop('selling_price', axis=1)
            y = clean_df['selling_price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            X_train_transformed = preprocessor.fit_transform(X_train).toarray()
            X_test_transformed = preprocessor.transform(X_test).toarray()
            df1 = pd.DataFrame(X_test_transformed, columns=preprocessor.get_feature_names_out())
            df1[len(df1.columns)] = y_test.tolist()
            df1.to_csv('test_dataset.csv', index=False)
            
            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor

        X_train_transformed, X_test_transformed, y_train, y_test, preprocessor = preprocess_input()

        if model_option == "Neural Network":
            # Your neural network code here
            # Make sure to adjust 'clean_df' as per your actual cleaned DataFrame variable
            st.success("Neural Network training completed.")
            st.session_state['Model'] = nn_model
            display_model_performance(nn_model)

            
            
            # Placeholder for Random Forest and XGBoost
        elif model_option == "Random Forest":
            st.success("Random Forest model completed.")
            st.session_state['Model'] = rf_model
            display_model_performance(rf_model)

            
            # Include Random Forest training code here
        elif model_option == "XGBoost":
            st.success("XGBoost model completed.")
            st.session_state['Model'] = xgb_model
            display_model_performance(xgb_model)
            # Include XGBoost training code here

        # Add buttons or additional logic for visualizing model performance
else:
    st.error("No dataset found. Please make sure the dataset is named 'cardekho_dataset_unclean.csv' and uploaded correctly.")

if st.button("Go to model testing"):
    switch_page("page5")