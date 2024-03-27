from streamlit_extras.switch_page_button import switch_page 
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import zscore

st.set_page_config(page_title="Data Cleaning and Visualization", layout="wide")

if 'df' in st.session_state:
    unclean_df = st.session_state['df']
else:
    st.error("No file uploaded. Please go back and upload a file.")
    st.stop()

def load_and_clean_data(df):
    # Data cleaning steps
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
    df['brand'] = df['brand'].str.title()
    df['fuel_type'] = df['fuel_type'].str.capitalize()
    
    return df

df_cleaned = load_and_clean_data(unclean_df)

# Enhanced header and metadata display
st.markdown("## Data Summary (Cleaned)")
st.dataframe(df_cleaned.describe())

st.markdown("## NA Values in Cleaned Data")
st.dataframe(df_cleaned.isna().sum())

# Visualization Section with improved layout
st.markdown("## Visualizations for Cleaned Data")

def plot_distributions(df):
    col1, col2 = st.columns(2)
    with col1:
        fig_vehicle_age = px.histogram(df, x='vehicle_age', title='Distribution of Vehicle Ages')
        st.plotly_chart(fig_vehicle_age)
    with col2:
        fig_km_driven = px.histogram(df, x='km_driven', title='Distribution of Kilometers Driven')
        st.plotly_chart(fig_km_driven)

def plot_relationships(df):
    col1, col2 = st.columns(2)
    with col1:
        fig_mileage_price = px.scatter(df, x='mileage', y='selling_price', color='fuel_type', title='Mileage vs. Selling Price by Fuel Type')
        st.plotly_chart(fig_mileage_price)
    with col2:
        fig_brand_price = px.box(df, x='brand', y='selling_price', title='Selling Price by Brand')
        st.plotly_chart(fig_brand_price)

def plot_treemap(df):
    car_brand_counts = df['brand'].value_counts().reset_index()
    car_brand_counts.columns = ['brand', 'count']
    fig = px.treemap(car_brand_counts, path=['brand'], values='count', title='Number of Entries per Car Brand')
    st.plotly_chart(fig)

plot_treemap(df_cleaned)
plot_distributions(df_cleaned)
plot_relationships(df_cleaned)

# Navigation button with improved labeling
if st.button('Proceed to Modeling'):
    switch_page("page4nomodel")
