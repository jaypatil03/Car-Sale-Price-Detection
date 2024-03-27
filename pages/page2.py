from streamlit_extras.switch_page_button import switch_page 
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Setting the page configuration for a uniform look across your application
st.set_page_config(page_title="Data Visualizations", layout="wide")

# Improved header styling
st.markdown("""
    <style>
    .header-style {
        font-size:24px;
        font-weight: bold;
        color: navy;
    }
    </style>
    """, unsafe_allow_html=True)

if 'df' in st.session_state:
    df = st.session_state['df']  # Accessing the dataframe stored in session state
else:
    st.error("No file uploaded. Please go back and upload a file.")
    st.stop()

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

# Displaying the summary statistics of the dataset in a cleaner way
st.write("## NA Values:")
st.write(df.isna().sum())
st.write("## Data Summary:")
st.write(df.describe())

# Visualization Section with improved header
st.markdown('<div class="header-style">Visualizations for Uploaded Data:</div>', unsafe_allow_html=True)
plot_treemap(df)
plot_distributions(df)
plot_relationships(df)

# Enhanced navigation button
if st.button('Go to Data Cleaning Page'):
    switch_page("page3")
