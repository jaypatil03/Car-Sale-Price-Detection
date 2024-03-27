import streamlit as st
from streamlit_extras.switch_page_button import switch_page 
import os 
import pandas as pd

st.set_page_config(page_title="Car Price Prediction", layout="wide", initial_sidebar_state="collapsed", page_icon=":car:")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    <h1 style='text-align: center; color: navy;'>Car Selling Price Prediction</h1>
    """, unsafe_allow_html=True)

def save_uploaded_file(uploadedfile): 
    with open(os.path.join("./", uploadedfile.name), "wb") as f: 
        f.write(uploadedfile.getbuffer()) 
        return os.path.join("./", uploadedfile.name) 

col1, col2 = st.columns([3, 2]) 

with col1: 
    st.markdown("<div class='big-font'>Project Details</div>", unsafe_allow_html=True)
    st.markdown("""
                The aim of this project is to develop a robust car price prediction system leveraging machine learning algorithms. 
                """)
    st.markdown("""
                By utilizing historical data on car sales, along with information on market trends and economic factors, the system will be trained to learn patterns and relationships within the data to make precise price predictions.
                """)
    with st.expander("Learn more about our approaches"):
        st.markdown("**Random Forest Approach:** An ensemble learning technique where multiple decision trees are constructed during training.")
        st.markdown("**XGBoost Approach:** A powerful gradient boosting algorithm known for its efficiency and accuracy.")
        st.markdown("**Neural Networks Approach:** Utilizes deep learning architectures to capture intricate nonlinear relationships within the data.")

with col2:  
    st.write("## Upload a CSV file:") 
    uploaded_file = st.file_uploader("", type=["csv"]) 
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file) 
        st.success(f"File saved to: {file_path}")
        df = pd.read_csv(file_path)
        st.session_state["df"] = df
        if st.button("Data Visualization"): 
            switch_page("page2")
