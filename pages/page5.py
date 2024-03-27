import streamlit as st
import plotly.express as px
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
if 'Model' in st.session_state:
    model = st.session_state['Model']
else:
    st.error("No Model Selected. Please go back and select a model.")
def save_uploaded_file(uploadedfile): 
    with open(os.path.join("./", uploadedfile.name), "wb") as f: 
        f.write(uploadedfile.getbuffer()) 
        return os.path.join("./", uploadedfile.name) 

st.write("Upload a CSV file:") 
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"]) 
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file) 
    st.success(f"File saved to: {file_path}")
    df = pd.read_csv(file_path)

if st.button('Perform testing'):
        # Assuming 'clean_df' is your cleaned DataFrame
        def display_model_performance():
            predictions = model.predict(X_test_transformed)
            predictions = predictions.flatten()
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            rmse = np.sqrt(mse)
            r2= r2_score(y, predictions)
            errors = predictions - y
            st.write(f'RMSE: {rmse}, MAE: {mae}, MSE: {mse}, R2 Score: {r2}')
            return mae, mse, rmse, r2,errors,predictions
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
            X = clean_df.iloc[:, :-1]
            y = clean_df.iloc[:, -1]
            X_test_transformed = X
            return X_test_transformed,y, preprocessor
        X_test_transformed, y, preprocessor = preprocess_input()
        mae, mse, rmse, r2_score, errors,predictions = display_model_performance()
        st.session_state['predictions'] = predictions
        st.session_state['y'] = y
        def generate_dynamic_insights(mae, mse, r2_score):
            insights_text = f"""
            ### Inference from Model Performance Metrics
            The output of your model indicates its performance on the test set in terms of mean absolute error (MAE), mean squared error (MSE), and the R2 score. Let's break down what each of these metrics tells us about the model's performance:

            **Mean Absolute Error (MAE): {mae:,.2f}**
            MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

            A MAE of {mae:,.2f} means that, on average, the model's predictions are about {mae:,.2f} units away from the actual values. The units depend on what you're predicting (e.g., dollars, if it's a selling price).

            **Mean Squared Error (MSE): {mse:,.2f}**
            MSE measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. MSE is more sensitive to outliers than MAE because it squares the errors before averaging them, which disproportionately increases the influence of large errors.

            An MSE of {mse:,.2f} indicates the model's predictions deviate from the actual values significantly, with a greater penalty on large errors. This large value suggests that while the model might be performing well on many predictions, there are likely some predictions with substantial errors.
            **R2 Score: {r2_score:.2f}**
            The R2 score, also known as the coefficient of determination, is a statistical measure of how well the regression predictions approximate the real data points. An R2 score of 1 indicates perfect correlation, while an R2 score of 0 would indicate that the model fails to capture any variance in the data.

            An R2 score of {r2_score:.2f} is quite high, suggesting that the model explains about {r2_score*100:.0f}% of the variance in the target variable, making it a strong model in terms of its ability to predict new, unseen data.

            **Interpreting the Results**
            The high R2 score suggests that your model is quite effective in capturing the variance in the selling price. It indicates a strong fit to the data.
            The MAE and MSE values provide a sense of the scale of the errors your model is making. The relatively high values of MAE and MSE might indicate that there are instances where the model's predictions are significantly off. It's common in real-world datasets for a few challenging outliers or complex patterns to result in larger errors.
            Considering these metrics together, your model seems to perform well overall, but there may be room for improvement, especially in handling outliers or specific segments of the data that might be harder to predict accurately.
            """
            return insights_text
        st.markdown(generate_dynamic_insights(mae, mse, r2_score))
def plot_predictions(y, predictions):
    fig = px.scatter(x=y, y=predictions, labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
                title='Actual vs. Predicted Selling Prices')
    fig.add_shape(type='line', line=dict(dash='dash'), x0=min(y), y0=min(y), x1=max(y), y1=max(y))
    st.plotly_chart(fig)

def hist_errors(errors):
    fig = px.histogram(errors, nbins=50, labels={'value': 'Prediction Error'}, 
            title='Distribution of Prediction Errors')
    st.plotly_chart(fig)

# Check if the button is pressed and if the required data is available in session state
if 'y' in st.session_state and 'predictions' in st.session_state:
    if st.button('Generate Plots'):
        y = st.session_state['y']
        predictions = st.session_state['predictions']
        # Ensure errors are calculated from the actual values and predictions
        errors = y - predictions

        plot_predictions(y, predictions)
        hist_errors(errors)
        st.markdown("""
            
            ### Top Plot: Actual vs. Predicted Selling Prices

            - This scatter plot compares the actual selling prices on the x-axis with the predicted selling prices on the y-axis.
            - The dashed line represents the line of perfect prediction. Points along this line are where the model's predictions exactly match the actual values.
            - The plot shows a general agreement between the actual and predicted values, especially for lower-priced items, as indicated by the concentration of points along the line of perfect prediction.
            - There is more variance in the model's predictions as the actual selling price increases. This is visible through the vertical spread of points as you move right along the x-axis, which could indicate that the model may not be as accurate for higher-priced items.

            ### Bottom Plot: Distribution of Prediction Errors

            - This histogram shows the distribution of the prediction errors, which are calculated as the actual selling price minus the predicted selling price.
            - The majority of predictions are clustered around zero, which indicates that for many predictions, the model was quite accurate.
            - There's a noticeable right skew in the distribution of errors, meaning there are more instances where the model under-predicted the selling price than over-predicted.
            - The presence of errors farther from zero (especially towards the right) suggests some instances of significant underprediction, which could be outliers or instances where the model's assumptions do not hold.

            ### Combined Inference

            - The model appears to be reasonably accurate for a significant portion of the dataset, especially for items with lower selling prices.
            - The increased variance and skew in errors for higher-priced items suggest that the model's features and patterns learned from the training data are less effective at capturing the factors that contribute to higher selling prices.
            - Improving model performance could involve looking into feature engineering to better capture the nuances of higher-priced items, investigating outliers to understand why the model performs poorly on them, and potentially using different models or ensemble techniques to see if they can better capture the complex relationships in the higher-priced segments of the market.

            Considering these observations and the previously mentioned performance metrics (MAE, MSE, R2 Score), the model is strong, but there might be room for improvement in predicting higher-end items.""")
    else:
            st.error("No prediction data found. Please run the model training first.")
