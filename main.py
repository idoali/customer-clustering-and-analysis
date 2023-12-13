import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle 
from sklearn.cluster import KMeans

marketing_data = pd.read_csv("data_input/marketing_campaign.csv", delimiter = "\\t")
products_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

education_data = pd.melt(marketing_data, id_vars = "Education", value_vars = products_cols)
education_pivot = pd.pivot_table(education_data, columns = "variable", index = "Education", aggfunc = "sum")

marital_data = pd.melt(marketing_data, id_vars = "Marital_Status", value_vars = products_cols)
marital_pivot = pd.pivot_table(marital_data, columns = "variable", index = "Marital_Status", aggfunc = "sum")

cluster_cols_one = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

cluster_cols_two = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'NumDealsPurchases',
                    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']

education_list = ["Basic", "Graduation", "Master", "PhD"]
status_list = ["Alone", "Divorced", "Married", "Single", "Together", "Widow", "YOLO"]

with open("models/product_place_model.pkl", "rb") as f:
    product_model = pickle.load(f)

with open("models/promotion_model.pkl", "rb") as f:
    promotion_model = pickle.load(f)

# Function for bar plot using Plotly
def create_marital_bar(category):
    fig = px.bar(x = marital_pivot.xs(category).values, y = products_cols)
    fig.update_layout(yaxis = {"categoryorder":"total ascending"})
    st.plotly_chart(fig)
    
# Function for bar plot using Plotly
def create_education_bar(category):
    fig = px.bar(x = education_pivot.xs(category).values, y = products_cols)
    fig.update_layout(yaxis = {"categoryorder":"total ascending"})
    st.plotly_chart(fig)

def one_hot_encode(data, categories=None):
    if categories is None:
        # Infer unique categories from the data
        categories = list(set(data))

    encoded_vector = [1 if data == category else 0 for category in categories]
    return encoded_vector

# Function for machine learning prediction
def predict_value(model, input_data):
    if model == "Product-Place Model":
        pred = product_model.predict(input_data)
    else:
        pred = promotion_model.predict(input_data)
    return pred

# Streamlit app

tab1, tab2 = st.tabs(["🎨 Visualization", "🚀 ML Model"])

with tab1:
    st.title("Visualization Tab")

    # First tab: Visualization
    st.text("Bar Plot 1:")
    selected_category_1 = st.selectbox("Select Category for Bar Plot 1:", marketing_data['Education'].unique())
    create_education_bar(selected_category_1)

    st.text("Bar Plot 2:")
    selected_category_2 = st.selectbox("Select Category for Bar Plot 2:", marketing_data['Marital_Status'].unique())
    create_marital_bar(selected_category_2)

with tab2:
    st.title("Machine Learning Prediction Tab")
    
    selected_model = st.selectbox("Select Model:", ["Product-Place Model", "Promotion Model"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        year_birth = st.number_input("Year of Birth") 

        education = st.selectbox("Select Education:", marketing_data['Education'].unique())
        education = one_hot_encode(education, education_list)
        
        marital = st.selectbox("Select Marital Status:", marketing_data['Marital_Status'].unique())
        marital = one_hot_encode(marital, status_list)
    
    with col2:
        if selected_model == "Product-Place Model":
            numerical_inputs = [st.number_input(i) for i in cluster_cols_one[3:]]
            
        elif selected_model == "Promotion Model":
            numerical_inputs = [st.number_input(i) for i in cluster_cols_two[3:]]
        
    the_inputs = [year_birth] + education + marital + numerical_inputs 
    
    # Predict and display result
    if st.button("Make Prediction"):
        input_data = np.array(the_inputs).reshape(1, -1)
        prediction = predict_value(selected_model, input_data)
        st.success(f"Customer Cluster: {prediction[0]}")

