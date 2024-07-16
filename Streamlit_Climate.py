# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:07:16 2024

@author: juliu
"""

import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
import json
import matplotlib
import requests

####### Get relative path
script_dir = os.path.dirname(os.path.abspath("Streamlit_Climate.py"))


####### My Streamlit stuff

st.title("Global Warming : Temperature Prediction ")


# CSS Code to modify the sidebar and the header color
st.markdown("""
<style>
     /* Optional: Adjust header colors if needed */
     .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
          color: #add8e6; /* Light blue color for headers */
          }
    [data-testid=stSidebar] {
        background-color: #000035 !important;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    [data-testid=stSidebar] [data-testid=stVerticalBlock] {
        gap: 0rem !important;
    }
    [data-testid=stSidebar] [data-testid=stMarkdown] p {
        color: white !important;
    }
    [data-testid=stSidebar] [data-testid=stImage] {
        margin-left: auto;
        margin-right: auto;
        display: block;
        width: 100% !important;
    }
    [data-testid=stSidebar] [data-testid=stButton] {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }
    [data-testid=stSidebar] [data-testid=stButton] > button {
        width: 100% !important;
        background-color: #add8e6 !important;
        color: #000035 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.2rem 1rem !important;
        text-align: left !important;
        transition: background-color 0.3s ease !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid=stSidebar] [data-testid=stButton] > button:hover {
        background-color: #87ceeb !important;
    }
    [data-testid=stSidebar] [data-testid=stButton] > button:active {
        background-color: #4682b4 !important;
    }
    .sidebar-about {
        position: fixed;
        bottom: 0;
        left: 0;
        padding: 1rem;
        color: white;
        font-size: 0.8rem;
    }
   
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Create a sidebar
sidebar = st.sidebar

# Add an image at the top of the sidebar
sidebar.image(os.path.join(script_dir,"Data/Globe.png"), use_column_width=True)

# Add a title to the sidebar
sidebar.title("Menu")

# Create buttons in the sidebar
if sidebar.button(":earth_asia: Introduction"):
    st.session_state.page = 'Intro'

if sidebar.button(":mag_right: Exploration"):
    st.session_state.page = 'Explore'

if sidebar.button(":bar_chart: DataVizualization"):
    st.session_state.page = 'Viz'

if sidebar.button(":female-technologist: Modeling"):
    st.session_state.page = 'Mod'
    
if sidebar.button(":crystal_ball: Prediction"):
    st.session_state.page = 'Pred'
    
if sidebar.button(":key: Conclusion"):
    st.session_state.page = 'Fin'

# Add about section at the bottom of the sidebar
sidebar.markdown(
    """
    <div class="sidebar-about">
        <p>Bootcamp: Data Analyst<br>
        Leeke Bremer<br>
        Lekshmy Vasantha<br>
        Julius Kricheldorff</p>
    </div>
    """,
    unsafe_allow_html=True
)


###### Load the Data Exploration

# path to data csvs
url_owid = os.path.join(script_dir,r'Data/hadcrut-surface-temperature-anomaly.csv')
url_git = os.path.join(script_dir,r'Data/Towid-co2-data.csv')
#url_merge = 'https://raw.githubusercontent.com/Leeke/blank-app-template-fkefcwagods/main/merge3.csv'
url_kaggle = os.path.join(script_dir,r'Data/merge.csv')
url_merge = os.path.join(script_dir,r'Data/merge.csv')
url_owid_continents = os.path.join(script_dir,r'Data/continents-according-to-our-world-in-data.csv')

# def load_original_data(url):
    
#     response = requests.get(url)
#     if response.status_code == 200:
#         return pd.read_csv(StringIO(response.text))
#     else:
#         st.error("Failed to load data from GitHub.")
#         return None
    
def load_original_data(string):
    return pd.read_csv(string)

owid_df = load_original_data(url_owid)
git_df = load_original_data(url_git)
kaggle_df = load_original_data(url_kaggle)
merge_df = load_original_data(url_merge)
merge_df = merge_df.drop(merge_df.columns[0], axis=1)

###### load the data Prediction
X_test = pd.read_csv(os.path.join(script_dir,"Data/X_test.csv"), index_col = 0)
y_test = pd.read_csv(os.path.join(script_dir,"Data/y_test.csv"), index_col = 0)
X_train = pd.read_csv(os.path.join(script_dir,"Data/X_train.csv"), index_col = 0)
y_train = pd.read_csv(os.path.join(script_dir,"Data/y_train.csv"), index_col = 0)


# load the models
rid = joblib.load(os.path.join(script_dir,"Data/rid"))
rfor = joblib.load(os.path.join(script_dir,"Data/rfor.joblib"))
boost= joblib.load(os.path.join(script_dir,"Data/boost"))

def transfrom_lat(direct, data): 
    if direct == "radians":
        return((data*np.pi)/180)
    elif direct == "degree":
        return((data*180)/np.pi)

# load the scaler 
loaded_scaler = joblib.load(os.path.join(script_dir,'Data/col_transform2.joblib'))

# rescale X_train so that we get the year and other variables
CO2_cum_t = loaded_scaler.named_transformers_['minmaxscaler-1'].inverse_transform(X_test.co2_cum_total.to_frame())
CO2_cum_c = loaded_scaler.named_transformers_['minmaxscaler-2'].inverse_transform(X_test.co2_cum.to_frame())
CO2 = loaded_scaler.named_transformers_['minmaxscaler-3'].inverse_transform(X_test.co2.to_frame())
Popul = loaded_scaler.named_transformers_['minmaxscaler-4'].inverse_transform(X_test.population.to_frame())
GDP = loaded_scaler.named_transformers_['minmaxscaler-5'].inverse_transform(X_test.gdp.to_frame())
Continent = loaded_scaler.named_transformers_['ordinalencoder-1'].inverse_transform(X_test.continent.to_frame())
Country = loaded_scaler.named_transformers_['ordinalencoder-2'].inverse_transform(X_test.country.to_frame())
Test_year = loaded_scaler.named_transformers_['minmaxscaler-6'].inverse_transform(X_test.year.to_frame())
Latitude = transfrom_lat(direct = "degree", data=X_test.latitude).to_frame().to_numpy()
Longitude = transfrom_lat(direct = "degree", data=X_test.longitude).to_frame().to_numpy()

# put into df
Predictions = pd.DataFrame()
Predictions = pd.DataFrame({"Year":np.array(Test_year).reshape(-1)}, index=y_test.index)

#get predictions for all models
Predictions['RF'] = rfor.predict(X_test)
Predictions['Boost']  = boost.predict(X_test)
Predictions['Ridge']  = rid.predict(X_test)
Predictions['TrueV'] = y_test.temp_anomaly

####### Data for modeling


################# START PAGES 

if st.session_state.page == ("Intro"):
    st.header("Introduction")
    st.markdown("* This project  aims to analyze the temperature anomalies across the world over the time period 1900 to 2017.")
    st.subheader("Project proceeds through the following phases:")
    st.markdown("""
            **Exploration and Visualization**  
            
            **Modelling and Prediction**
            """)
    st.markdown("""
    - In the data exploration phase, dataframes are created that capture the main variables such as Co2 emission,Population,Year etc that helps us to visualize the anomalies.
    - Different visual analytic story boards are created to analyze the variables contributing to the anomaly in greater detail.
    - Project also uses different machine learning models to extract salient feature parameters.
    """)
    #presenters = ["Julius", " Leeke", "Lekshmy"]
    st.markdown("""
    <div style='text-align: right;'>
      <h4><b>Project Team Members</b></h4>
      <p>Julius</p>
      <p>Leeke</p>
      <p>Lekshmy</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: right;'>
    <b>Date: July 16, 2024</b>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.page == 'Explore':
      st.header("Data Exploration")

      st.markdown(
        """We mainly used 2 publicly available data sets for our analysis:"""
      )
      st.write("### OWID Dataset")
      st.markdown(
        """The OurWorldInData Dataset contains the Surface temperature anomaly per country and year.
        While the data frame contains no missing values, not all countries' temperature deviations for all years are available. Nonetheless, it complements GitData, so we decided to include it.
        Period: 1850-2017"""
      )
      st.write("Dataset")
      st.dataframe(owid_df.head(10))
      st.write("Dataset shape")
      st.write(owid_df.shape)
      st.write("Dataset describtion")
      st.dataframe(owid_df.describe())
    
      st.write("Percentage of NAs per Column: 0")
    
      st.write("### GitData Set")
      st.markdown(
        """Data on Co2 and other greenhouse gas emissions (co2, ghg, n2o, ch4) as well as gpd, population by country and year.
          This dataset was identified as highly relevant as it contained information on many relevant variables such as gross domestic product (GDP), population numbers, energy consumption levels, and greenhouse gas emissions. 
          Greenhouse gas emissions are divided into emissions in tonnes by CO2, methane-, total greenhouse gases-, and nitrious oxide and are given in absolute numbers and scaled by population. 
          Moreover, contributions to CO2 emissions are subdivided into industries such as cement, consumption, energy, coal, gas, oil, trade, other industries, and land use change. While the variables in the data set were judged highly relevant, the completeness varies from ~14% of values missing for population values to over 91% for other industries. Moreover, different countries had different numbers of years of data available. For
          1
          instance, for Germany, data is available from 1792 to 2022, whereas for Serbia, only data from 1850 onwards are included. Thus, there is an additional source of missingness in the data that does not show up in the data set as NaNs.
          Period: 1750-2022"""
        )
      st.write("Dataset")
      st.dataframe(git_df.head(10))
      st.write("Dataset shape")
      st.write(git_df.shape)
      st.write("Dataset describtion")
      st.dataframe(git_df.describe())
    
      # Calculate the percentage of NAs per column
      na_percentages = git_df.isna().mean() * 100
      na_percentages_df = na_percentages.reset_index()
      na_percentages_df.columns = ['Column', 'Percentage of NAs']
        
      expand = st.expander("Percentage of NAs per Column")#icon=":material/info:")
      expand.write("Percentage of NAs per Column")
      expand.write(na_percentages_df)
    
    
    
      st.write("### Data Merge")
      st.markdown(
        """A large proportion of data contained missing values. To circumvent the missingness problem according to different start and end records, we used data from 1900 to 2017, which also captured the largest increase in CO2 and temperature. Further, we decided to exclude variables for modeling with a hard cut-off of 80% of missingness, as these were judged to contain too little information to be attributed reliably. Moreover, rows containing three missing variables or more were excluded from further analysis.
    15
    In conclusion, we decided to merge the GitData and OWID data sets containing data from the period from 1900 to 2017. The target variable in our analysis is temperature anomaly, and features include:
    CO2, GDP, Population, Year, Country, Continent, Primaryenergyconsumption, Temperaturechangefromgreenhousegases, Temperature change from CO2, Temperature change from Methane, Temperature change from Nitric-Oxide
        """
        )
      st.write("Dataset")
      st.dataframe(merge_df.head(10))
      st.write("Dataset shape")
      st.write(merge_df.shape)
  
  
if st.session_state.page == "Viz":
    st.header("Data Visualization")
    
    df_aggregated = merge_df.groupby(['year'])['temp_anomaly'].mean().reset_index()
    fig = px.line(df_aggregated, x='year', y='temp_anomaly', title='Average World Surface temperature anomaly 1900 - 2016')
    st.plotly_chart(fig)
    
    df_aggregated = merge_df.groupby(['year', 'continent'])['temp_anomaly'].mean().reset_index()
    fig = px.line(df_aggregated, x='year', y='temp_anomaly', color='continent', title='Surface temperature anomaly by continent 1900 - 2016')
    st.plotly_chart(fig)
    
    #st.write("#### Boxplot temp , continents")
    
    fig = px.box(merge_df, x='continent', y='temp_anomaly', title='Boxplot of Temperature Anomany by Continents')
    st.plotly_chart(fig)
    
    
    #st.write("#### c20 pie")
    co2_categories = git_df[['cement_co2','oil_co2','coal_co2','consumption_co2', 'flaring_co2','gas_co2','other_industry_co2', 'trade_co2']].sum()
    fig = px.pie(co2_categories, names=co2_categories.index, values=co2_categories.values, title='Distribution of CO2 Gas Emissions')
    st.plotly_chart(fig)
    
    #st.write("#### c20 time")
    
    df_aggregated = merge_df.groupby(['year', 'continent'])['co2'].sum().reset_index()
    fig = px.line(df_aggregated, x='year', y='co2', color='continent', title='CO2 Emmissions by Continent')
    st.plotly_chart(fig)
    
    st.write("##### Correlation Heatmap")
    
    # Calculate the correlation matrix
    merge_df_num = merge_df.drop(['country', 'continent'], axis=1) 
    corr_matrix = merge_df_num.corr()
    
    # Create the heatmap using plotly.express
    fig = px.imshow(corr_matrix, 
                  labels=dict(color="Correlation"), 
                  x=corr_matrix.columns, 
                  y=corr_matrix.columns,
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
    st.write("#### Worldmap")
    # file_path =  os.path.join(script_dir,r"Data\countries.geo.json")
    # # response = requests.get(file_path)
    # # if response.status_code == 200:
    # #   counties = response.json()
    # with open(file_path) as file:
    #     counties = json.load(file)
    file_path = 'https://raw.githubusercontent.com/shichik07/ClimateAnalysisStreamlit/main/Data/countries.geo.json'
    response = requests.get(file_path)
    if response.status_code == 200:
        counties = response.json()
    
    df_1900_2000 = owid_df.loc[(owid_df['Year'] <= 2000) & (owid_df['Year'] >= 1900)]
    fig = px.choropleth_mapbox(df_1900_2000, geojson=counties, locations='Code', color='Surface temperature anomaly',
                             color_continuous_scale="icefire",
                             range_color=(-3, +3),
                             mapbox_style="carto-positron",
                             zoom=0.5,
                             opacity=0.5,
                             labels={'unemp':'unemployment rate'}
                            )
    fig.update_layout(title="World Surface temperature anomaly 1900 -2000")
    st.plotly_chart(fig)



    
    df_2000 = owid_df.loc[(owid_df['Year'] >= 2000)]
    fig = px.choropleth_mapbox(df_2000, geojson=counties, locations='Code', color='Surface temperature anomaly',
                             color_continuous_scale="icefire",
                             range_color=(-3, +3),
                             mapbox_style="carto-positron",
                             zoom=0.5,
                             opacity=0.5,
                             labels={'unemp':'unemployment rate'}
                            )
    fig.update_layout(title="World Surface temperature anomaly 2000 and later")
    st.plotly_chart(fig)

if st.session_state.page == "Mod":
    st.header("Modeling")
    st.markdown("""
    - To improve the performance of machine learning models and also to increase model robustness we have done Feature Engineering before applying the models such as applying min max scaler for numerical features and ordinal encoding for categorical features. 
    We have used radian scaler as well to convert degrees to radians
      for latitude and longitude variables. In addition we created a class  to calculate cumilative co2 count.
    - We have utilized Grid Search CV tool of scikit to find the best combination of hyper parameters for Randomn Forest.We have chosen best hyper parameters found during Grid Search for Randomn Forest Algorithm.  
    - To improve the  generalization performance of Randomn Forest Model we have used five fold cross validation.
     
    """)
    st.markdown("### Results")
    st.markdown(
    """| Model | Train RMSE | Train MAE | Train R² | Test RMSE | Test MAE | Test R² |
|-------|------------|-----------|----------|-----------|----------|---------|
| Ridge Regression |0.50|0.37|0.3|0.52|0.38|0.29|
| Gradient Boosting |0.43|0.31|0.5|0.46|0.33|0.44|
| Random Forest |0.39|0.28|0.57|0.40|0.28|0.59|""")
    
    # choice = ['Random Forest', 'Ridge', 'Gradient Boosting Regression']
    # option = st.selectbox('Choice of the model', choice)
    # st.write('The chosen model is :', option)
    
    
    # if option == 'Random Forest':
    #    pred_train= rfor.predict(X_train)
       
    # elif option == 'Ridge':
    #    pred_train= rid.predict(X_train)
       
    # elif option == 'Gradient Boosting Regression':
    #    pred_train= boost.predict(X_train)
       
    
    
    # #Radio button input for user to select metrics for model
    # display = st.radio('What do you want to show ?', ('Accuracy', 'MSE','RMSE','MAE'))
    # if option == 'Random Forest':
       
    #    if display == 'Accuracy':
    #     st.write(rfor.score(X_test, y_test))
    #    if display == 'MSE':
    #       st.write(mean_squared_error(y_test,rfor.predict(X_test))) 
    #    if display ==  'MAE':
    #       st.write(mean_absolute_error(y_test, rfor.predict(X_test)))
    #    if display=='RMSE':
    #       st.write(np.sqrt(mean_absolute_error(y_test, rfor.predict(X_test))))
    # elif option == 'Ridge':       
         
    #      if display == 'Accuracy':
    #         st.write(rid.score(X_test, y_test))
    #      if display == 'MSE':
    #         st.write(mean_squared_error(y_test, rid.predict(X_test))) 
    #      if display ==  'MAE':
    #         st.write(mean_absolute_error(y_test, rid.predict(X_test)))
    #      if display=='RMSE':
    #         st.write(np.sqrt(mean_absolute_error(y_test, rid.predict(X_test))))
    # elif option == 'boost':       
    #      pred_test= boost.predict(X_test)
    #      if display == 'Accuracy':
    #         st.write(boost.score(X_test, y_test))
    #      if display == 'MSE':
    #         st.write(mean_squared_error(y_test,boost.predict(X_test))) 
    #      if display ==  'MAE':
    #         st.write(mean_absolute_error(y_test, boost.predict(X_test)))
    #      if display=='RMSE':
    #         st.write(np.sqrt(mean_absolute_error(y_test, boost.predict(X_test))))
    
    # Create two columns
    st.write("#### Ridge Regression")
    st.markdown("""
                **Hyperparameter:** 
                * alpha = 0.01
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        plt.figure(figsize = (6,6))
        fig,ax = plt.subplots() 
        ax= sns.scatterplot(x = Predictions['TrueV'], y = Predictions['Ridge'])
        xmin, xmax = ax.get_xbound()
        ymin = (xmin * 1) + 0
        ymax = (xmax * 1) + 0
        l = matplotlib.lines.Line2D([xmin, xmax], [ymin, ymax], linestyle = '--', color = 'black')
        ax.add_line(l)
        plt.title("Predicted versus Actual - Ridge Regression")
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        st.pyplot (ax.figure)
    
    with col2:
        # Create a graph for ridge regression with coefficient weights
        r_dict = {"Coefficient Value": rid.coef_[0], 'Feature': rid.feature_names_in_}
        plt.figure(figsize = (6,6))
        feat_importances_r =  pd.DataFrame(data=r_dict)
        feat_importances_r.sort_values(by='Coefficient Value', ascending=False, inplace=True, key=abs)
        ab = sns.barplot(data = feat_importances_r, y = 'Feature', x = 'Coefficient Value')
        ab.axvline(x = 0, color = "black", linewidth = 1, linestyle = '--', alpha = 0.6)
        st.pyplot(ab.figure)
    
    # Create two columns again
    st.write("#### Gradient Boosting")
    st.markdown("""
                **Hyperparameter:** 
                * n_estimators = 100 
                * learning_rate = 0.1 
                * max_depth = 4
                """)
    col1, col2 = st.columns(2)
    
    with col1:
        plt.figure(figsize = (6,6))
        ax = sns.scatterplot(x = Predictions['TrueV'], y = Predictions['Boost'])
        xmin, xmax = ax.get_xbound()
        ymin = (xmin * 1) + 0
        ymax = (xmax * 1) + 0
        l = matplotlib.lines.Line2D([xmin, xmax], [ymin, ymax], linestyle = '--', color = 'black')
        ax.add_line(l)
        plt.title("Predicted versus Actual - Gradient booster")
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        st.pyplot(ax.figure)
        
    with col2:
        # Create a variable importance graph for gradient boosting
        g_dict = {"Importance": boost.feature_importances_ , 'Feature': boost.feature_names_in_ }
        plt.figure(figsize = (6,6))
        feat_importances =  pd.DataFrame(data=g_dict)
        feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
        ax=sns.barplot(data = feat_importances, y = 'Feature', x = 'Importance')
        plt.title("Feature Importance - Gradient Boosting")
        st.pyplot(ax.figure)
            
    st.write("#### Random Forest")
    st.markdown("""
                **Hyperparameter:** 
                * n_estimators = 400 
                * Min_samples_split = 5
                * max_depth = 50
                """)
    # and again
    col1, col2 = st.columns(2)
    
    
    with col1:
        # Plot Residuals of the Random Forest Model
        plt.figure(figsize = (10,8))
        
        ax = sns.scatterplot(x = Predictions['TrueV'], y = Predictions['RF'])
        xmin, xmax = ax.get_xbound()
        ymin = (xmin * 1) + 0
        ymax = (xmax * 1) + 0
        l = matplotlib.lines.Line2D([xmin, xmax], [ymin, ymax], linestyle = '--', color = 'black')
        ax.add_line(l)
        plt.title("Predicted versus Actual - RandomForest")
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        st.pyplot(ax.figure)
        
     
    
     #Plot Model performance of the Random Forest Model
    # plt.figure(figsize = (10,6))
    # plt.subplot(131)
    # pm1 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "R2"], x = "Data", y = "Value")
    # pm1.bar_label(pm1.containers[0],fontsize=10);
    # plt.title("R2 Score",)
    # plt.subplot(132)
    # pm2 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "RMSE"], x = "Data", y = "Value")
    # pm2.bar_label(pm2.containers[0], fontsize=10);
    # plt.title("RMSE Score")
    # plt.subplot(133)
    # pm3 = sns.barplot(data = perf_df.loc[perf_df["Metric"] == "MAE"], x = "Data", y = "Value")
    # pm3.bar_label(pm3.containers[0], fontsize=10);
    # plt.title("MAE Score");
    # st.pyplot(ax.figure)
    with col2:
        # Plot feature importance for random forrest
        f_dict = {"Importance": rfor.feature_importances_, 'Feature': rfor.feature_names_in_, 'Model': "RandomForest"}
        plt.figure(figsize = (6,6))
        feat_importances =  pd.DataFrame(data=f_dict)
        feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
        ax2 = sns.barplot(data = feat_importances, y = 'Feature', x = 'Importance')
        plt.title("Feature Importance - Random Forest");
        st.pyplot(ax2.figure)
  
            
    
if st.session_state.page == 'Pred':
    algs = ['Ridge','Boost','RF']
    cols = ['#FF8C00', '#00FFFF', '#FF00FF']
    Names = ['Ridge ', 'Boosting','Forest']
    opac = 0.3
    if st.checkbox("Show Prediction on Test Set"):
        Predictions = Predictions.groupby('Year').agg({'TrueV': ["mean", "std"],
                                         'Boost':["mean", "std"],
                                         'Ridge':["mean", "std"],
                                         'RF':["mean", "std"]}).reset_index()
        Predictions = Predictions.sort_values(by='Year')

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Ridge Regression", "Gradient Boosting", "Random Forest"))
        
        for i, (pred, color) in enumerate(zip(algs, cols), 1):
            # Add actual values
            # Add fill (error band)
            fig.add_trace(go.Scatter(
                x=Predictions['Year'].tolist() + Predictions['Year'].tolist()[::-1],
                y=(Predictions[pred, 'mean'] + Predictions[pred, 'std']).tolist() + (Predictions[pred, 'mean'] - Predictions[pred, 'std']).tolist()[::-1],
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                opacity=opac  # This controls the fill opacity
            ), row=i, col=1)
            
            # add upper and lower bound
            fig.add_trace(go.Scatter(
                x=Predictions['Year'].tolist() + Predictions['Year'].tolist()[::-1],
                y=(Predictions['TrueV', 'mean'] + Predictions['TrueV', 'std']).tolist() + (Predictions['TrueV', 'mean'] - Predictions['TrueV', 'std']).tolist()[::-1],
                fill='toself',
                fillcolor='White',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                opacity=opac  # This controls the fill opacity
            ), row=i, col=1)
        
            if i == 1:
                fig.add_trace(go.Scatter(x=Predictions['Year'], y=Predictions['TrueV', 'mean'], name='Test Data', line=dict(color='#E0E0E0')), row=i, col=1)
                
            else: 
                fig.add_trace(go.Scatter(x=Predictions['Year'], y=Predictions['TrueV', 'mean'], name='Test Data', line=dict(color='#E0E0E0'), showlegend=False), row=i, col=1)
            
            # Add prediction
            fig.add_trace(go.Scatter(x=Predictions['Year'], y=Predictions[pred, 'mean'], name=Names[i-1]+' Prediction', line=dict(color=color, width=2)), row=i, col=1)
        
        # Update layout
        fig.update_layout({
            'height':900,
            'width':1000,
            'title_text':"Prediction Overall",
            'font_color':'white',
            })
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        # Show the plot
        #fig.show()
        st.plotly_chart(fig)
    
    
    def get_default_values():
        return {
            "Year": int(Predictions.Year.min()),
            "Latitude": 0,
            "Longitude": 0,
            "CO2": int(CO2.min()),
            "CO2 cumulative total": int(CO2_cum_t.min()),
            "CO2 cumulative total by country": int(CO2_cum_c.min()),
            "GDP": int(GDP.min()),
            "Population": int(Popul.min())
        }

    # Initialize session state for slider values if not already present
    if 'slider_values' not in st.session_state:
        st.session_state.slider_values = get_default_values()
    
    # Create empty containers for the sliders
    slider_containers = {
        "Year": st.empty(),
        "Latitude": st.empty(),
        "Longitude": st.empty(),
        "CO2": st.empty(),
        "CO2 cumulative total": st.empty(),
        "CO2 cumulative total by country": st.empty(),
        "GDP": st.empty(),
        "Population": st.empty()
    }
    
    # Define a multiselect to control visibility
    available_sliders = list(slider_containers.keys())
    selected_sliders = st.multiselect("Select sliders to display", available_sliders, default=["Year", "CO2 cumulative total"])
    
    # Define the sliders within the containers
    # Reset button
  
    for slider_name, container in slider_containers.items():
        with container:
            if slider_name == "Year":
                st.session_state.slider_values[slider_name] = st.slider(label="Year", min_value=int(Predictions.Year.min()), max_value=int(Predictions.Year.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "Latitude":
                st.session_state.slider_values[slider_name] = st.slider(label="Latitude", min_value=int(Latitude.min()), max_value=int(Latitude.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "Longitude":
                st.session_state.slider_values[slider_name] = st.slider(label="Longitude", min_value=int(Longitude.min()), max_value=int(Longitude.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "CO2":
                st.session_state.slider_values[slider_name] = st.slider(label="CO2", min_value=int(CO2.min()), max_value=int(CO2.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "CO2 cumulative total":
                st.session_state.slider_values[slider_name] = st.slider(label="CO2 cumulative total", min_value=int(CO2_cum_c.min()), max_value=int(CO2_cum_t.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "CO2 cumulative total by country":
                st.session_state.slider_values[slider_name] = st.slider(label="CO2 cumulative total by country", min_value=int(CO2_cum_c.min()), max_value=int(CO2_cum_c.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "GDP":
                st.session_state.slider_values[slider_name] = st.slider(label="GDP", min_value=int(GDP.min()), max_value=int(GDP.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
            elif slider_name == "Population":
                st.session_state.slider_values[slider_name] = st.slider(label="Population", min_value=int(Popul.min()), max_value=int(Popul.max()), value=st.session_state.slider_values[slider_name], step=1, key=f"slider_{slider_name}")
    
    # Control visibility based on the multiselect
    for slider_name, container in slider_containers.items():
        if slider_name not in selected_sliders:
            container.empty()
    
    # You can now use these values in your predictions, e.g.:
    p_year = st.session_state.slider_values["Year"]
    p_lat = st.session_state.slider_values["Latitude"]
    p_lon = st.session_state.slider_values["Longitude"]
    p_CO2 = st.session_state.slider_values["CO2"]
    p_CO2_cum = st.session_state.slider_values["CO2 cumulative total"]
    p_CO2_cum_country = st.session_state.slider_values["CO2 cumulative total by country"]
    p_GDP = st.session_state.slider_values["GDP"]
    p_Population = st.session_state.slider_values["Population"]
   
    # create example dataframe
    Example = X_train.loc[1,].to_frame().transpose()
    Example = Example.to_dict(orient='records')[0]
    
    if st.button("Reset Sliders"):
        st.session_state.slider_values = get_default_values()
        st.experimental_rerun()
      
    new_vals = pd.Series({'year':p_year,
                        'co2': p_CO2,
                        'co2_cum':p_CO2_cum_country,
                        'co2_cum_total':p_CO2_cum,
                        'gdp':p_GDP,
                        'population':p_Population,
                        'continent': "Asia",
                        'country': "India",
                        'latitude': p_lat,
                        'longitude':p_lon})
    # Update prediction data    
    Example.update(new_vals) 
    Example = pd.Series(Example).to_frame().transpose()
    Example[['co2_cum_total','co2_cum', 'co2', 'population', 'gdp','continent', 'country', 'year']] = loaded_scaler.transform(Example)
    
    # Get predictions
    Predictions_New = pd.DataFrame()
    Predictions_New['RF'] = rfor.predict(Example).round(2)
    Predictions_New['Boost']  = boost.predict(Example).round(2)
    Predictions_New['Ridge']  = rid.predict(Example).round(2)
    
    #set size of metrics
    st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 100px;
        }
    [data-testid="stMetricDelta"] {
        font-size: 80px;
        }
    </style>
    """,
    unsafe_allow_html=True,)
    
    # Now Predict a Temperature
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.metric(label="Predicted Temperature " + Names[0], value = '',delta=float(Predictions_New['Ridge'].values))
    with col2:
        st.metric(label="Predicted Temperature " + Names[1], value = '',delta=float(Predictions_New['RF'].values))
    with col3:
        st.metric(label="Predicted Temperature " + Names[2], value = '',delta=float(Predictions_New['Boost'].values))
if st.session_state.page == 'Fin':
    st.markdown("## Conclusion")
    
    st.markdown("* Temperature deviations related to climate change can be predicted using features associated with human activity, utilizing ridge regression, gradient boosting, and random forests models")
    st.markdown("* **Cumulative emitted CO2** was shown to be one of the most informative variables for predicting temperature deviations, aligning with existing literature")
    st.markdown("* The **Random Forest model** achieved the highest predictive accuracy with an R² of ~59%, accurately tracking most temperature deviations within ±1 degree, but struggled with lower deviations and older data")
    
    
    st.markdown("## Difficulties encountered")
    st.markdown("* Large amounts of missing values, leading the team to eliminate variables and cases and backward-filling as imputation")
    st.markdown("* Computational power limitations during cross-validation of the random forest algorithm --> restricting the analysis to four parameters and specific parameter ranges")
    
    st.markdown("## Future Improvements")
    st.markdown("* Data availability was limited for some countries, especially in the 19th and early 20th centuries, leading to increased data imputation using a simple backward filling method")
    st.markdown("* Using a more sophisticated or theory-driven imputation method and including a variable for lagged CO2 values might improve the model's predictive accuracy")
    st.markdown("* While the current approach identified variables predicting past temperature deviations, developing models to generalize to future data could have significant societal and business relevance")
    