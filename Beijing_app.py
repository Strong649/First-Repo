
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import st_folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score  # Import accuracy_score
from math import sqrt
from IPython.display import display
import io
from io import StringIO
from sklearn import preprocessing
import warnings
import requests
warnings.filterwarnings('ignore')


# Load your data
data = pd.read_csv("combined_data_new.csv")
data['date'] = pd.to_datetime(data['date'])
data['year_month'] = data['date'].dt.to_period('M')
data.set_index('date', inplace=True)
selected_pollutants_df = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
selected_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']
location_data = pd.DataFrame({
    'lon': [39.9874704, 40.0894089, 40.295869, 39.9319966, 39.9356673, 39.89631, 40.3971199, 39.93365, 40.0577176, 39.88921, 39.99767, 39.8797309],
    'lat': [116.4042655, 116.3030507, 116.223521, 116.4341427, 116.3609186, 116.39793, 116.68783, 116.46732, 116.8665467, 116.39912, 116.25758, 116.367375],
    'location': ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
     }, dtype=str)
df_pollutants = data[selected_pollutants_df]
df_pollutants = df_pollutants.apply(pd.to_numeric, errors='coerce')
df_pollutants = df_pollutants.dropna()
mean_pollutant_per_station_new = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
mean_pollutant_per_station = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']].mean()
combined_df_yearly = data.groupby(['station', data['year_month'].dt.year]).agg({
    'PM2.5': 'mean',
    'PM10': 'mean',
    'SO2': 'mean',
    'NO2': 'mean',
    'CO': 'mean',
    'O3': 'mean',
    'TEMP': 'mean',
    'PRES': 'mean',
    'DEWP': 'mean',
    'RAIN': 'mean',
    'WSPM': 'mean',
    'AQI': 'mean'
}).reset_index()

def data_overview():
    st.title("Data Overview")

    st.write("Station Locations")
    st.write("This is the locations of each station from which the data has been taken and the mean value for each pollutant across the entire 2013-17 period.")
    m = folium.Map(location=[39.9167, 116.3833], zoom_start=9)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkblue', 'pink', 'gray', 'black', 'lightblue', 'lightgreen', 'beige']
    for index, row in location_data.iterrows():
        # Create HTML string for the tooltip, add mean_pollutant_per_station
        html_tooltip = f"""
            <b>Location:</b> {row['location']}<br>
            <b>Mean PM2.5:</b> {mean_pollutant_per_station.loc[row['location']]['PM2.5']:.2f}<br>
            <b>Mean PM10:</b> {mean_pollutant_per_station.loc[row['location']]['PM10']:.2f}<br>
            <b>Mean SO2:</b> {mean_pollutant_per_station.loc[row['location']]['SO2']:.2f}<br>
            <b>Mean NO2:</b> {mean_pollutant_per_station.loc[row['location']]['NO2']:.2f}<br>
            <b>Mean CO:</b> {mean_pollutant_per_station.loc[row['location']]['CO']:.2f}<br>
            <b>Mean O3:</b> {mean_pollutant_per_station.loc[row['location']]['O3']:.2f}<br>
            <b>Mean AQI:</b> {mean_pollutant_per_station.loc[row['location']]['AQI']:.2f}<br>
        """

        folium.Marker([row['lon'], row['lat']],
                      tooltip=folium.Tooltip(html_tooltip),
                      icon=folium.Icon(color=colors[index])
                     ).add_to(m)

    st_data = st_folium(m, width=800)
    st.write("As shown in the map above, there are several stations that are clustered together and these have a significant increase in pollutant levels and Mean AQI levels which can be due to the centralised location of the stations, and the population density can be a contributing factor to this.")

    # Display basic information
    st.write("**Shape of the dataset:**", data.shape)
    st.write("**First few rows:**")
    st.write(data.head())
    st.write("The data shows every hour data, which contributes well to a large dataset, however to be able to establish trends the data would need to be grouped in multiple different ways; such as by each day, each month and each year. As well as per station as well.")

    # Display summary statistics
    st.write("**Summary statistics:**")
    st.write(data.describe())

    # Display missing values
    st.write("**Missing values:**")
    st.write(data.isnull().sum())
    st.write("When considering how to handle missing values, there is many different ways we can manage them without skewing the dataset itself. I found that using the mean value for each of the missing values meant for a dataset that best fit my purposes. As shown in the table above, each missing value was replaced with the mean value in its place.")

def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    st.write("These visualisations are to show the differences in each station and also the differences of each pollutant.")

    fig = px.histogram(data, x='station', y=selected_pollutants_df, title='Mean Pollutant Levels by Station', barmode='stack')

    # Add dropdown menus to select pollutants
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True, True, True, True, True]}],
                        label="All Pollutants"
                    ),
                    dict(
                        args=[{"visible": [True, False, False, False, False, False]}],
                        label="PM2.5"
                    ),
                                   dict(
                    args=[{"visible": [False, True, False, False, False, False]}],
                    label="PM10"
                ),
                dict(
                    args=[{"visible": [False, False, True, False, False, False]}],
                    label="SO2"
                ),
                dict(
                    args=[{"visible": [False, False, False, True, False, False]}],
                    label="NO2"
                ),
                dict(
                    args=[{"visible": [False, False, False, False, True, False]}],
                    label="CO"
                ),
                dict(
                    args=[{"visible": [False, False, False, False, False, True]}],
                    label="O3"
                )
            ]),
        )
    ]
)
    st.plotly_chart(fig)
    st.write("This plot shows that 'CO' is by far the largest pollutant of all, with station Wanshouxigong having the highest overall pollutant levels on average through 2013 to 2017.")

    # Box plot
    st.subheader("Box Plot of AQI")
    selected_station = st.selectbox("Select Station", data['station'].unique())

    # Filter data based on the selected station
    filtered_data = data[data['station'] == selected_station]

    # Create the box plot for the selected station
    fig = px.box(filtered_data, x="AQI", title=f"Box Plot of AQI for {selected_station}")
    st.plotly_chart(fig)

    #line graph
    st.subheader("Line Graph of AQI for each station")
    # Create a line graph for AQI over time for each station
    fig = px.line(
    combined_df_yearly,
    x='year_month',
    y='AQI',
    color='station',
    title='AQI for Each Stations Over Years',
    labels={'year_month': 'Year', 'AQI': 'AQI'},
    width =800,
    height =500
    )
    st.plotly_chart(fig)
    st.write("The line graph shows that the majority of the stations had a greater increase of AQI in 2014, with some stations remaining at these elevated levels and some stations decreasing after this point in time.")

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = df_pollutants.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    #Sunburst Model
    st.subheader("Sunburst Charts")
    def Mean_Pollutant_Station():
     mean_pollutant_per_station_new = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
    mean_pollutant_per_station_new_melted = mean_pollutant_per_station_new.reset_index().melt(
    id_vars='station', var_name='Pollutant', value_name='Mean Value'
    )
    fig = px.sunburst(
        mean_pollutant_per_station_new_melted,
        path=['station', 'Pollutant'],
        values='Mean Value',
        color='Mean Value',
        hover_data=['Mean Value'],
        color_continuous_scale='matter',
        title='Mean Pollutant Levels by Station'
    )
    st.plotly_chart(fig)
    def Yearly_Pollutant():
     Yearly_Pollutant = combined_df_yearly
    fig = px.sunburst(combined_df_yearly, path=['station', 'year_month'], values='AQI',
          color='AQI', hover_data=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'],
          color_continuous_scale='RdYlGn_r',
          title='Yearly Pollutant Levels by Station')
    st.plotly_chart(fig)


def modeling_and_prediction():
    st.title("Modeling and Prediction")
    st.header("Model Hyperparameters")
    n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 5)

# Load Data
    st.header("Load and Visualize Data")
    st.write(data.head())

    # Select Features and Target
    features = st.multiselect("Select Features", selected_pollutants)
    target = st.selectbox("Select Target", selected_pollutants)

    if features and target:
        X = data[features]
        y = data[target]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # KNN Regression
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Evaluation Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

        # Visualization
        st.subheader("Actual vs Predicted")
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
        plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Data', alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        st.pyplot(plt)

        #grid search parameters
        parameters = {"n_neighbors": range(1, 50)}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters, cv=5)
        gridsearch.fit(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_params = gridsearch.best_params_
        st.subheader("Best Parameters found")
        st.write(best_params)


        trains_pred_grid = gridsearch.predict(X_train)
        test_pred_grid = gridsearch.predict(X_test)

        train_mse = mean_squared_error(y_train, trains_pred_grid)
        train_rmse = sqrt(train_mse)
        test_mse = mean_squared_error(y_test, test_pred_grid)
        test_rmse = sqrt(test_mse)
        r2_grid = r2_score(y_test, test_pred_grid)

        k_range = range(1, 21)
        train_rmse_list = []
        test_rmse_list = []

        for k in k_range:
          knn = KNeighborsRegressor(n_neighbors=k)
          knn.fit(X_train_scaled, y_train)
          train_preds = knn.predict(X_train_scaled)
          test_preds = knn.predict(X_test_scaled)
          train_rmse_list.append(np.sqrt(mean_squared_error(y_train, train_preds)))
          test_rmse_list.append(np.sqrt(mean_squared_error(y_test, test_preds)))

        st.subheader("Model Performance with Grid Search")
        st.write(f"Training RMSE: {train_rmse:.2f}")
        st.write(f"Test RMSE: {test_rmse:.2f}")
        st.write(f"R2 Score (Optimised Model): {r2_grid:.2f}")

        st.subheader("Actual vs Predicted (Optimised Model)")
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Data')
        plt.scatter(range(len(test_pred_grid)), test_pred_grid, color='red', label='Predicted Data', alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.title("Actual vs Predicted Values (Optimised Model)")
        plt.legend()
        st.pyplot(plt)

        st.subheader("RMSE vs. Number of Neighbors")
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, train_rmse_list, label='Train RMSE')
        plt.plot(k_range, test_rmse_list, label='Test RMSE')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('RMSE')
        plt.legend()
        plt.title('RMSE vs. Number of Neighbors')
        plt.legend()
        st.pyplot(plt)


# Main app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Select Mode", ["Data Overview", "EDA", "Modeling"])

    if app_mode == "Data Overview":
        data_overview()
    elif app_mode == "EDA":
        exploratory_data_analysis()
    elif app_mode == "Modeling":
        modeling_and_prediction()

if __name__ == "__main__":
    main()
