# Capstone-Project-PHASE-3
Machine Learning Project: Taiwan Air Quality Index Data

Project Title: Taiwan Air Quality Index Data
Organization: Entri Elevate
Date: 30/12/2024

Project Overview:
The goal of this machine learning project is to forecast the Air Quality Index (AQI) and predict PM2.5 levels based on air pollutant concentrations. The Taiwan Air Quality Index Data provides valuable insights into the air quality at various monitoring stations across Taiwan. By analyzing relationships between AQI and pollutants such as PM2.5, CO, and SO2, this project uses regression techniques to predict air quality trends. Through this, we aim to predict the impact of pollutant concentrations on air quality and provide actionable insights for air pollution management.

Objective:

To forecast the Air Quality Index (AQI) based on various air pollutants.
To predict PM2.5 levels, which are a significant factor in air quality monitoring.
Data Description:
The dataset is sourced from Kaggle (https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024). Initially containing 1,048,576 rows of extensive air quality data, it was filtered to focus on a month's worth of data (62,964 rows) to make the analysis more manageable. The dataset includes multiple features, including date, time, geographical coordinates, AQI, pollutant levels, wind speed and direction, and several moving averages for pollutants such as PM2.5, PM10, SO2, CO, etc.

Key features include:

Date: Date and time of the air quality reading.
Sitename: Monitoring station name.
AQI: Air Quality Index.
Pollutants: Various pollutants such as SO2, CO, O3, PM10, PM2.5, NO2, NOx, NO.
Wind Speed & Direction: Meteorological data.
Average Pollutant Levels: Moving averages of pollutants like PM2.5, PM10, SO2, CO, etc.
Geographical Coordinates: Latitude and longitude of the monitoring stations.
Station ID: Unique identifier for the station.
Data Collection:
The dataset was imported from the Kaggle repository, and an initial exploration was conducted to understand the distribution and relationships within the data.

Data Preprocessing:

Missing values were handled using imputation techniques to ensure completeness.
Outliers were detected and removed using statistical methods to improve model accuracy.
Skewed data was transformed to normalize the distribution of numerical features.

Exploratory Data Analysis (EDA): During EDA, various visualizations were created to understand the relationships and distribution of the data, 
including: Histograms and Boxplots for feature distribution.
Pair Plots and Heatmap Correlation to identify correlations between features.
Pie Diagrams, Bar Plots, and Count Plots to explore categorical features.
Line Plots and KDE (Kernel Density Estimation) for time-series analysis of air quality trends.

Feature Engineering and Selection:

Categorical features were encoded using techniques like one-hot encoding.
Redundant or irrelevant features were removed through feature selection techniques like Random Forest and Select K Best.
The final set of features was chosen based on their relevance to the AQI prediction.
Data Splitting:
The dataset was split into training and testing sets, with 80% allocated for training the models and 20% for testing to evaluate performance.

Feature Scaling:
Numerical features were scaled using Min-Max Scaling to standardize the range of values across all features, ensuring the models perform optimally.

Machine Learning Models:
Several machine learning algorithms were implemented to predict AQI and PM2.5 levels:

Regression Models:
Random Forest Regressor
SVR (Support Vector Regressor)
Linear Regression
Gradient Boosting Regressor
Adaboost Regressor

Model Evaluation:

For regression, performance was evaluated using MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R2 Score.
The models were compared based on their predictive accuracy, and the best performing models were selected.

Hyperparameter Tuning:
The performance of the models was optimized using hyperparameter tuning techniques like Grid Search and Random Search to find the best combination of model parameters.

Model Saving and Testing with Unseen Data:
Once trained, the final model was saved for future use. The model’s generalization capability was assessed using unseen data from the test set to simulate real-world predictions.

Results Interpretation and Conclusion:
The models were evaluated, and their predictive accuracy was analyzed. The best performing models for predicting AQI and PM2.5 were identified. Limitations of the dataset, such as missing values or data imbalances, were also discussed, and possible improvements were noted.

Conclusion:
The Taiwan Air Quality Index (AQI) project focused on predicting air quality and PM2.5 levels using regression techniques. After preprocessing and feature engineering, models such as Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor were applied to predict AQI and pollutant concentrations. The results provided valuable insights into the relationship between pollutants and air quality. The project demonstrated the effectiveness of regression models in forecasting air quality, and future improvements could include fine-tuning hyperparameters and incorporating additional features for better accuracy.
