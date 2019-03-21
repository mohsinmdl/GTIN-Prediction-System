# coding: utf-8

# In[274]:


'''
 @(#)Generic TimeSeries.ipynb

    Copyright (c) 2018 - 2020 Majestic Interntional Technologies, Inc.
    Majestic Technologies, Lahore, Pakistan.
    All rights reserved.

    This software is the confidential and proprietary information of
    Majestic Technologies, Inc. ("Confidential Information").  You shall not
    disclose such Confidential Information and shall use it only in
    accordance with the terms of the license agreement you entered into
    with Majestic.
 '''

# '''                           +---------+---------------------------------------+
#                               |  Title  |    Predicting Sales of diffrent FormExample  |
# +-----------------------------+---------+---------------------------------------+----------------------------------+
# |
# | Project           :    Majestic Technologies
# | Author(s)         :    M. Usman Shafique <usmanm4m@gmail.com>| Abdur Rauf <raufjavid5@gmail.com>|
# | Date              :    04-April-2018
# | Version           :    1.0
# | Status            :    Active
# | Technology        :    Python
# | IDE               :    Jupyter Notebook
# | Naming Convention :    PEP 8 = Python Enhancement Proposal
# | Library used      :    FaceBook prophet(Prophet is an open source forecasting tool built by Facebook)
# |                           * It can be used for time series modeling and forecasting trends into the future
# |                           * Prophet installation instructions >>https://facebook.github.io/prophet/docs/installation.html
# +----------------------------+---------------------------------------------------------------------------------------+
# | Purpose of Program/Project | Predicting the fulture sales of different FormExample based on their
# |                            | past sales.
# +----------------------------+---------------------------------------------------------------------------------------+
# | Input                      | A training dataset, in which one row contain Date and the other contain the Qunatity.
# |                            |
# | Output                     | Prediction of Future Sales for next 24 weeks(6 month).
# +----------------------------+---------------------------------------------------------------------------------------+
#
#                                                     +-------------------+
#                                                     | Coding Guidelines |
# +---------------------------------------------------+-------------------+--------------------------------------------+
# |
# |# 1- Standard Syntax for variables, functions declarations i-e import_dataset()
# |# 2- Documentations, Comments should be capitalized.
# |# 3- Dont leave unused code in files.
# |# 4- Commenting format should be like i-e # Comments
#
#
#                                +------------------------+
#                                |    Hyperparameters     |
# +------------------------------+---+--------------------+-----------------------------------------------------------+
# |   variable                       |                                 Description
# +----------------------------------+--------------------------------------------------------------------------------+
# |   dataset                        | a dataframe to store data.
# |   df                             | a dataframe to store data.
# |   ds ,y                          | The naming convention of using ( 'ds' for the date and 'y' for the value)
# |   filePath                       | path and name of the file
# |   title                          | name of plot(Graph)
# |   labels                         | label for X-axis and Y_axis
# |   _type                          | type of file
# |   sort_by                        | sorted by given feature
# |   sorted_file_name               | sorted data file
# |   W                              | Weekly aggregated
# |   Weekly_data                    | aggregated weekly data
# |   Time Series                    | value (Quantity )
# |   y , yhat                       | y  for actual and yhat for predicted
# +----------------------------------+--------------------------------------------------------------------------------+
# '''
# print('Welcome to \'DATA SCIENCES\' World -)-)-)-:):)')


# # -----------------------------------Program Order and Flow---------------------------------------

# ### **Step 1 : Import libraries and load the data**
#   > **1.1 Import libraries and load the Data**
#   
#   > **1.2 Sort the Date**
#   
#   > **1.3 Convert Date into Date Time Type**
#   
#   > **1.4 Re-Sample DataSet **
#   
#   > **1.5 Naming Convention **
# 
# ### **Step 2 : Understanding and Visualization of Data**
#   
#   > **2.1 Check Null Values**
#   
#   > **2.2 Check Outliers by box plot**
#   
#   > **2.3 Check Outliers by Z-Score Method**
#   
#   > **2.4 Visualization**
# 
# 
# ### **Step 3 : Preprocessing**
#  > **3.1 Fill Missing values and Verify **
#  
#  > **3.2 Remove Outliers and Verify**
#  
#  > **3.3 Visualization**
#  
#  > **3.4 Check Data is Stationary or Not- Augmented Dicky Fuller Test**
#  
#  > **3.5 Apply Log Transformation if data is not statioanary**
#  
# 
# ### **Step 4  : Instantiate a Prophet model and fit it to our data**
# 
# 
# ### **Step 5  : Train Model to Predict Future Forecast **
# 
# 
# ### **Step 6  : Plotting Forecast**
# 
# 
# ### **Step 7 : Evaluation**
#   >**7.1 R-squared value**
#   
#   >**7.2 Mean Absolute Error**
#   
# ###    Step 8: Comparing Actual Vs Predicted Values  
# 
# ### **Step 9 : Future Predictions**
# 
# ### Step 10: Export the Predictions to CSV File
# 
# ### Step 11: Print the Future Predictions to PDF file 
# 
# ### Step 12: Saving the model - Dump File.
# 
# ### Step 13: Loading the model - Load File.

# --------------------------------------------------------------------------------------------------------------

# # **Step 1  :Importing the required libraries and import the dataset**

# In[275]:


import pandas as pd
import pickle
import numpy as np
#from fbprophet import Prophet
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import pylatex as pl
import os

# In[276]:


'''
/*-------------------------------------------- IMPORT_DATASET-- ----*/
|  Function import_dataset
|
|  Purpose:  Store the data in dataframe.
|
|  Parameters:
|  filePath (IN) -- the path.
|
|  Returns (OUT):  The dataframe.
 *-------------------------------------------------------------------*/      
 '''


def import_dataset(filePath):
    dataset = pd.read_csv(filePath)

    return dataset


# In[277]:


'''
/*-------------------------------------------- SORTING_DATA ----------*/
|  Function sort_data
|
|  Purpose:  Sorting the Data
|
|  Parameters:
|  filePath (IN) -- the path and name of file
|  sort_by  (IN) -- sort by given feature
|  sorted_file_name -- sorted data file
|  Returns (OUT):  sorted file
 *-------------------------------------------------------------------*/      
 '''


def sort_data(dataset, sort_by, sorted_file_name):
    results = dataset.sort_values(by=[sort_by])
    file_name = results.to_csv(sorted_file_name, index=False)


# In[278]:


'''
/*-------Convert Date (DATA_TYPE) INTO DateTimeIndex (DataType) ----*/
|  Function convert_in_DateTime_index
|
|  Purpose:  Conversion of Date DataType into DateTimeIndex DataType
|
|  Parameters:
|  filePath (IN) -- the path
|
|  Returns (OUT):  sales_data after converting into datatimeIndex dataType
 *-------------------------------------------------------------------*/      
 '''


def convert_in_DateTime_index(filePath):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    sales_data = pd.read_csv(filePath, parse_dates=['GTINShipDate'], index_col='GTINShipDate', date_parser=dateparse)

    return sales_data


# In[611]:


'''
/*-------------------------------------------RE-SAMPLE_DATA----------*/
|  Function weekly_sample
|
|  Purpose:  Conversion of Date DataType into DateTimeIndex DataType
|
|  Parameters:
|  data (IN) -- The Orgional data
|
|  Returns (OUT): aggragated data
 *-------------------------------------------------------------------*/      
 '''


def weekly_sample(data, aggregate):
    weekly = data.resample(aggregate).sum()
    print("Last 5 records of dataset:\n ")
    print(weekly.tail())

    return weekly


# In[280]:


'''
/*-------------------------------------------RESET_INDEX----------*/
|  Function reset_index
|
|  Parameters:
|  weekly_data (IN) -- the weekly data
|
|  Returns (OUT): dateframe with reset index
 *-------------------------------------------------------------------*/       
 '''


def reset_index(weekly_data):
    df = weekly_data.reset_index()

    return df


# ### **very important Note**
# 
# >There are only two columns in the data, a date and a value.
# 
# ### The naming convention of using ( 'ds' for the date and 'y' for the value ) is apparently a requirement to use Prophet
# 
# 
# >**it's expecting those exact names and will not work otherwise!**

# In[281]:


'''
/*-------------------------------------------RENAME_COLOUMN----------*/
|  Function rename
|
|  Parameters:
|  weekly_data (IN) -- The aggregated weekly data
|
|  Returns (OUT): aggragated data
 *-------------------------------------------------------------------*/      
 '''


def rename(weekly_data):
    df = weekly_data.rename(columns={'GTINShipDate': 'ds', 'Quantity': 'y'})

    return df


# -------------------------------------------------------------------------------------

# # **Step 2  :Understanding and  Visualization of Data**

# In[282]:


'''
/*---------------------------------------------STATISTICS----------*/
|  Function shape,describe
|
|  Purpose:Statistics about the data
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): summary of statistics
 *-------------------------------------------------------------------*/      
 '''


def shape(df):
    shape = df.shape
    return shape


def describe(df):
    describe = df.describe()
    return describe


# **check Null Values**

# In[415]:


'''
/*---------------------------------------CHECK_NULL_VALUES----------*/
|  Function checkNullValues
|
|  Purpose: Check null values in our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): Null values
 *-------------------------------------------------------------------*/      
 '''


def checkNullValues(df):
    # Ist replace the 0 values with NaN.

    df['Quantity'] = df['Quantity'].replace(0, np.NaN)
    df['GTINShipDate'] = df['GTINShipDate'].replace(0, np.NaN)

    # get the number of missing data points per column
    missing_values_count = df.isnull().sum()

    return missing_values_count


# **check outliers (with Box Plot)**

# In[284]:


'''
/*-----------------------------CHECK_OUTLIERS_WITH_BOX_PLOT----------*/
|  Function check_outliers
|
|  Purpose: Check outliers in our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): graph that shows outliers
 *-------------------------------------------------------------------*/      
 '''


def check_outliers_with_box_plot(df):
    MAX_ROWS = 10
    pd.set_option('display.max_rows', MAX_ROWS)
    pd.set_option('display.max_columns', 200)
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.figure(figsize=(10, 8))
    plt.subplot(212)
    plt.xlim(df["Quantity"].min(), df["Quantity"].max() * 1.1)
    sns.boxplot(x=df["Quantity"])


# **check outliers (with z_score)**

# In[344]:


'''
/*---------------------------CHECK_OUTLIERS_WITH_Z_SCORE ------------*/
|  Function check_outliers_z_score
|
|  Purpose: Check outliers in our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): outliers
 *-------------------------------------------------------------------*/ 
  '''


def check_outliers_z_score(df):
    return df[(np.abs(stats.zscore(df['Quantity'])) > 3)]


# #### To understand the data we plot ist 50 Records in Bar Graph 

# In[286]:


'''
/*--------------------------------------------BAR_GRAPH ------------*/
|  Function check_outliers_z_score
|
|  Purpose: Plot Bar Graph to check the data variability
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): Graph
 *-------------------------------------------------------------------*/ 
  '''


def bar_graph(df):
    ax = df.set_index('ds').head(20).plot.bar(figsize=(20, 8))
    ax.set_ylabel('Quantity')
    ax.set_xlabel('Date')
    plt.show()


# #### Complete Data shows in line Graph

# In[287]:


'''
/*--------------------------------------------LINE_GRAPH ------------*/
|  Function check_outliers_z_score
|
|  Purpose: Plot Line Graph to check the data variability
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): Graph
 *-------------------------------------------------------------------*/ 
  '''


def line_graph(df):
    ax = df.set_index('ds').plot(figsize=(15, 6))
    ax.set_ylabel('Passengers')
    ax.set_xlabel('Date')
    plt.show()


# # **Step 3  : Preprocessing **

# ### Fill Missing values

# #### our data have some missing values so lets fill that data with interpolation

# In[389]:


'''
/*-------------------------------------- Drop_Missing_Values -------*/
|  Function fill_missing_values
|
|  Purpose: To Fill missing values in our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): DataFrame with filled values 
 *-------------------------------------------------------------------*/ 
  '''


def drop_missing_values(df):
    # df.fillna(df.interpolate(),inplace=True)
    df = df.dropna(inplace=True)

    return df


# ### Removing outliers
# 
# - If your data have outliers than Run this code to remove outliers otherwise skip this step.

# In[410]:


'''
/*--------------------------------------------REMOVE_OUTLIERS --------*/
|  Function remove_outlier
|
|  Purpose: To remove outliers
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): DataFrame with no outlier
 *-------------------------------------------------------------------*/ 
  '''


def remove_outlier(df):
    # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'
    df = df[np.abs(df.Quantity - df.Quantity.mean()) <= (3 * df.Quantity.std())]

    return df


# **verify that outlier is removed** 

# In[353]:


'''
/*-------------------------------VERIFY_OUTLIERS_ARE_REMOVED --------*/
|  Function remove_outlier
|
|  Purpose: To verify outliers are removed
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): DataFrame with no outlier
 *-------------------------------------------------------------------*/ 
  '''


def verify_outlier_removed(df):
    verify = df[(np.abs(stats.zscore(df['y'])) > 3)]  # see, there is no outlier exist now

    return verify


# #### See, the Plots again after cleaning data 

# In[352]:


'''
/*--------------------------------------------VISUALIZATION --------*/
|  Function plot_to_check_data_is_clean
|
|  Parameters:
|  df(IN) -- DataFrame
|
|  Returns (OUT): Graph
 *-------------------------------------------------------------------*/ 
  '''


def plot_to_check_data_is_clean(df):
    ax = df.set_index('ds').head(20).plot.bar(figsize=(20, 8))
    ax.set_ylabel('Passengers')
    ax.set_xlabel('Date')
    plt.show()

    ax = df.set_index('ds').plot(figsize=(15, 6))
    ax.set_ylabel('Passengers')
    ax.set_xlabel('Date')
    plt.show()


# ### ** Check if Data is Stationary or Not- Augmented Dicky Fuller Test**

# **Dickey-Fuller Test:**
# 
# - **This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for difference confidence levels.** 
# 
# 
# - **If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.** 

# In[351]:


'''
/*------------------------------------------DICKY_FULLER_TEST-------*/
|  Function test_stationary
|
|  Purpose: To check data is statioanry or Not
|
|  Parameters:
|  timeseries (IN) -- pass the 'Qunatity' coloumn to dicky_fuller_test
|
|  Returns (OUT): Different Statistics values
 *-------------------------------------------------------------------*/ 
  '''


def test_stationary(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    # Plot rolling statistics:
    # orig = plt.plot(timeseries, color='blue', label='Original')
    # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean')
    #
    # plt.show(block=False)

    # Perform Augmented Dickey-Fuller test:
    print('Results of Augmented Dickey-Fuller test:')
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dftest[0], dftest[4]


# - **if Test Statistic value is greater than all the Critical values its mean our data is not Stationary so no need to use Log Transformation or Differencing.**

# - **One of the first tricks to reduce trend and Seasonality can be transformation**

# ### if Data is not Stationary Apply Log Transformation to reduce the Seasonalities and Trends.

# In[354]:


'''
/*------------------------------------------LOG_TRANSFORMATION-------*/
|  Function log_transformation
|
|  Purpose: To make the data stationary
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): DataFrame with log values 
 *-------------------------------------------------------------------*/ 
  '''


def log_transformation(df):
    df['y'] = np.log(df['y'])
    # df.set_index('ds').plot(figsize=(20, 8))


# **In this simpler case, it is easy to see a forward trend in the data. But its not very intuitive in presence of noise. So we can use some techniques to estimate or model this trend and then remove it from the series**

# # Step 4: **We can now instantiate a Prophet model and fit it to our data**

# In[355]:


'''
/*------------------------------------------APPLY_FACEBOOK_PROPHET_MODEL-------*/
|  Function apply_model
|
|  Purpose: Fit the Model to our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|
|  Returns (OUT): Trained Model 
 *-------------------------------------------------------------------*/ 
  '''


def apply_model(df):
    model = Prophet(changepoint_prior_scale=0.05, weekly_seasonality=True)
    model = model.fit(df)

    return model


# ** very important**
# 
# 
# ** Adjusting trend flexibility **
# If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can adjust the strength of the sparse prior using the input argument changepoint_prior_scale. By default, this parameter is set to 0.05.
# 
# **Increasing it will make the trend more flexible AND Decreasing it will make the trend less flexible**

# # Step 5:  Predict Future Forecast

# In[356]:


'''
/*------------------------------------------FUTURE PREDICTIONS-------*/
|  Function predict
|
|  Purpose: Give the TimeFrame
|
|  Parameters:
|  model (IN) -- Trained Model
|  no_of_periods (IN) -- TimeFrame e.g 24
|  frequency (IN) -- Weekly , Monthly  Yearly
|
|  Returns (OUT): Future Predictions
 *-------------------------------------------------------------------*/ 
  '''


# include_history= FALSE >>> mean we dont want the predictions of history only want the future predictions.
def predict(model, no_of_periods, frequency):
    future = model.make_future_dataframe(periods=no_of_periods, freq=frequency, include_history=False)
    forecast = model.predict(future)

    return future, forecast


# # Step 6: Plotting Forecast

# - **Prophet has a plotting mechanism called plot.**
# 
# - **This plot functionality draws the original data (black dots),**
# 
# - **The blue line indicates the forecasted values**
# 
# - **The light blue shaded region is the uncertainty **

# In[357]:


'''
/*-------------------------------------------PLOTTING_FORECAST-------*/
|  Function plotting_forecast
|
|  Purpose: Plotting Future Predictions
|
|  Parameters:
|  model (IN) -- Trained Model
|  forecast (IN) -- Predictions
|
|  Returns (OUT): Shows the (yearly, weekly, Trend) Graph 
 *-------------------------------------------------------------------*/ 
  '''


def plotting_forecast(model, forecast):
    model.plot(forecast)
    model.plot_components(forecast)


# # Step 7: Evaluation 1) R-Squared value 2) Mean Absolute Error

# In[358]:


'''
/*-------------------------------------------Evaluation with log Transformation-------*/
|  Function evaluation
|
|  Purpose: Fit the Model to our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|  forecast (IN) -- Predictions
|
|  Returns (OUT): Mean_Absolute_Error , R2_square error ,metric_df 
 *-------------------------------------------------------------------*/ 
  '''


def evaluation_with_log(forecast, df):
    metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)

    mean_absolute = mean_absolute_error(np.exp(metric_df['y']), np.exp(metric_df['yhat']))
    r2_square = r2_score(np.exp(metric_df['y']), np.exp(metric_df['yhat']))

    return metric_df, mean_absolute, r2_square


# In[359]:


'''
/*-------------------------------------------Evaluation with_out log Transformation-------*/
|  Function evaluation
|
|  Purpose: Fit the Model to our dataset
|
|  Parameters:
|  df (IN) -- DataFrame
|  forecast (IN) -- Predictions
|
|  Returns (OUT): Mean_Absolute_Error , R2_square error ,metric_df 
 *-------------------------------------------------------------------*/ 
  '''


def evaluation_with_out_log(forecast, df):
    metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)

    mean_absolute = mean_absolute_error((metric_df['y']), (metric_df['yhat']))
    r2_square = r2_score((metric_df['y']), (metric_df['yhat']))

    return metric_df, mean_absolute, r2_square


# # Step 8: Comparing Actual Vs Predicted Values

# In[360]:


'''
/*-------------------------------------------ACTUAL_VS_PREDICTED with_log------*/
|  Function ActualVsPredicted
|
|  Purpose: Fit the Model to our dataset
|
|  Parameters:
|  metric_df (IN) -- combined actual vs predicted values
|
|  Returns (OUT): metric_df ( orgional values of actual and predicted after renaming )
 *-------------------------------------------------------------------*/ 
  '''


def ActualVsPredicted_with_log(metric_df):
    metric_df['y'] = np.exp(metric_df['y'])
    metric_df['yhat'] = np.exp(metric_df['yhat'])
    metric_df = metric_df.rename(columns={'ds': 'Date', 'y': 'actual', 'yhat': 'predicted'})
    metric_df = metric_df.round(2).tail()

    return metric_df


# In[361]:


'''
/*-------------------------------------------ACTUAL_VS_PREDICTED with_out_log------*/
|  Function ActualVsPredicted
|
|  Purpose: Fit the Model to our dataset
|
|  Parameters:
|  metric_df (IN) -- combined actual vs predicted values
|
|  Returns (OUT): metric_df ( orgional values of actual and predicted after renaming )
 *-------------------------------------------------------------------*/ 
  '''


def ActualVsPredicted_with_out_log(metric_df):
    metric_df['y'] = (metric_df['y'])
    metric_df['yhat'] = (metric_df['yhat'])
    metric_df = metric_df.rename(columns={'ds': 'Date', 'y': 'actual', 'yhat': 'predicted'})
    metric_df = metric_df.round(2).tail()

    return metric_df


# # Step 9: Future Predictions

# In[362]:


'''
/*-------------------------------------------FUTURE_PREDICTIONS_with_log------------*/
|  Function futurePredictions
|
|  Purpose: Transform back to its real values by exponential
|
|  Parameters:
|  forecast (IN) -- predicted log values
|
|  Returns (OUT):  orgional predicted values with upper_bound and lower_bound
 *--------------------------------------------------------------------------*/ 
'''


def futurePredictions_with_log(forecast, next):
    forecast['Prediction'] = np.exp(forecast['yhat'])
    forecast['Prediction_lower_bound'] = np.exp(forecast['yhat_lower'])
    forecast['Prediction_upper_bound'] = np.exp(forecast['yhat_upper'])
    forecast['Date'] = (forecast['ds'])
    # forecast = forecast[(forecast.Date > next)][['Date', 'Prediction', 'Prediction_lower_bound', 'Prediction_upper_bound']].round(2)
    forecast = forecast[['Date', 'Prediction', 'Prediction_lower_bound', 'Prediction_upper_bound']].round(2)

    return forecast


# In[363]:


'''
/*-------------------------------------------FUTURE_PREDICTIONS_with_out_log------------*/
|  Function futurePredictions
|
|  Purpose: Transform back to its real values by exponential
|
|  Parameters:
|  forecast (IN) -- predicted log values
|
|  Returns (OUT):  orgional predicted values with upper_bound and lower_bound
 *--------------------------------------------------------------------------*/ 
'''


def futurePredictions_with_out_log(forecast, next):
    forecast['Prediction'] = (forecast['yhat'])
    forecast['Prediction_lower_bound'] = (forecast['yhat_lower'])
    forecast['Prediction_upper_bound'] = (forecast['yhat_upper'])
    forecast['Date'] = (forecast['ds'])
    # forecast = forecast[(forecast.Date > next)][['Date', 'Prediction', 'Prediction_lower_bound', 'Prediction_upper_bound']].round(2)
    forecast = forecast[['Date', 'Prediction', 'Prediction_lower_bound', 'Prediction_upper_bound']].round(2)

    return forecast


# # Step 10: Export the Future Predictions to CSV file

# In[364]:


'''
/*-------------------------------------------EXPORT_CSV------------*/
|  Function export_to_csv
|
|  Purpose: generate a csv file
|
|  Parameters:
|  forecast (IN) -- predicted values
|
|  Returns (OUT):  make a csv file
 *--------------------------------------------------------------------------*/ 
'''


def export_to_csv(forecast):
    df_ = forecast[['Date', 'Prediction', 'Prediction_lower_bound', 'Prediction_upper_bound']].round(2)
    df_.to_csv('GTIN_Application/static/csv3/GTINPredictions.csv', encoding='utf-8', index=False)


def full_export_to_csv(forecast, file_path, fixed_index):
    df_ = forecast[['Date', 'Prediction', 'Prediction_lower_bound', 'Prediction_upper_bound']].round(2)
    df_['GTIN'] = file_path

    if fixed_index == 0:
        print('Index is :  ')
        print(fixed_index)
        df_.to_csv('GTIN_Application/static/csv3/FullFuturePredictions.csv', mode='a', encoding='utf-8', index=False )
    else:
        print('Indes is gya :  ')
        print(fixed_index)
        df_.to_csv('GTIN_Application/static/csv3/FullFuturePredictions.csv', mode='a', encoding='utf-8', index=False, header=False)



# # Step 11: Print the Future Predictions to PDF file 

# **Link** https://tex.stackexchange.com/questions/340349/how-to-print-a-data-frame-from-pandas-using-pylatex

# In[365]:


'''
/*-------------------------------------------EXPORT_PDF------------*/
|  Function convertPdf
|
|  Purpose: generate a pdf file
|
|  Parameters:
|  forecast (IN) -- predicted values
|
|  Returns (OUT):  make a pdf file
 *--------------------------------------------------------------------------*/ 
'''


def convertPdf(forecast):
    print = pd.DataFrame({'Date': forecast['Date'],
                          'Prediction': forecast['Prediction'],
                          'Prediction Lower Bound': forecast['Prediction_lower_bound'],
                          'Prediction Upper Bound': forecast['Prediction_upper_bound']})

    print = print.round(2)
    print.index.name = 'x'
    M = np.matrix(print.values)
    doc = pl.Document()
    with doc.create(pl.Section('Future Predictions')):
        with doc.create(pl.Tabular('ccccc')) as table:
            table.add_hline()
            table.add_row([print.index.name] + list(print.columns))
            table.add_hline()
            for row in print.index:
                table.add_row([row] + list(print.loc[row, :]))
            table.add_hline()

    doc.generate_pdf('Future Predictions', clean_tex=False)


# # Saving the model - Dump File.

# In[366]:


'''
/*-------------------------------------------- SAVING_MODEL -------*/
|  Function save_model
|
|  Purpose:  Saving the model into Dump File
|
|  Parameters:
|  filePath (IN) -- the path and name of file
|  _type  (IN) -- type of file
|  Returns (OUT):  saved(Dumped) model
 *-------------------------------------------------------------------*/      
 '''


def save_model(model, filePath, _type):
    return pickle.dump(model, open(filePath, _type))


# # Loading the model - Load File.

# In[367]:


'''
/*-------------------------------------------- LOADING_FILE -------*/
|  Function load_model
|
|  Purpose:  Loading the model from file.
|
|  Parameters:
|  filePath (IN) -- the path and name of file
|  _type  (IN) -- type of file
|  Returns (OUT):  loaded model
 *-------------------------------------------------------------------*/      
 '''


def load_model(filePath, _type='rb'):
    if len(filePath) == 0:
        print("Path not provided")
        return
    else:
        model = pickle.load(open(filePath, _type))
        return model


# # Main Method


# Hyper parameters
def main_method(file_path, aggregate, no_of_periods):
    file_path = file_path + '.csv'
    sort_by = 'GTINShipDate'
    sorted_file_name = file_path
    aggregate = aggregate  # monthly , 'W'= Weekly , 'D'=Daily
    no_of_periods = no_of_periods
    frequency = aggregate  # freq ’day’, ’week’, ’month’, ’quarter’, ’year’, 1(1 sec), 60(1 minute) or 3600(1 hour)

    # Import the dataSet
    dataset = import_dataset(file_path)

    missing_values = checkNullValues(dataset)

    # Drop Missing Values
    drop_missing_values(dataset)

    missing_values = checkNullValues(dataset)

    # Sort values
    sort_data(dataset, sort_by, sorted_file_name)

    # Import the dataSet
    dataset = import_dataset(sorted_file_name)

    # Convert the date coloumn into dateTimeIndex coloumn
    dataset = convert_in_DateTime_index(sorted_file_name)

    # Aggregate the data into weekly
    aggregated_data = weekly_sample(dataset, aggregate)

    # In[782]:

    # Reset the index
    aggregated_data = reset_index(aggregated_data)

    missing_values = checkNullValues(aggregated_data)

    # drop the null values
    drop_missing_values(aggregated_data)

    missing_values = checkNullValues(aggregated_data)

    # Rename the coloumns name
    df = rename(aggregated_data)

    # dicky_fuller_test
    test_statistics, critical_value = test_stationary(df.y)

    critical_value = str(critical_value.values())

    critical_value = str(critical_value).split()
    critical_value = critical_value[2]
    test1 = critical_value[0:5]
    critical_value = float(test1)

    # Apply_log_transformation if data is not statinary e.g if Test Statistics value is less than critical value

    if (test_statistics > critical_value):
        # Apply log Transformation
        print("log_transformation_applied")
        log_transformation(df)
        # Apply model
        model = apply_model(df)
        # give the frequecy and no of periods you want to predict
        future, forecast = predict(model, no_of_periods, frequency)

        print('\n ')

        Future_Predictions = futurePredictions_with_log(forecast, next)
        print("-------------------------------Future Predictions-------------------------- \n \n")
        print(Future_Predictions)
    else:
        # Apply model
        model = apply_model(df)
        # give the frequecy and no of periods you want to predict
        future, forecast = predict(model, no_of_periods, frequency)

        Future_Predictions = futurePredictions_with_out_log(forecast, next)
        print("-------------------------------Future Predictions-------------------------- \n \n")
        print(Future_Predictions)
    # Export to CSV
    export_to_csv(forecast)


#     last date

# Hyper parameters
def last_Date(file_path, aggregate):
    file_path = file_path + '.csv'
    sort_by = 'GTINShipDate'
    sorted_file_name = file_path
    aggregate = aggregate  # monthly , 'W'= Weekly , 'D'=Daily

    # Import the dataSet
    dataset = import_dataset(file_path)

    missing_values = checkNullValues(dataset)

    # Drop Missing Values
    drop_missing_values(dataset)

    missing_values = checkNullValues(dataset)

    # Sort values
    sort_data(dataset, sort_by, sorted_file_name)

    # Import the dataSet
    dataset = import_dataset(sorted_file_name)

    # Convert the date coloumn into dateTimeIndex coloumn
    dataset = convert_in_DateTime_index(sorted_file_name)

    # Aggregate the data into weekly
    aggregated_data = weekly_sample(dataset, aggregate)

    # In[782]:

    # Reset the index
    aggregated_data = reset_index(aggregated_data)

    missing_values = checkNullValues(aggregated_data)

    # drop the null values
    drop_missing_values(aggregated_data)

    missing_values = checkNullValues(aggregated_data)

    # Rename the coloumns name
    df = rename(aggregated_data)

    next = df.tail(1)
    next = str(next).split()

    next = next[3]
    print('Last Date is '+next)

    return next


# # Main Method


# Hyper parameters
def all_GTIN_Predictions(file_path, aggregate, no_of_periods , fixed_index):
    file_path_name = file_path
    file_path = file_path + '.csv'
    sort_by = 'GTINShipDate'
    sorted_file_name = file_path
    aggregate = aggregate  # monthly , 'W'= Weekly , 'D'=Daily
    no_of_periods = no_of_periods
    frequency = aggregate  # freq ’day’, ’week’, ’month’, ’quarter’, ’year’, 1(1 sec), 60(1 minute) or 3600(1 hour)

    # Import the dataSet
    dataset = import_dataset(file_path)

    missing_values = checkNullValues(dataset)

    # Drop Missing Values
    drop_missing_values(dataset)

    missing_values = checkNullValues(dataset)

    # Sort values
    sort_data(dataset, sort_by, sorted_file_name)

    # Import the dataSet
    dataset = import_dataset(sorted_file_name)

    # Convert the date coloumn into dateTimeIndex coloumn
    dataset = convert_in_DateTime_index(sorted_file_name)

    # Aggregate the data into weekly
    aggregated_data = weekly_sample(dataset, aggregate)

    # In[782]:

    # Reset the index
    aggregated_data = reset_index(aggregated_data)

    missing_values = checkNullValues(aggregated_data)

    # drop the null values
    drop_missing_values(aggregated_data)

    missing_values = checkNullValues(aggregated_data)

    # Rename the coloumns name
    df = rename(aggregated_data)

    # dicky_fuller_test
    test_statistics, critical_value = test_stationary(df.y)

    critical_value = str(critical_value.values())

    critical_value = str(critical_value).split()
    critical_value = critical_value[2]
    test1 = critical_value[0:5]
    critical_value = float(test1)

    # Apply_log_transformation if data is not statinary e.g if Test Statistics value is less than critical value

    if (test_statistics > critical_value):
        # Apply log Transformation
        print("log_transformation_applied")
        log_transformation(df)
        # Apply model
        model = apply_model(df)
        # give the frequecy and no of periods you want to predict
        future, forecast = predict(model, no_of_periods, frequency)

        print('\n ')

        Future_Predictions = futurePredictions_with_log(forecast, next)
        print("-------------------------------Future Predictions-------------------------- \n \n")
        print(Future_Predictions)
    else:
        # Apply model
        model = apply_model(df)
        # give the frequecy and no of periods you want to predict
        future, forecast = predict(model, no_of_periods, frequency)

        Future_Predictions = futurePredictions_with_out_log(forecast, next)
        print("-------------------------------Future Predictions-------------------------- \n \n")
        print(Future_Predictions)
    # Export to CSV
    full_export_to_csv(forecast, file_path_name, fixed_index)


