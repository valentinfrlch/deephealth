from asyncio import futures
from deep import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import re

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error


def decompose(df, datapoint, period=24):
    # get length of dataframe
    df = df.interpolate(method='bfill', axis=0)
    samples = len(df)
    print("samples: ", samples)
    res = seasonal_decompose(df[datapoint].values[-samples:], period=period)
    
    
    observed = res.observed
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    #plot the complete time series
    fig, axs = plt.subplots(4, figsize=(16,8))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(observed)
    axs[0].grid()
    
    #plot the trend of the time series
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(trend)
    axs[1].grid()
    
    #plot the seasonality of the time series. Period=24 daily seasonality | Period=24*7 weekly seasonality.
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(seasonal)
    axs[2].grid()
    
    #plot the noise of the time series
    axs[3].set_title('NOISE', fontsize=16)
    axs[3].plot(residual)
    axs[3].scatter(y=residual, x=range(len(residual)), alpha=0.5)
    axs[3].grid()
    
    plt.show()
    
    
def autocorrelation(df, datapoint):
    # autocorrelation plot
    plot_acf(df[datapoint].values, lags=50)
    plt.show()


def predict(df, datapoint, horizon=24*7, plot=False):
    X = df.drop(datapoint, axis=1)
    y = df[datapoint]
    
    # take last week of data to validate model
    X_model, X_test = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
    y_model, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    
    #create, train and do inference of the model
    model = LGBMRegressor()
    model.fit(X_model, y_model)
    predictions = model.predict(X_test)
    
    
    #calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    predictions = np.convolve(predictions, kernel, mode='same')
    
    #rolling average of pandas Series
    y_test = y_test.rolling(window=kernel_size).mean()
    

    #plot reality vs prediction for the last week of the dataset
    if plot:
        fig = plt.figure(figsize=(16,6))
        plt.title(f'{datapoint} Real vs Prediction - MAE {mae}', fontsize=20)
        plt.plot(y_test, color='lightblue')
        plt.plot(pd.Series(predictions, index=y_test.index), color='black')
        plt.legend(labels=['Real', 'Prediction'], fontsize=16)
        plt.grid()
        #save the plot
        plt.savefig(f'./visualisations/plots/{datapoint}.png')
    return model
    
def importance(model):
    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()
    
def train(df):
    dp = "RespiratoryRatecountmin"
    lambda dp:re.sub('[^A-Za-z0-9_]+', '', dp)

    # iterate through columns
    for dp in df.columns:
        # show progress
        print(f"Processing {dp}")
        try:
            lambda dp:re.sub('[^A-Za-z0-9_]+', '', dp)
            model = predict(df, dp)
        except ValueError:
            continue

def preprocess(path):
    df = convert(path)
    df = df.select_dtypes(exclude=['object'])
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # remove "Date" column
    # drop NaN or object
    # convert object to float
    df = df.drop('Date', axis=1)
    return df

def next_week(df, horizon=24*7):
    for dp in df.columns:
        X = df.drop(dp, axis=1)
        # take last week of data to validate model
        X_test = X.iloc[-horizon:,:]
        
        model = predict(df, dp)
        # use model to predict future week
        Y = model.predict(X_test)
        
        
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        Y = np.convolve(Y, kernel, mode='same')
        past_data = df[dp]
        past_data = past_data.rolling(window=kernel_size).mean()
        
        # Y to dataframe
        # remove last 24 hours
        Y = pd.DataFrame(Y, columns=[dp])
        Y = Y.iloc[:-7,:] # remove last 24 hours
        
        merged = pd.merge(past_data, Y, how='outer')
        past = merged[dp].iloc[:-horizon]
        future = merged[dp].iloc[-horizon:]
        
        # visualize past data and add predicted data in another color
        fig = plt.figure(figsize=(16,6))
        plt.title(f'{dp} Prediction', fontsize=20)
        # different colors for past and future data
        # lightblue for past data and orange for future data
        plt.plot(past, color='lightblue')
        plt.plot(future, color='orange')
        # plt.plot(merged)
        plt.legend(labels=['Past Data', 'Predicted Future'], fontsize=16)
        # add a red line to show where the prediction starts
        plt.axvline(x=X_test.index[-1], color='red')
        plt.grid()
        plt.show()
        
        
        
    
    
    
if __name__ == '__main__':
    df = preprocess("dataset/export.csv")
    next_week(df)
    
    