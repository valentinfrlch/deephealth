from deep import *
import helpers as helpers
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
from matplotlib import font_manager
import re
import progressbar
from matplotlib import dates as mdates

from lightgbm import LGBMRegressor
import datetime
from scipy.interpolate import make_interp_spline


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

    # plot the complete time series
    fig, axs = plt.subplots(4, figsize=(16, 8))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(observed)
    axs[0].grid()

    # plot the trend of the time series
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(trend)
    axs[1].grid()

    # plot the seasonality of the time series. Period=24 daily seasonality | Period=24*7 weekly seasonality.
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(seasonal)
    axs[2].grid()

    # plot the noise of the time series
    axs[3].set_title('NOISE', fontsize=16)
    axs[3].plot(residual)
    axs[3].scatter(y=residual, x=range(len(residual)), alpha=0.5)
    axs[3].grid()

    plt.show()


def autocorrelation(df, datapoint):
    # autocorrelation plot
    plot_acf(df[datapoint].values, lags=50)
    plt.show()


def predict(df, datapoint, horizon=7, plot=False, smoothness=10):
    X = df.drop(datapoint, axis=1)
    y = df[datapoint]

    X_model, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_model, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # create, train and do inference of the model
    model = LGBMRegressor()
    model.fit(X_model, y_model)
    predictions = model.predict(X_test)

    # calculate MAPE between predictions and test data
    accuracy = 0

    # rolling average of predictions and y_test
    y_test = y_test.rolling(smoothness).mean()
    
    # predictions to dataframe
    predictions = pd.DataFrame(
        predictions, index=y_test.index, columns=[datapoint])
    # rolling average of predictions
    predictions = predictions.rolling(smoothness).mean()

    # plot reality vs prediction for the last week of the dataset
    if plot:
        # add font
        font_dir = ["./assets/"]
        font_files = font_manager.findSystemFonts(fontpaths=font_dir)
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)

        plt.figure(figsize=(32, 12), facecolor='#021631')
        ax = plt.axes()

        plt.grid(color='#6E7A8B')

        for tick in ax.get_xticklabels():
            tick.set_fontname("Product Sans")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Product Sans")

        ax.set_facecolor('#021631')

        ax.spines['bottom'].set_color('#6E7A8B')
        ax.spines['top'].set_color('#6E7A8B')
        ax.spines['left'].set_color('#6E7A8B')
        ax.spines['right'].set_color('#6E7A8B')

        ax.set_title(f'{name_reconstruct(datapoint)} Measurements vs deepHealth Model',
                     fontsize=20, color='white', fontname='Product Sans')
        # set axis tick color
        ax.tick_params(axis='x', colors='#6E7A8B')
        ax.tick_params(axis='y', colors='#6E7A8B')

        # set text color
        plt.rcParams['text.color'] = 'white'

        # set prediction index to same index as y_test
        predictions.index = y_test.index

        plt.plot(y_test, label='Measurements', color='#00E89D')
        plt.plot(predictions, label='Prediction', color='#77B7EE')

        plt.savefig(
            f'./visualisations/training_plots/{name_reconstruct(datapoint)}.png')
        # close the plot
        # remove plt from memory
        plt.close()
        plt.clf()

        # add logo to plot
        helpers.logo(
            f'./visualisations/training_plots/{name_reconstruct(datapoint)}.png')

    return model, accuracy


def importance(model, dp):
    # create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # plot variable importances of the model
    plt.title(f'Variable Importances - {dp}', fontsize=16)
    # only show top 10
    sns.barplot(x='importance', y='feature',
                data=df_importances.iloc[:20, :], palette='Blues_d', orient='h')
    # change size of plot
    plt.gcf().set_size_inches(15, 4)
    plt.show()
    plt.close()
    plt.clf()


def train(df, horizon=7, smoothness=10):
    # iterate through columns
    for dp in df.columns:
        # show progress
        print(f"Processing {dp}")
        try:
            lambda dp: re.sub('[^A-Za-z0-9_]+', '', dp)
            mode, accuracy = predict(
                df, dp, horizon=horizon, plot=True, smoothness=smoothness)
        except ValueError as e:
            print(e)
            continue


def preprocess(path):
    df = convert(path, "json")
    df = df.select_dtypes(exclude=['object'])
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return df


def predict_next(df, horizon=7, smoothness=10):
    # create a progress bar
    bar = progressbar.ProgressBar(maxval=len(df.columns), widgets=[
                                  'Initializing Model...', progressbar.Percentage(), " ", progressbar.Bar('█')], term_width=150)
    bar.start()
    for dp in df.columns:
        bar.update(bar.currval + 1)
        bar.widgets[0] = f'Processing {name_reconstruct(dp, True, True)} '
        X = df.drop(dp, axis=1)
        X_test = X.iloc[-horizon:, :]

        model, accuracy = predict(df, dp)
        Y = model.predict(X_test)

        past_data = df[dp]

        Y = pd.DataFrame(Y, columns=[dp])

        # merge past data with predictions and use Date from df as index
        merged = pd.concat([past_data, Y], axis=0)
        # create new indexes for the future data start from the last date in the dataset
        # get the last date in the dataset
        last_date = past_data.index[-1]
        # generate indexes for the future data starting on last_date with a frequency of 1 day
        future_index = pd.date_range(last_date, periods=horizon, freq='1D')

        merged = merged.rolling(window=smoothness, min_periods=1).mean()
        
        # split into past and future
        past, future = merged.iloc[:-horizon], merged.iloc[-horizon:]
        future.index = future_index

        # get the last value of the past data
        last_value = past.iloc[-1].values[0]
        # get the last date of the past data
        last_date = past.index[-1]
        

        # put the last value and date into a dataframe as first row use date as index
        future = pd.concat([pd.DataFrame([[last_value, last_date]], columns=[
                           dp, 'Date']).set_index('Date'), future])
        
        future = future.drop(future.columns[1], axis=1)
        
        # convert index to datetime
        future.index = pd.to_datetime(future.index)
        past.index = pd.to_datetime(past.index)

        lineplot(f"Prediction of {name_reconstruct(dp)}", dp, [
                 [past, "#77B7EE"], [future, "#00E89D"]])


def lineplot(title, dptitle, data, consecutive=True):
    """_summary_: Produces a lineplot of given data.

    Args:
        data (_type_: list): _description_: A list of data to be plotted. List must contain tuple with color value.
    """
    # styling of plot
    plt.figure(figsize=(32, 12)), facecolor='#021631')
    ax = plt.axes()

    font_dir = ["./assets/"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dir)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    plt.title(title, fontsize=20, color='white',
              fontname='Product Sans', y=1.038)

    plt.grid(color='#6E7A8B')
    
    # display monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.set_facecolor('#021631')

    ax.spines['bottom'].set_color('#6E7A8B')
    ax.spines['top'].set_color('#6E7A8B')
    ax.spines['left'].set_color('#6E7A8B')
    ax.spines['right'].set_color('#6E7A8B')
    # set axis tick color
    ax.tick_params(axis='x', colors='#6E7A8B')
    ax.tick_params(axis='y', colors='#6E7A8B')

    # set text color
    plt.rcParams['text.color'] = 'white'
    plt.rcParams["font.family"] = "Product Sans"
    
    # interpolate the dataframe and pplot it
    for d in data:
        try:
            d[0] = d[0][~d[0].index.duplicated(keep='first')]
            d[0] = d[0].sort_index()
            d[0] = d[0].reindex(pd.date_range(d[0].index[0], d[0].index[-1], freq='1H'))
            d[0] = d[0].interpolate(method='cubic')
            plt.plot(d[0], color=d[1])
        except ValueError as e:
            print(e)
            continue


    if consecutive:
        if len(data) > 1:
            x1, y1 = data[0][0].index[len(
                data[0][0]) - 1], data[0][0].iloc[-1].values[0]
            x2, y2 = data[1][0].index[0], data[1][0].iloc[0].values[0]
            plt.plot([x1, x2], [y1, y2], color=d[1][1], linestyle='--')
            plt.plot(x1, y1, marker='o', color=data[0][1], markersize=8)
            plt.plot(x1, y1, marker='o', color="white", markersize=4)

    plt.savefig(
        f'./visualisations/predictions/{name_reconstruct(dptitle)}.png')
    # close the plot
    plt.close()
    plt.clf()
    helpers.logo(
        f'./visualisations/predictions/{name_reconstruct(dptitle)}.png')


def name_reconstruct(name, equalize=False, bold=False):
    # add a space before every capital letter
    # metrics to ignore
    metrics = ["dBASPL", "mmHg", "kmhr", "min",
               "ms", "count", "countmin", "kcal", "g", "kg"]
    metrics.sort(key=len, reverse=True)

    for metric in metrics:
        if name.endswith(metric):
            name = name[:-len(metric)]
            break
    name = name.replace("_", " ")
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    name = name[0].upper() + name[1:]

    if equalize:
        # if name is too long, shorten it and add "..."
        if len(name) > 30:
            name = name[:27] + "..."
        name = name.ljust(30)
    # make the name bold
    if bold:
        name = f"\033[1m{name}\033[0m"
    return name


if __name__ == '__main__':
    accuracies = []
    df = preprocess("dataset/export.json")
    """
    #next_week(df)
    for dp in df.columns:
        try:
            print(f"Processing {dp}")
            model, accuracy = predict(df, dp, plot=False))
            # importance(model, dp)
            accuracies.append((dp, accuracy))
        except ValueError:
            print(f"Error processing {dp}")
            continue
    # get mean of all accuracies
    mean_accuracy = np.mean([float(acc[1][:-1]) for acc in accuracies])
    print(f"Mean accuracy: {mean_accuracy}%") """

    predict_next(df, horizon=30, smoothness=7)
    # decompose(df, 'Mood')
    # train(df, horizon=90, smoothness=10)
    # model, accuracy = predict(df, "Mood", horizon=90, plot=True)
    # importance(model, 'Mood')
