# libraries:
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import xml.etree.ElementTree as ET


def preprocess(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file, sep=',')
        # remove columns that have only NaN values
        df = df.dropna(axis=1, how='all')

        # convert date column to weekday
        df["Weekday"] = pd.to_datetime(df["Date"]).dt.weekday
        # set the uid to "0" for all rows
        df["uid"] = '0'
        df['index'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])

        # fill nan values with interpolation
        df = df.interpolate(method='linear', limit_direction='forward')
        print(df.head(20))
    elif file.endswith('.xml'):
        # create element tree object
        tree = ET.parse(file)
        print('Reading ' + file + '...')
        # for every health record, extract the attributes
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]

        health_records = pd.DataFrame(record_list)

        # proper type to dates
        for col in ['creationDate', 'startDate', 'endDate']:
            health_records[col] = pd.to_datetime(health_records[col])

        # handle SleepAnalysis records
        if 'HKCategoryTypeIdentifierSleepAnalysis' in health_records['type'].unique():
            sleep_analysis_records = health_records[health_records['type']
                                                    == 'HKCategoryTypeIdentifierSleepAnalysis']
            sleep_analysis_records['value'] = sleep_analysis_records['value'].map({
                'HKCategoryValueSleepAnalysisInBed': 0,
                'HKCategoryValueSleepAnalysisAwake': 1,
                'HKCategoryValueSleepAnalysisAsleepCore': 2,
                'HKCategoryValueSleepAnalysisAsleepREM': 3
            })
            health_records.update(sleep_analysis_records)

        # value is numeric, NaN if fails
        health_records['value'] = pd.to_numeric(
            health_records['value'], errors='coerce')
        # some records do not measure anything, just count occurences
        # filling with 1.0 (= one time) makes it easier to aggregate
        health_records['value'] = health_records['value'].fillna(1.0)

        # shorter observation names
        health_records['type'] = health_records['type'].str.replace(
            'HKQuantityTypeIdentifier', '')
        health_records['type'] = health_records['type'].str.replace(
            'HKCategoryTypeIdentifier', '')
        health_records.tail()

        # pivot the dataframe so that each type is a column the and each row is a date
        # since some types have multiple records per day, we interpolate the others
        df = health_records.pivot_table(
            index='creationDate',
            columns=['type'],
            values='value',

        ).interpolate(method='time', limit_direction='both')

        # filter out all columns that have the word "dietary" in them
        df = df[df.columns.drop(list(df.filter(regex='Dietary')))]
        # save the file as it takes a while to process
        df.to_pickle('dataset/export.pkl')
        print("Saved as 'export.pkl'")
    elif file.endswith('.pkl'):
        df = pd.read_pickle(file)

    else:
        df = None
        print("File type not supported")
    return df


def augment(data, type='stress'):
    # daily stress level
    if type == 'stress':
        weights = {
            'HeartRate': 0.25,
            'RespiratoryRate': 0.15,
            'RestingHeartRate': 0.2,
            'PhysicalEffort': 0.15,
            'HeartRateVariabilitySDNN': 0.35
        }

        # normalize the data
        normed_data = (data - data.min()) / (data.max() - data.min())
        stress_score = sum(weight * normed_data[var]
                           for var, weight in weights.items())

        # add the stress score to the dataframe
        data['Stress'] = stress_score
        return data



def visualize_average(data, columns, start_date='2020-01-01', end_date='2024-03-03', overlay=False):
    """Plot minute averaged values over the course of a day for the given date range.
    """
    # filter data for the given date range
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # create a new plotly graph object
    fig = go.Figure()

    # iterate over each column
    for column in columns:
        # group by the minute of day and calculate the mean
        data_grouped = data[column].groupby(
            data.index.hour * 60 + data.index.minute).mean()

        # add the data to the plot
        fig.add_trace(go.Scatter(x=data_grouped.index,
                      y=data_grouped.values, mode='lines', name=column))

        # if overlay is true, add polynomial trend line to the plot
        if overlay:
            # calculate coefficients for the polynomial that minimizes the squared error
            coefficients = np.polyfit(
                data_grouped.index, data_grouped.values, 5)
            # create a polynomial function with these coefficients
            polynomial = np.poly1d(coefficients)
            # calculate the y values for this polynomial
            y_trend = polynomial(data_grouped.index)
            # add the trend line to the plot
            fig.add_trace(go.Scatter(x=data_grouped.index, y=y_trend,
                          mode='lines', name=f'{column} trend line', line=dict(color='green')))

    # set the title and labels
    fig.update_layout(
        title=f'Minute Average over the course of a day',
        xaxis_title='Minute of Day',
        yaxis_title='Value',
        template='plotly_dark'  # use the plotly_dark theme
    )

    # set the x-axis to the minutes in "hh:mm" format
    fig.update_xaxes(tickvals=[i for i in range(
        0, 24*60, 60)], ticktext=[f'{i:02d}:00' for i in range(24)])

    fig.show()


def visualize_range(data, columns, start_date='2020-01-01', end_date='2024-03-03', overlay=False):
    """Plot values over the course of a day for the given date range.
    """
    # filter data for the given date range
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # create a new plotly graph object
    fig = go.Figure()

    # iterate over each column
    for column in columns:
        # add the data to the plot
        fig.add_trace(go.Scatter(x=data.index,
                      y=data[column].values, mode='lines', name=column))

        # if overlay is true, add polynomial trend line to the plot
        if overlay:
            # calculate coefficients for the polynomial that minimizes the squared error
            coefficients = np.polyfit(
                range(len(data)), data[column].values, 5)
            # create a polynomial function with these coefficients
            polynomial = np.poly1d(coefficients)
            # calculate the y values for this polynomial
            y_trend = polynomial(range(len(data)))
            # add the trend line to the plot
            fig.add_trace(go.Scatter(x=data.index, y=y_trend,
                          mode='lines', name=f'{column} trend line', line=dict(color='green')))

    # set the title and labels
    fig.update_layout(
        title=f'Values over the course of a day',
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_dark'  # use the plotly_dark theme
    )

    fig.show()
    

def correlation(data):
    # create a correlation matrix from the "value" column
    corr = data.corr()

    # create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # apply the mask to the correlation matrix
    masked_corr = corr.mask(mask)

    # create a heatmap from the correlation matrix
    heatmap = ff.create_annotated_heatmap(
        z=masked_corr.values,
        x=list(masked_corr.columns),
        y=list(masked_corr.index),
        annotation_text=masked_corr.round(2).fillna("").values,
        colorscale='Viridis',
        showscale=True,
        reversescale=True,
    )
    # show the plot
    heatmap.update_layout(
        plot_bgcolor='rgb(10,10,10)',  # dark background
        paper_bgcolor='rgb(10,10,10)',  # dark background
        font_color='white',  # white text
        template='plotly_dark'  # use the plotly_dark theme
    )

    heatmap.show()


def export_to_csv(data):
    data.to_csv('dataset/export.csv', index=False)


def get_features(data):
    print('-'*50)
    print('Available features:')
    print(data.columns)
    print('-'*50)
    print(data["SleepAnalysis"].describe())
    print('-'*50)
    # print last 10 rows of SleepAnalysis column
    print(data["SleepAnalysis"].tail(10))


if __name__ == "__main__":
    data = preprocess('dataset/export.pkl')
    data = augment(data, type='stress')
    get_features(data)
    # export_to_csv(data)
    visualize_range(data, ['Stress'], overlay=True,
              start_date='2024-02-26', end_date='2024-03-03')
    # correlation(data)
