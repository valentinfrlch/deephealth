import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import networkx as nx
from pyvis.network import Network

# convert csv to dataframe


def synthesize(df, value):
    if value == "sleep_delta":
        v = df["Sleep Analysis [In Bed] (hr)"] - \
            df["Sleep Analysis [Asleep] (hr)"]
    elif value == "mood":
        v = mood(df)
    elif value == "audio":
        v = audio(df)
    elif value == "bp":
        v = (df["Blood Pressure [Systolic] (mmHg)"] +
             df["Blood Pressure [Diastolic] (mmHg)"]) / 2
    else:
        v = None
    return v

def fitness(df):
    # calculate a fitness score for each day in percentage
    # get all the necessary data from the dataframe
    sleep_delta = df["Sleep Delta (hr)"]
    asleep = df["Sleep Analysis [Asleep] (hr)"]
    v02_max = df["vo2_max"]
    resting_hr = df["resting_heart_rate"]
    body_fat = df["body_fat_percentage"]

    sleep_delta_max = 3
    asleep_max = 9
    absolute_sleep_max = 9
    v02_max_max = 60
    resting_hr_max = 60
    body_fat_max = 20

    # disregrard values that are 0
    sleep_delta = sleep_delta[sleep_delta != 0]
    asleep = asleep[asleep != 0]
    v02_max = v02_max[v02_max != 0]
    resting_hr = resting_hr[resting_hr != 0]
    body_fat = body_fat[body_fat != 0]
    
    # calculate the fitness score
    fitness = (sleep_delta / sleep_delta_max) + (asleep_max / asleep) + (v02_max / v02_max_max) + (resting_hr_max / resting_hr) + (body_fat_max / body_fat) * 100

    df["Fitness"] = fitness
        


def mood(df):
    """Get the mood, and emotion from mood.json and insert it into the dataframe

    Args:
        df (dataframe): The dataframe to insert the mood into
    """
    # read the mood.json file
    with open("dataset/mood.json") as f:
        mood = json.load(f)

        # create a new dataframe with the mood data
        mood_df = pd.DataFrame(mood)
        # convert date to datetime but ignore the time
        mood_df["date"] = pd.to_datetime(mood_df["date"]).dt.date
        # set the date as index
        mood_df.set_index("date", inplace=True)
        print(df.index)
        for date in df.index:
            # convert the date to datetime but ignore the time
            date = pd.to_datetime(date).date()
            # check if there is mood data for the date
            if date in mood_df.index:
                # add the mood and emotion to the dataframe
                print(date)


                

def audio(df):
    # add the higher value of headphone and environmental audio exposure
    max_audio_exposure = df[[
        "Headphone Audio Exposure (dBASPL)", "Environmental Audio Exposure (dBASPL)"]].max(axis=1)
    return max_audio_exposure


def convert(file, mode="json"):
    if mode == "csv":
        # remove all empty columns from csv
        df = pd.read_csv(file)

    else:
        # create a dataframe from json file
        with open(file) as f:
            data = json.load(f)
            names = []
            for i in data["data"]["metrics"]:
                names.append(i["name"])

            df = pd.DataFrame(columns=names)
            # get values from data["data"]["metrics"]["data"]
            for i in range(len(names)):
                # print progress in percentage and clear line so it doesn't print multiple times
                print("\r" + str(round(i / len(names) * 100)) + "%", end="")
                # get the values for each metric and add it to the dataframe
                for j in range(len(data["data"]["metrics"][i]["data"])):
                    # print progress in percentage
                    try:
                        # get the date and value from the json file and add the value to the dataframe at the correct date
                        date = pd.to_datetime(data["data"]["metrics"][i]["data"][j]["date"]).date()
                        value = data["data"]["metrics"][i]["data"][j]["qty"]
                        df.at[date, names[i]] = value
                    except KeyError:
                        continue

                # Blood Pressure
                if names[i] == "blood_pressure":
                    for j in range(len(data["data"]["metrics"][i]["data"])):
                        try:
                            # get the date and value from the json file and add the value to the dataframe at the correct date
                            date = pd.to_datetime(data["data"]["metrics"][i]["data"][j]["date"]).date()
                            systolic = data["data"]["metrics"][i]["data"][j]["systolic"]
                            diastolic = data["data"]["metrics"][i]["data"][j]["diastolic"]
                            df.at[date, "Blood Pressure [Systolic] (mmHg)"] = systolic
                            df.at[date, "Blood Pressure [Diastolic] (mmHg)"] = diastolic
                        except KeyError:
                            print("KeyError")
                            continue

                # Sleep Analysis
                if names[i] == "sleep_analysis":
                    for j in range(len(data["data"]["metrics"][i]["data"])):
                        try:
                            date = pd.to_datetime(data["data"]["metrics"][i]["data"][j]["date"]).date()
                            df.at[date,
                                  "Sleep Analysis [In Bed] (hr)"] = data["data"]["metrics"][i]["data"][j]["inBed"]
                            df.at[date,
                                  "Sleep Analysis [Asleep] (hr)"] = data["data"]["metrics"][i]["data"][j]["asleep"]
                            # add Sleep Delta
                            delta = abs(data["data"]["metrics"][i]["data"][j]["inBed"] - data["data"]["metrics"][i]["data"][j]["asleep"])
                            df.at[date, "Sleep Delta (hr)"] = delta
                        except KeyError as e:
                            continue
                
        # Mood
        #mood(df)
        
        # Fitness
        fitness(df)
        
        for i in df.columns:
            try:
                if i != "Date":
                    df[i] = df[i].astype(float)
                else:
                    df[i] = pd.to_datetime(df[i])
            except ValueError as e:
                continue

    df.sort_index(inplace=True)
    

    # todo: synthesize() function
    headphone = [col for col in df.columns if "headphone" in col][0]
    environmental = [col for col in df.columns if "environmental" in col][0]

    # add the higher value of headphone and environmental audio exposure and call the new column audio
    df["Max Audio Exposure"] = df[[headphone, environmental]].max(axis=1)
    # add average of systolic and diastolic blood pressure
    systolic = [col for col in df.columns if "Systolic" in col][0]
    diastolic = [col for col in df.columns if "Diastolic" in col][0]
    df["Blood Pressure Indicator"] = (df[systolic] + df[diastolic]) / 2

    """
    df.insert(i, "Sleep Delta (hr)", synthesize(df, "sleep_delta"))
    i = df.columns.get_loc("Headphone Audio Exposure (dBASPL)")
    df.insert(i, "Max Audio Exposure (dBASPL)", synthesize(df, "audio"))
    i = df.columns.get_loc("Blood Pressure [Systolic] (mmHg)")
    df.insert(i, "Mean Blood Pressure (mmHg)", synthesize(df, "bp"))
    """

    return df


def correlation(df):
    # correlation matrix
    corr = df.corr()
    print("correlation matrix created")
    # plot the heatmap
    plt.subplots(figsize=(50, 50))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-
                1.0, vmax=+1.0, cmap='Spectral')
    print("data visualisation completed")
    # plt.show()
    # save the figure to file
    plt.savefig('visualisations/correlation.png')


def network(df, style, threshold=0.5):
    corr = df.corr()
    # correlation network graph
    # instantiate networkx graph
    plt.figure(figsize=(50, 50))

    if style == "static":
        net = nx.Graph()

        # add nodes = columns from dataframe
        net.add_nodes_from(df.columns)
        # add edges = correlation values
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold:
                    net.add_edge(corr.columns[i], corr.columns[j])
        pos = nx.spring_layout(net, k=1.5, iterations=50)
        nx.draw(net, pos, with_labels=True, node_color='blue',
                edge_color='skyblue', width=1, linewidths=1)
        # save the figure to file
        plt.savefig('visualisations/static_network.png')
    elif style == "dynamic":
        net = Network(height='1300px', width='100%', directed=False, notebook=False, neighborhood_highlight=False, select_menu=False,
                      filter_menu=False, bgcolor='#222222', layout=None, heading='', cdn_resources='local')
        net.set_options(open("options/default.txt").read())
        # net.show_buttons(filter_=['edges', 'nodes'])

        net.add_nodes(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.4:
                    # set the color of the edge according to the correlation value
                    if corr.iloc[i, j] > 0:
                        color = "#5bc3eb"
                    else:
                        color = "#f06449"
                    width = abs(corr.iloc[i, j]) * 0.01
                    # change the highlight color of the edge
                    net.add_edge(corr.columns[i], corr.columns[j], color=color,
                                 title="Correlation: " + str(abs(round(corr.iloc[i, j] * 100))) + "%", value=width)

        net.save_graph("visualisations/dynamic_network.html")


def pair(df, data):
    g = sns.pairplot(df[data],
                     kind='kde',
                     plot_kws=dict(fill=False, color='black', linewidths=1),
                     diag_kws=dict(fill=False, color='black', linewidth=1))

    # add observation dots
    g.map_offdiag(sns.scatterplot, marker='.', color='black')
    g.fig.set_size_inches(50, 50)
    plt.savefig("visualisations/pairplot.png")


def line(df, data, average=True, window=7, normalize=False):
    plt.figure(figsize=(50, 50))
    # add average line if average is true
    # normalize the data if normalize is true
    if normalize:
        for d in range(len(data)):
            df[data[d]] = df[data[d]] / df[data[d]].max()
    for i in range(len(data)):
        if average:
            avg = df[data[i]].rolling(window=window).mean()
            # plot rolling average
            plt.plot(df['Date'], avg, color="C" + str(i),
                     linewidth=5, label=data[i] + " average")
        plt.plot(df['Date'], df[data[i]], label=data[i], linewidth=1)
        # add legend with custom size
        plt.legend(prop={'size': 40})

    plt.savefig('visualisations/lineplot.png')


if __name__ == '__main__':
    df = convert('dataset/export.json')
    # correlation(df)
    # t = df.columns
    # to list
    # t = t.tolist()
    # query1 = df.columns[df.columns.str.contains('Heart Rate Var')][0]
    # query2 = df.columns[df.columns.str.contains('Mood')][0]
    # query3 = df.columns[df.columns.str.contains('Max')][0]
    # line(df, [query1, query2, query3], True, 50, True)
    # pair(df, [query1, query2])
    # network(df, "dynamic", 0.75)
