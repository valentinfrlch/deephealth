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


def mood(df):
    # add json to dataframe
    with open('dataset/mood.json') as f:
        data = json.load(f)

        date = []
        happiness = []
        tags = []

        for i in data:
            date.append(i['date'])
            happiness.append(i['mood'])
            tags.append(i['emotions'])

        # get all different happiness values
        tag_values = []
        for i in tags:
            for j in i:
                if j not in tag_values:
                    tag_values.append(j)

        for d, h, t in zip(date, happiness, tags):
            # locate row with matching date
            date = df.loc[df['Date'] == d]
            # get index of row
            try:
                i = date.index[0]
            except IndexError:
                continue
            # insert happiness and tags into dataframe at index
            df.at[i, 'Mood'] = h
            for v in tag_values:
                if v in t:
                    #  insert True if tag is in list
                    df.at[i, v] = True
                    # convert column to boolean
                else:
                    df.at[i, v] = False
                df[v] = df[v].astype(bool)


def audio(df):
    # add the higher value of headphone and environmental audio exposure
    max_audio_exposure = df[[
        "Headphone Audio Exposure (dBASPL)", "Environmental Audio Exposure (dBASPL)"]].max(axis=1)
    return max_audio_exposure


def convert(file, mode="csv"):
    if mode == "csv":
        # remove all empty columns from csv
        df = pd.read_csv(file)

    else:
        # create a dataframe from json file
        with open(file) as f:
            data = json.load(f)
            # convert to dataframe
            names = []
            values = []
            datetime = []
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
                        df.at[j, names[i]
                              ] = data["data"]["metrics"][i]["data"][j]["qty"]
                        df.at[j, "Date"] = data["data"]["metrics"][i]["data"][j]["date"]
                    except KeyError:
                        continue

                # Blood Pressure
                if names[i] == "blood_pressure":
                    for j in range(len(data["data"]["metrics"][i]["data"])):
                        try:
                            df.at[j,
                                  "Blood Pressure [Systolic] (mmHg)"] = data["data"]["metrics"][i]["data"][j]["systolic"]
                            df.at[j,
                                  "Blood Pressure [Diastolic] (mmHg)"] = data["data"]["metrics"][i]["data"][j]["diastolic"]
                            df.at[j, "Date"] = data["data"]["metrics"][i]["data"][j]["date"]
                        except KeyError:
                            continue

                # Sleep Analysis
                if names[i] == "sleep_analysis":
                    for j in range(len(data["data"]["metrics"][i]["data"])):
                        try:
                            df.at[j,
                                  "Sleep Analysis [In Bed] (hr)"] = data["data"]["metrics"][i]["data"][j]["inBed"]
                            df.at[j,
                                  "Sleep Analysis [Asleep] (hr)"] = data["data"]["metrics"][i]["data"][j]["asleep"]
                            # add Sleep Delta
                            delta = data["data"]["metrics"][i]["data"][j]["inBed"] - \
                                data["data"]["metrics"][i]["data"][j]["asleep"]
                            df.at[j, "Sleep Delta (hr)"] = delta
                            df.at[j, "Date"] = data["data"]["metrics"][i]["data"][j]["date"]
                        except KeyError as e:
                            continue
                
                # if data["data"]["metrics"][i]["data"] has a key calles "heartRate"
                try:
                    if "heartRate" in data["data"]["metrics"][i]["data"][0]:
                        for j in range(len(data["data"]["metrics"][i]["data"]["heartRate"])):
                            df.at[j,
                                  "Heart Rate"] = data["data"]["metrics"][i]["data"][j]["heartRate"]["hr"]
                            df.at[j, "Date"] = data["data"]["metrics"][i]["data"][j]["date"]
                except Exception as e:
                    continue
                
                
            print(df.columns.__contains__("Heart Rate"))

            names.append("Date")

        for i in df.columns:
            try:
                if i != "Date":
                    df[i] = df[i].astype(float)
                else:
                    # convert date to datetime
                    df[i] = pd.to_datetime(df[i])
            except ValueError as e:
                print(e)
                continue

    # convert to datetime
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    df.sort_index(inplace=True)

    for i in df.columns:
        df[i].interpolate(method="linear", inplace=True)

    df.dropna(axis=1, how='all', inplace=True)

    headphone = [col for col in df.columns if "headphone" in col][0]
    environmental = [col for col in df.columns if "environmental" in col][0]

    # add the higher value of headphone and environmental audio exposure and call the new column audio
    df["Max Audio Exposure"] = df[[headphone, environmental]].max(axis=1)
    # add average of systolic and diastolic blood pressure
    systolic = [col for col in df.columns if "Systolic" in col][0]
    diastolic = [col for col in df.columns if "Diastolic" in col][0]
    df["Average Blood Pressure"] = (df[systolic] + df[diastolic]) / 2

    """
    df.insert(i, "Sleep Delta (hr)", synthesize(df, "sleep_delta"))
    i = df.columns.get_loc("Headphone Audio Exposure (dBASPL)")
    df.insert(i, "Max Audio Exposure (dBASPL)", synthesize(df, "audio"))
    i = df.columns.get_loc("Blood Pressure [Systolic] (mmHg)")
    df.insert(i, "Mean Blood Pressure (mmHg)", synthesize(df, "bp"))
    # mood(df)
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
                      filter_menu=False, bgcolor='#222222', font_color="white", layout=None, heading='', cdn_resources='local')
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
    df = convert('dataset/export.csv')
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
