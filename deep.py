import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    max_audio_exposure = df[["Headphone Audio Exposure (dBASPL)", "Environmental Audio Exposure (dBASPL)"]].max(axis=1)
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
                # print progress
                print("converting " + names[i] + " to dataframe")
                # get the values for each metric and add it to the dataframe
                for j in range(len(data["data"]["metrics"][i]["data"])):
                    # print progress in percentage
                    try:
                        print(str(round((j / len(data["data"]["metrics"][i]["data"])) * 100, 2)) + "%", end="\r")
                        df.at[j, names[i]] = data["data"]["metrics"][i]["data"][j]["qty"]
                        df.at[j, "Date"] = data["data"]["metrics"][i]["data"][j]["date"]
                    except KeyError:
                        continue
            
            # insert weekdays into dataframe
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            # add weekdays as columns
            for i in weekdays:
                df[i] = ""
    
    # convert date column to datetime
    df.Date = pd.to_datetime(df.Date)
        
    # create columns for every weekday and set to True if date is on that weekday
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in weekdays:
        df[day] = df["Date"].dt.day_name() == day
        df[day] = df[day].astype(bool)
        
        
    df.dropna(axis=1, how='all', inplace=True)
    # interpolate missing values in dataframe
    df.interpolate(method='ffill', axis=0, inplace=True)
    # get index of column "Sleep Analysis [Asleep] (hr)"
    i = df.columns.get_loc("Sleep Analysis [Asleep] (hr)")
    df.insert(i, "Sleep Delta (hr)", synthesize(df, "sleep_delta"))
    i = df.columns.get_loc("Headphone Audio Exposure (dBASPL)")
    df.insert(i, "Max Audio Exposure (dBASPL)", synthesize(df, "audio"))
    i = df.columns.get_loc("Blood Pressure [Systolic] (mmHg)")
    df.insert(i, "Mean Blood Pressure (mmHg)", synthesize(df, "bp"))
    mood(df)
    print("converted")
    
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
        net = Network(height='1300px', width='100%', directed=False, notebook=False, neighborhood_highlight=False, select_menu=False, filter_menu=False, bgcolor='#222222', font_color="white", layout=None, heading='', cdn_resources='local')
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
