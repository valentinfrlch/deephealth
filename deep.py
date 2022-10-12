from re import T
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import networkx as nx
from pyvis.network import Network

# convert csv to dataframe


def synthesize(df, value):
    if value == "sleep_delta":
        v = df["Sleep Analysis [Asleep] (hr)"] - \
            df["Sleep Analysis [In Bed] (hr)"]
    elif value == "mood":
        v = mood(df)
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
            i = date.index[0]
            # insert happiness and tags into dataframe at index
            df.at[i, 'Mood'] = h
            for v in tag_values:
                if v in t:
                    df.at[i, v] = 1
                else:
                    df.at[i, v] = 0
            
            

def convert(csv):
    # remove all empty columns from csv
    df = pd.read_csv(csv)
    # convert date column to datetime
    df.Date = pd.to_datetime(df.Date)
    df.dropna(axis=1, how='all', inplace=True)
    # get index of column "Sleep Analysis [Asleep] (hr)"
    i = df.columns.get_loc("Sleep Analysis [Asleep] (hr)")
    df.insert(i, "Sleep Delta (hr)", synthesize(df, "sleep_delta"))
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


def network(df, style):
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
                if abs(corr.iloc[i, j]) > 0.5:
                    net.add_edge(corr.columns[i], corr.columns[j])
        pos = nx.spring_layout(net, k=1.5, iterations=50)
        nx.draw(net, pos, with_labels=True, node_color='blue',
                edge_color='skyblue', width=1, linewidths=1)
        # save the figure to file
        plt.savefig('visualisations/static_network.png')
    elif style == "dynamic":
        net = Network(height="900px", width="100%",
                      bgcolor="#222222", font_color=True)
        net.set_options(open("options/default.txt").read())
        # net.show_buttons(filter_=['edges', 'nodes'])

        # add nodes with size according to how many edges they have
        # get how many edges each node has
        net.add_nodes(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.45 or abs(corr.iloc[i, j]) < -0.45:
                    # set the color of the edge according to the correlation value
                    if corr.iloc[i, j] > 0:
                        color = "red"
                    else:
                        color = "blue"
                    net.add_edge(corr.columns[i], corr.columns[j], color=color,
                                 title="Correlation: " + str(corr.iloc[i, j]))
        # save the graph
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


def line(df, data, average=True):
    plt.figure(figsize=(50, 50))
    # add average line if average is true
    if average:
        for i in range(len(data)):
            avg = df[data[i]].rolling(window=7).mean()
            # plot rolling average
            plt.plot(df['Date'], avg, color='black',
                     linewidth=3, label=data[i] + " average")
            # plot data
    for i in range(len(data)):
        plt.plot(df['Date'], df[data[i]], label=data[i], linewidth=1)
        plt.savefig('visualisations/lineplot.png')


if __name__ == '__main__':
    df = convert('dataset/export.csv')

    correlation(df)
    # pair(df, ["Mood", "Heart Rate Variability (ms)", "Sleep Delta (hr)"])
    # line(df, ["Blood Pressure [Systolic] (mmHg)", "Blood Pressure [Diastolic] (mmHg)"], False)
    network(df, "static")
