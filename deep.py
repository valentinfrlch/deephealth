import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# convert csv to dataframe

def synthesize(df, value):
    if value == "sleep_delta":
        v = df["Sleep Analysis [Asleep] (hr)"] - \
            df["Sleep Analysis [In Bed] (hr)"]
    elif value == "mood":
        v = add_mood(df)
    else:
        v = None
    return v



def convert(csv):
    # remove all empty columns from csv
    df = pd.read_csv(csv)
    df.dropna(axis=1, how='all', inplace=True)
    df.insert(47, "Sleep Delta (hr)", synthesize(df, "sleep_delta"))
    # df.insert(48, "Mood", synthesize(df, "mood"))
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
    

def pair(df):
    g = sns.pairplot(df[['Sleep Delta (hr)', 'Blood Pressure [Systolic] (mmHg)']],
                     kind='kde',
                     plot_kws=dict(fill=False, color='purple', linewidths=1),
                     diag_kws=dict(fill=False, color='purple', linewidth=1))

    # add observation dots
    g.map_offdiag(sns.scatterplot, marker='.', color='black')
    g.fig.set_size_inches(50, 50)
    plt.show()
    
    
if __name__ == '__main__':
    df = convert('dataset/export.csv')
    correlation(df)
    #pair(df)
    
    
    