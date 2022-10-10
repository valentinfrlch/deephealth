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
    df = pd.read_csv(csv)
    df.insert(47, "Sleep Delta (hr)", synthesize(df, "sleep_delta"))
    # df.insert(48, "Mood", synthesize(df, "mood"))
    return df
    
    
def correlation(df):
    # correlation matrix
    corr = df.corr()
    # plot the heatmap
    plt.subplots(figsize=(50, 50))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-
                1.0, vmax=+1.0, cmap='Spectral')
    plt.show()
    

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
    
    
    