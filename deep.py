import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# convert csv to dataframe

def convert(csv):
    df = pd.read_csv(csv)
    return df
    
    
def correlation(df):
    # correlation matrix
    corr = df.corr()
    # plot the heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1.0, vmax=+1.0, cmap='Spectral')
    plt.show()
    
def pair(df):
    g = sns.pairplot(df[['Walking Heart Rate Average (count/min)', 'Walking + Running Distance (km)']], 
             kind='kde',
             plot_kws=dict(fill=False, color='purple', linewidths=1),
             diag_kws=dict(fill=False, color='purple', linewidth=1))

    # add observation dots
    g.map_offdiag(sns.scatterplot, marker='.', color='black')
    plt.show()
    
    
if __name__ == '__main__':
    df = convert('dataset/export.csv')
    correlation(df)
    #pair(df)
    
    
    