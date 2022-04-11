import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def eda(df):
    '''Exploratory Data Analysis (EDA)'''
    print(df.head())
    print(df.info())

    for i, column in enumerate(df.columns, 1):
        plt.subplot(3,3,i)
        sns.histplot(df[column])

    #df.plot.hist(subplots=True)
    #plt.show()

    #create pairplot
    sns.pairplot(df)
    plt.show()

    #pairplot excluding 

    return

def main(run_eda=False):
    df = pd.read_csv("heart.csv")

    if run_eda:
        eda(df)

    return df