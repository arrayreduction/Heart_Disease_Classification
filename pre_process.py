import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def eda(df):
    print(df.head())
    print(df.info())

    sns.pairplot(df)
    plt.show()

    return

def main():
    eda(df=pd.read_csv("heart.csv"))

    return