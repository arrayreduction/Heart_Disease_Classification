import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smdiag

def eda(df):
    '''Exploratory Data Analysis (EDA)'''
    print(df.head())
    df.info()
    print()

    #Plot histograms for the 12 columns, modified from
    #https://stackoverflow.com/questions/55389247/how-to-plot-a-matrix-of-seaborn-distplots-for-all-columns-in-the-dataframe
    for i, column in enumerate(df.columns, 1):
        plt.subplot(4,3,i)
        sns.histplot(df[column])

    #create pairplot
    sns.pairplot(df, corner=True)
    plt.show()

    #check for nans
    #Print series containing column:any(NaN) pairs
    #If any NaNs exist, create df of them
    na = df.isna().any()
    print("NaN present in column:")
    print(na, "\n")

    if any(na) == True:
        df_na = df[df.isna().any(axis=1)]
        print(df_na)

    #Get count of zero values and print
    counts = []
    print("Zero values in column:")
    for column in df.columns:
        counts.append([column, (df[column].values == 0).sum()])

    print(counts, "\n")

    #Check parametic assumtions for chloresterol, ready
    #to either run independent t-test or one-way ANOVA test with male/female

    #Drop zeros, these aren't part of the "true" distribution, they are to be inputed
    #And get the index match sexes
    ch = df['Cholesterol']
    sx = df['Sex']
    sx = sx[ch.values != 0]
    ch = ch[ch.values != 0]

    #Test normality
    sm.qqplot(ch, line='45', fit=True)
    plt.show()

    ks_test = smdiag.kstest_normal(ch, dist='norm')
    print(f'KS test for chloresterol: {ks_test}\n')


    #Test variance
    sns.boxplot(x=sx, y=ch)
    plt.show()

    return

def pre_processing(df):
    '''Perform preprocessing on the data set according to the EDA'''

    return df

def main(run_eda=False):
    df = pd.read_csv("heart.csv")

    if run_eda:
        eda(df)

    df = pre_processing(df)

    return df