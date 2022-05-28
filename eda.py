import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smdiag
from scipy.stats import mannwhitneyu

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
        plt.suptitle("Histograms of feature and target variables")

    #create pairplot
    sns.pairplot(df, corner=True)
    plt.suptitle("Pair Plot of feature and target variables")
    plt.tight_layout()
    plt.show()

    #create correlation plot
    cr = df.corr()
    sns.heatmap(cr, center=0, annot=True).set(title="Correlations of feature and target variables")
    plt.tight_layout()
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

    #Check parametric assumtions for cholesterol, ready
    #to either run independent t-test or alternative test with male/female

    #Drop zeros, these aren't part of the "true" distribution, they are to be imputed
    #And get the index match sexes
    ch = df['Cholesterol']
    sx = df['Sex']
    sx = sx[ch.values != 0]
    ch = ch[ch.values != 0]

    #Test normality
    _, ax = plt.subplots()
    ax.set_title("QQ plot for non-zero cholesterol values")
    sm.qqplot(ch, line='45', fit=True, ax=ax)
    plt.show()

    ks_test = smdiag.kstest_normal(ch, dist='norm')
    print(f'K-S test for cholesterol: {ks_test}\n')

    #Test variance
    sns.boxplot(x=sx, y=ch).set(title="Cholesterol for male and female patients")
    plt.show()

    #Fails parametric contraints on normality grounds
    #Therefore, will procede using Mann Whitney U Test

    male_ch = ch[sx == 'M']
    female_ch = ch[sx == 'F']

    U, p = mannwhitneyu(male_ch, female_ch, method="auto")
    print(f'Mann Witney U test for cholesterol male/female groups: ({U}, {p})')

    return
