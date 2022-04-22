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

    #create pairplot
    sns.pairplot(df, corner=True)
    plt.show()

    #create correlation plot
    cr = df.corr()
    sns.heatmap(cr, center=0, annot=True)
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
    #to either run independent t-test or alternative test with male/female

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
    print(f'K-S test for chloresterol: {ks_test}\n')


    #Test variance
    sns.boxplot(x=sx, y=ch)
    plt.show()

    #Fails parametric contraints on normaliry grounds
    #Therefore, will procede using Mann Whitney U Test

    male_ch = ch[sx == 'M']
    female_ch = ch[sx == 'F']

    U, p = mannwhitneyu(male_ch, female_ch, method="auto")
    print(f'Mann Witney U test for chloresterol male/female groups: ({U}, {p})')

    return

def pre_processing(df):
    '''Perform preprocessing on the data set according to the EDA'''

    df = df.copy()

    #Impute zero values in Cholesterol by male/female groups
    #as EDA showed significate diffence in thier distribution

    male_ch = df[df['Sex'] == 'M']['Cholesterol']
    male_ch = male_ch[male_ch != 0]
    mu_male = male_ch.mean()
    print(f'Male mean ch {mu_male}')

    female_ch = df[df['Sex'] == 'F']['Cholesterol']
    female_ch = female_ch[female_ch != 0]
    mu_female = female_ch.mean()
    print(f'Female mean ch {mu_female}')

    #df.where replaces when condition resolves false , therefore need to flip our condtion
    df['Cholesterol'].where((df['Sex'] == 'M') | (df['Cholesterol'] != 0), mu_female, inplace=True)
    df['Cholesterol'].where((df['Sex'] == 'F') | (df['Cholesterol'] != 0), mu_male, inplace=True)

    #Impute zero value in resting BP
    resting_bp = df[df['RestingBP'] != 0]['RestingBP']
    mu_bp = resting_bp.mean()
    print(f'Mean bp {mu_bp}')
    
    df['RestingBP'].where((df['RestingBP'] != 0), mu_bp, inplace=True)

    return df

def main(run_eda=False):
    df = pd.read_csv("heart.csv")

    if run_eda:
        eda(df)

    df = pre_processing(df)

    return df