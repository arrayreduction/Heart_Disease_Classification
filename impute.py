def impute_ch(df, verbose=False):
    '''Perform imputation on the data set according to the EDA'''

    df = df.copy()

    #Impute zero values in Cholesterol by male/female groups
    #as EDA showed significate diffence in thier distribution

    male_ch = df[df['Sex'] == 'M']['Cholesterol']
    male_ch = male_ch[male_ch != 0]
    mu_male = male_ch.mean()

    female_ch = df[df['Sex'] == 'F']['Cholesterol']
    female_ch = female_ch[female_ch != 0]
    mu_female = female_ch.mean()

    #df.where replaces when condition resolves false , therefore need to flip our condtion
    df['Cholesterol'].where((df['Sex'] == 'M') | (df['Cholesterol'] != 0), mu_female, inplace=True)
    df['Cholesterol'].where((df['Sex'] == 'F') | (df['Cholesterol'] != 0), mu_male, inplace=True)

    #Impute zero value in resting BP
    resting_bp = df[df['RestingBP'] != 0]['RestingBP']
    mu_bp = resting_bp.mean()

    df['RestingBP'].where((df['RestingBP'] != 0), mu_bp, inplace=True)

    if verbose:
        print(f'Male mean ch {mu_male}')
        print(f'Female mean ch {mu_female}')
        print(f'Mean bp {mu_bp}')

    return df