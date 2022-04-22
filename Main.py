import pandas as pd
import eda
import impute

def main():
    RUN_EDA = False

    df = pd.read_csv("heart.csv")

    if RUN_EDA:
        eda.eda(df)
    
    df = impute.impute_ch(df)

if __name__ == '__main__':
    main()