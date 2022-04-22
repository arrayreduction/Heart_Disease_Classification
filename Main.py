import pandas as pd
from eda import eda
from impute import impute_ch
from transformers import drop_col_transformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def main():
    RUN_EDA = False

    df = pd.read_csv("heart.csv")

    if RUN_EDA:
        eda(df)
    
    df = impute_ch(df)

    #ADD TEST TRAIN SPLIT? OR JUST RELY ON CV WITHOUT HELD OUT? TBC

    #Split into X and y
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    #Make Column Transformer for scaling numerical features (Min/Max 0 to 1)
    #and one hot encoding categorical features

    num_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    num_transformer = Pipeline(
        [
            #potentially turn my imputer into a class and add here
            #('imputer', ),
            ('scaler', MinMaxScaler((0,1)))
        ]
    )

    cat_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    cat_transformer = Pipeline(
        [
            #CONSIDER, DO WE NEED TO LOOK AT LABEL ENCODER FOR TREE BASED METHODS?
            #COULD change 'prep' in pipeline to passthrough, and select different
            #Preprocessing column transformers with parameters
            ('ohe', OneHotEncoder())
        ]
    )

    preprocess = ColumnTransformer(
        transformers = [
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ]
        )

    pipe = Pipeline(
        [
            #drop_col and classifiers set through params
            ('prep', preprocess),
            ('drop_col', 'passthrough'),
            ('clf','passthrough')
        ]
    )

    C = [1, 10, 100]
    gamma = [0.001, 0.01, 0.1, 1, 10]

    param_grid = [
        {
            #drop 3rd column, RestingBP, optionally
            'drop_col':['passthrough', drop_col_transformer([3])],
            'clf':[LogisticRegression()],
            'clf__C':C,
            'clf__penalty':['none', 'l2']
        },
        {
            #drop 3rd column, RestingBP, optionally
            'drop_col':['passthrough', drop_col_transformer([3])],
            'clf':[SVC()],
            'clf__C':C,
            'clf__kernel':['linear', 'rbf'],
            'clf__gamma':gamma
        }
    ]

    #Just a test for the moment, may not stick with this CV method or the f1 metric

    grid = GridSearchCV(pipe, n_jobs=7, param_grid=param_grid, cv=10, scoring='f1', verbose=3)
    grid.fit(X, y)

if __name__ == '__main__':
    main()