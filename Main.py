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
from xgboost import XGBClassifier

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
            #drop 4th column, RestingBP, optionally
            'drop_col':['passthrough', drop_col_transformer([3])],
            'clf':[LogisticRegression()],
            'clf__C':C,
            'clf__penalty':['none', 'l2', 'l1', 'elasticnet'],
            'clf__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        {
            #drop 4th column, RestingBP, optionally
            'drop_col':['passthrough', drop_col_transformer([3])],
            'clf':[SVC()],
            'clf__C':C,
            'clf__kernel':['linear', 'rbf'],
            'clf__gamma':gamma
        },
        {
            'drop_col':['passthrogh', drop_col_transformer([3])],
            'clf':[XGBClassifier()],
            'clf__n_estimators':[100, 250, 500],
            'clf__min_child_weight':[1, 3, 5, 10],
            'clf__learning_rate': [1, 0.1, 0.01, 0.001],
            'clf__max_depth': [3, 4, 5, 8, 10, None],
            'clf__subsample': [0.6, 0.8, 1.0],
            'clf__colsample_bytree': [0.6, 0.8, 1.0],
            'clf__gamma': gamma
        }
    ]

    #Just a test for the moment, may not stick with this CV method or the f1 metric

    grid = GridSearchCV(pipe, n_jobs=7, param_grid=param_grid, cv=10, scoring='f1', verbose=2)
    grid.fit(X, y)
    results = pd.DataFrame(grid.cv_results_)
    print(results)

    #litte bit of code for dumping the result header, to see what's available
    #for col in results.columns:
    #    print(col)

if __name__ == '__main__':
    main()