import pandas as pd
from eda import eda
import yaml
from yaml.loader import FullLoader
from impute import impute_ch
from transformers import drop_col_transformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pickle import dump, load
import matplotlib.pyplot as plt

def main():
    RUN_EDA = False
    INITIAL_FIT = False

    df = pd.read_csv("heart.csv")

    if RUN_EDA:
        eda(df)
    
    df = impute_ch(df)

    #Split into X and y
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    #Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=347)

    #Make Column Transformer for scaling numerical features (Min/Max 0 to 1)
    #and one hot encoding categorical features

    #Note, FastingBS is technically categorical, but is already 0/1 encoded. 
    #As minmax scaler uses y = (x – min) / (max – min), 0/1 encodings are unaltered
    #and we have just wasted a small amount of CPU time
    num_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    num_transformer = Pipeline(
        [
            ('scaler', MinMaxScaler((0,1)))
        ]
    )

    cat_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    cat_transformer = Pipeline(
        [
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
    gamma_svm = gamma.copy()
    gamma_svm.append('scale')

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
            'clf__gamma':gamma_svm,
            'clf__class_weight':['balanced', None]
        },
        {
            'drop_col':['passthrough', drop_col_transformer([3])],
            'clf':[XGBClassifier(use_label_encoder=False)],
            'clf__n_estimators':[100, 250, 500],
            'clf__min_child_weight':[1, 3, 5, 10],
            'clf__learning_rate': [1, 0.1, 0.01, 0.001],
            'clf__max_depth': [3, 4, 5, 8, 10, None],
            'clf__subsample': [0.6, 0.8, 1.0],
            'clf__colsample_bytree': [0.6, 0.8, 1.0],
            'clf__gamma': gamma,
            'clf__eval_metric':['logloss']
        }
    ]

    #Search over the parameters

    if INITIAL_FIT:
        scoring={"AUC":'roc_auc', "Accuracy":'accuracy', "F1":'f1', "Prec":'precision', "Recall":'recall'}
        grid = GridSearchCV(pipe, n_jobs=7, param_grid=param_grid, cv=10, scoring=scoring, refit='F1', verbose=0)
        grid.fit(X_train, y_train)
        results = grid.cv_results_

        df_results = pd.DataFrame(results)

        #Push to excel for exmination
        #df_results.to_excel("cv_results.xlsx")

        #Get best parameters for each model
        LR = df_results[df_results['param_clf'].astype(str).str.contains("LogisticRegression", regex=False)]
        XGB = df_results[df_results['param_clf'].astype(str).str.contains("XGBClassifier", regex=False)]
        SV = df_results[df_results['param_clf'].astype(str).str.contains("SVC", regex=False)]

        LR_best = LR[LR['rank_test_F1'] == LR['rank_test_F1'].min()]
        XGB_best = XGB[XGB['rank_test_F1'] == XGB['rank_test_F1'].min()]
        SV_best = SV[SV['rank_test_F1'] == SV['rank_test_F1'].min()]

        print(LR_best.rank_test_F1, XGB_best.rank_test_F1, SV_best.rank_test_F1, "\n")
        print(LR_best.mean_test_F1, XGB_best.mean_test_F1, SV_best.mean_test_F1, "\n")

        #Examination of the 24 tied logisitic regression models (viewing table in excel)
        #shows they are tied in all scored metrics except AUC, so get the best of the tied results by AUC_ROC

        LR_best = LR_best[LR_best['rank_test_AUC'] == LR_best['rank_test_AUC'].min()]

        #Get a dictionary representation of the parameters
        #Cast to sting, strip index and leading space, then evaluate to dict
        #NB this isn't the best way of doing this, just the way I thought of at the time.
        pd.set_option('max_colwidth', 1500)
        LR_best_params = LR_best['params'].to_string(index=False)
        XGB_best_params = XGB_best['params'].to_string(index=False)
        SV_best_params = SV_best['params'].to_string(index=False)
        pd.reset_option('max_colwidth')

        LR_best_params = yaml.load(LR_best_params, Loader=FullLoader)
        XGB_best_params = yaml.load(XGB_best_params, Loader=FullLoader)
        SV_best_params = yaml.load(SV_best_params, Loader=FullLoader)

        #Deal with the junk with got added to the paramaters, these are mostly duplicated keys
        #The end result is the best model for each classifier
        LR_best_params.update({'clf': LogisticRegression()})
        XGB_best_params.update({'clf': XGBClassifier()})
        SV_best_params.update({'clf': SVC(), 'clf__class_weight': None})
        del SV_best_params['gamma=0.001)']

        def del_all(dictionary, to_remove):
            """Remove list of elements from mapping."""
            for key in to_remove:
                del dictionary[key]

        remove = ['booster=None','colsample_bylevel=None','colsample_bynode=None','colsample_bytree=None',
                            'enable_categorical=False','gamma=None','gpu_id=None','importance_type=None','interaction_constraints=None',
                            'learning_rate=None','max_delta_step=None','max_depth=None', 'min_child_weight=None', 'missing=nan',
                            'monotone_constraints=None', 'n_estimators=100', 'n_jobs=None', 'num_parallel_tree=None', 'predictor=None',
                            'random_state=None','reg_alpha=None','reg_lambda=None','scale_pos_weight=None','subsample=None','tree_method=None',
                            'use_label_encoder=False','validate_parameters=None','verbosity=None)']
        
        del_all(XGB_best_params, remove)

        print(LR_best_params, "\n")
        print(XGB_best_params, "\n")
        print(SV_best_params, "\n")

        #Dump best_params to disk, so we don't have to re-run post processing
        with open("LR_best_params.pickle", 'wb') as file:
            dump(LR_best_params, file)
        
        with open("XGB_best_params.pickle", 'wb') as file:
            dump(XGB_best_params, file)
        
        with open("SV_best_params.pickle", 'wb') as file:
            dump(SV_best_params, file)

        with open("LR_best.pickle", 'wb') as file:
            dump(LR_best, file)
        
        with open("XGB_best.pickle", 'wb') as file:
            dump(XGB_best, file)
        
        with open("SV_best.pickle", 'wb') as file:
            dump(SV_best, file)

        #Dump results to disk, so we don't have to re-run the CV at any point
        with open("results_df.pickle", 'wb') as file:
            dump(results, file)
    else:
        with open("results_df.pickle", 'rb') as file:
            results = load(file)

        with open("LR_best_params.pickle", 'rb') as file:
            LR_best_params = load(file)

        with open("XGB_best_params.pickle", 'rb') as file:
            XGB_best_params = load(file)

        with open("SV_best_params.pickle", 'rb') as file:
            SV_best_params = load(file)

        with open("LR_best.pickle", 'rb') as file:
            LR_best = load(file)

        with open("XGB_best.pickle", 'rb') as file:
            XGB_best = load(file)

        with open("SV_best.pickle", 'rb') as file:
            SV_best = load(file)

    print()

    print(f"Logistic train: f1 {LR_best.mean_test_F1}, prec {LR_best.mean_test_Prec}, recall {LR_best.mean_test_Recall}, \
        accuracy {LR_best.mean_test_Accuracy}, AUC {LR_best.mean_test_AUC}")
    print(f"XGB train: f1 {XGB_best.mean_test_F1}, prec {XGB_best.mean_test_Prec}, recall {XGB_best.mean_test_Recall}, \
        accuracy {XGB_best.mean_test_Accuracy}, AUC {XGB_best.mean_test_AUC}")
    print(f"SVM train: f1 {SV_best.mean_test_F1}, prec {SV_best.mean_test_Prec}, recall {SV_best.mean_test_Recall}, \
        accuracy {SV_best.mean_test_Accuracy}, AUC {SV_best.mean_test_AUC}")

    #Create an ensemble/stacked model using the three existing best models
    #We reuse the existing pipeline for the first level

    level0 = []
    level0.append(('LR', pipe.set_params(**LR_best_params)))
    level0.append(('XGB', pipe.set_params(**XGB_best_params)))
    level0.append(('SV', pipe.set_params(**SV_best_params)))
    level1 = "passthrough"
    stack = StackingClassifier(estimators=level0, final_estimator=level1)

    C = [1, 10, 25, 50, 100]
    alpha = [10, 1 , 0.1, 0.001, 0.0001]

    param_grid = [{
        'final_estimator':[LogisticRegression()],
        'final_estimator__C': C,
        'final_estimator__penalty':['none', 'l2', 'l1', 'elasticnet'],
        'final_estimator__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'passthrough':[False]
        },
        {
        'final_estimator':[MultinomialNB()],
        'final_estimator__alpha':alpha,
        'passthrough':[False]
        }]

    scoring={"AUC":'roc_auc', "Accuracy":'accuracy', "F1":'f1', "Prec":'precision', "Recall":'recall'}
    grid = GridSearchCV(estimator=stack, param_grid=param_grid, cv=10, scoring=scoring, refit='F1', n_jobs=7, verbose=1)
    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)
    best_index_ = grid.best_index_

    f1_test = results['mean_test_F1'][best_index_]
    prec_test = results['mean_test_Prec'][best_index_]
    rec_test = results['mean_test_Recall'][best_index_]
    acc_test = results['mean_test_Accuracy'][best_index_]
    auc_test = results['mean_test_AUC'][best_index_]

    print(f"Stacked Train: f1 {f1_test}, prec {prec_test}, recall {rec_test}, accuracy {acc_test}, AUC {auc_test} \n")

    #Now put each model against the validation set, to see if there's evidence of overfitting on the
    #training set. We retrain on the whole training set and test against the validation set

    final_est = grid.best_estimator_.final_estimator
    stack.final_estimator = final_est
    stack.set_params(**grid.best_params_)

    print("Best stacked estimator: \n")
    print(grid.best_params_)

    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)[:, 1]

    f1_test = f1_score(y_true=y_test,y_pred=y_pred)
    prec_test = precision_score(y_true=y_test,y_pred=y_pred)
    rec_test = recall_score(y_true=y_test,y_pred=y_pred)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_proba)    #Requires predict_proba
    acc_test = accuracy_score(y_true=y_test,y_pred=y_pred)
    print(f"Stacked Test: f1 {f1_test}, prec {prec_test}, recall {rec_test}, accuracy {acc_test}, AUC {auc_test}")

    pipe.set_params(**LR_best_params).fit(X_train, y_train)
    y_pred = pipe.set_params(**LR_best_params).predict(X_test)
    y_proba = pipe.set_params(**LR_best_params).predict_proba(X_test)[:, 1]

    f1_test = f1_score(y_true=y_test,y_pred=y_pred)
    prec_test = precision_score(y_true=y_test,y_pred=y_pred)
    rec_test = recall_score(y_true=y_test,y_pred=y_pred)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_proba)    #Requires predict_proba
    acc_test = accuracy_score(y_true=y_test,y_pred=y_pred)
    print(f"LR Test: f1 {f1_test}, prec {prec_test}, recall {rec_test}, accuracy {acc_test}, AUC {auc_test}")

    pipe.set_params(**XGB_best_params, clf__use_label_encoder=False).fit(X_train, y_train)
    y_pred = pipe.set_params(**XGB_best_params).predict(X_test)
    y_proba = pipe.set_params(**XGB_best_params).predict_proba(X_test)[:, 1]

    f1_test = f1_score(y_true=y_test,y_pred=y_pred)
    prec_test = precision_score(y_true=y_test,y_pred=y_pred)
    rec_test = recall_score(y_true=y_test,y_pred=y_pred)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_proba)    #Requires predict_proba
    acc_test = accuracy_score(y_true=y_test,y_pred=y_pred)
    print(f"XGB Test: f1 {f1_test}, prec {prec_test}, recall {rec_test}, accuracy {acc_test}, AUC {auc_test}")

    pipe.set_params(**SV_best_params, clf__probability=True).fit(X_train, y_train)
    y_pred = pipe.set_params(**SV_best_params).predict(X_test)
    y_proba = pipe.set_params(**SV_best_params).predict_proba(X_test)[:, 1]

    f1_test = f1_score(y_true=y_test,y_pred=y_pred)
    prec_test = precision_score(y_true=y_test,y_pred=y_pred)
    rec_test = recall_score(y_true=y_test,y_pred=y_pred)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_proba)    #Requires predict_proba
    acc_test = accuracy_score(y_true=y_test,y_pred=y_pred)
    print(f"SV Test: f1 {f1_test}, prec {prec_test}, recall {rec_test}, accuracy {acc_test}, AUC {auc_test}")

    #Get a set of confustion matrices for testing data
    clf = stack.set_params(**grid.best_params_)
    cm = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    cm.ax_.set_title("Confusion matrix for stacked ensemble classifier")
    plt.show()

    clf = pipe.set_params(**LR_best_params).fit(X_train, y_train)
    cm = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    cm.ax_.set_title("Confusion matrix for logisitic regression classifier")
    plt.show()

    clf = pipe.set_params(**XGB_best_params).fit(X_train, y_train)
    cm = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    cm.ax_.set_title("Confusion matrix for XGBoost classifier")
    plt.show()

    clf = pipe.set_params(**SV_best_params).fit(X_train, y_train)
    cm = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    cm.ax_.set_title("Confusion matrix for support vector machine classifier")
    plt.show()

if __name__ == '__main__':
    main()