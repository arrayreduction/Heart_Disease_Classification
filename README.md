# Heart_Disease_Classification
Heart disease classification modelling based on data from https://www.kaggle.com/fedesoriano/heart-failure-prediction

The project covers exploratory data analysis, data-preprocessing, and model selection.

Main.py the central scipt, all other code are treated as modules for this script. There are two constants which must be set;

RUN_EDA: Boolean      Determines whether the exploratory data analysis is performed
INITIAL_FIT: Boolean  Determines whether to fit the fit the main classification models or load previously trained model paramaters from disk.
                      This prevents having to re-run the hyper-paramater tuning once initial models have been selected.
                      True: train new models, best models are saved to disk
                      False: load pre-trained model parameters from disk
