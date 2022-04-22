import numpy as np

class drop_col_transformer():
    '''Class to provide a transformer for dropping columns specified at construction, adapted
    code from https://stackoverflow.com/questions/68402691/adding-dropping-column-instance-into-a-pipeline.
     to use numpy arrays for use within sklearn pipeline'''
    def __init__(self, columns):
        self.columns=columns

    def transform(self, X, y=None):
        X = X.copy()
        X = np.delete(X, self.columns, axis=1)

        return X

    def fit(self, X, y=None):
        return self 