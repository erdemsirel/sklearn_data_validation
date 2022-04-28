import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: Mean - Standard Deviation - Z-Score
# Column Order
# 

X_train = pd.read_csv("https://raw.githubusercontent.com/erdemsirel/ds555_hw3_scoring/master/X_train.csv")

class DataValidator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def outlier_detector(self,X, y=None):
        X = pd.DataFrame(X).copy()
        X.describe

    def fit(self,X,y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        return X
    
if __name__== "__main__":
