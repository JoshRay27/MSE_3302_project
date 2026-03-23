import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestImageClassifier:
    def __init__(self, n_estimators=300, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        self.model.fit(self, X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        return acc