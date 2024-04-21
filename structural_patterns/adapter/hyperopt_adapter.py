from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from hyperopt import STATUS_OK, tpe, hp, fmin, Trials

class HyperoptAdapter(BaseEstimator):
    def __init__(self, estimator, param_grid, n_trials=10, cv=5, scoring=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        space = {}
        for param_name, param_values in self.param_grid.items():
            space[param_name] = hp.choice(param_name, param_values)

        best = fmin(
            fn=self._objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=Trials()
        )

        best_params = {k: v["val"] for k, v in best.items()}
        self.best_estimator_ = self.estimator.set_params(**best_params)

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def _objective(self, params):
        self.estimator.set_params(**params)
        return {"loss": -self.estimator.score(X, y), "status": STATUS_OK}


if __name__ == '__main__':
    # Example usage:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load dataset
    iris = load_iris()
    
    # Define the search space
    adapter = HyperoptAdapter(estimator=RandomForestClassifier(), param_grid={'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]})
    adapter.fit(X=iris.data, y=iris.target)

    adapter.predict(X=iris.data)