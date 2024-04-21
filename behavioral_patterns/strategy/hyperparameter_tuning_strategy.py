from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from pandas import DataFrame

class HyperparameterTuningStrategy(ABC):
    @abstractmethod
    def tune_hyperparameters(self, model: BaseEstimator, X_train: DataFrame, y_train: DataFrame, params_grid: dict) -> BaseEstimator:
        pass

class GridSearchTuning(HyperparameterTuningStrategy):
    def tune_hyperparameters(self, model: BaseEstimator, X_train: DataFrame, y_train: DataFrame, params_grid: dict) -> BaseEstimator:
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

class RandomSearchTuning(HyperparameterTuningStrategy):
    def tune_hyperparameters(self, model: BaseEstimator, X_train: DataFrame, y_train: DataFrame, params_dist: dict) -> BaseEstimator:
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(estimator=model, param_distributions=params_dist, cv=5)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

class HyperparameterTuner:
    def __init__(self, tuning_strategy: HyperparameterTuningStrategy):
        self.tuning_strategy = tuning_strategy
    
    def set_tuning_strategy(self, tuning_strategy: HyperparameterTuningStrategy) -> None:
        self.tuning_strategy = tuning_strategy
    
    def tune(self, model: BaseEstimator, X_train: DataFrame, y_train: DataFrame, params: dict) -> BaseEstimator:
        return self.tuning_strategy.tune_hyperparameters(model, X_train, y_train, params)

if __name__ == '__main__':
    # Example usage:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestClassifier()

    # Define hyperparameters to tune
    params_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    params_dist = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
    }

    # Create a tuner with the desired strategy
    tuner = HyperparameterTuner(GridSearchTuning())

    # Tune hyperparameters
    best_model = tuner.tune(model, X_train, y_train, params_grid)

    # Alternatively, switch the strategy and tune again
    tuner.set_tuning_strategy(RandomSearchTuning())
    best_model_random = tuner.tune(model, X_train, y_train, params_dist)
