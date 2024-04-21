from abc import ABC, abstractmethod
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator

class ModelType(Enum):
    RANDOM_FOREST = 'random_forest'
    LOGISTIC_REGRESSION = 'logistic_regression'
    SVM = 'svm'
    LINEAR_REGRESSION = 'linear_regression'

class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, model_type: ModelType) -> BaseEstimator:
        pass

class ClassificationModelFactory(ModelFactory):
    def create_model(self, model_type: ModelType) -> BaseEstimator:
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier()
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression()
        elif model_type == ModelType.SVM:
            return SVC()
        else:
            raise ValueError("Invalid model type for classification")

class RegressionModelFactory(ModelFactory):
    def create_model(self, model_type: ModelType) -> BaseEstimator:
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor()
        elif model_type == ModelType.LINEAR_REGRESSION:
            return LinearRegression()
        elif model_type == ModelType.SVM:
            return SVR()
        else:
            raise ValueError("Invalid model type for regression")

if __name__ == '__main__':
    # Example usage:
    classification_factory = ClassificationModelFactory()
    regression_factory = RegressionModelFactory()

    # Create classification models
    rf_classifier = classification_factory.create_model(ModelType.RANDOM_FOREST)
    lr_classifier = classification_factory.create_model(ModelType.LOGISTIC_REGRESSION)
    svm_classifier = classification_factory.create_model(ModelType.SVM)

    # Create regression models
    rf_regressor = regression_factory.create_model(ModelType.RANDOM_FOREST)
    lr_regressor = regression_factory.create_model(ModelType.LINEAR_REGRESSION)
    svm_regressor = regression_factory.create_model(ModelType.SVM)