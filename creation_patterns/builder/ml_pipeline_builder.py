from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator

class MLPipelineBuilder:
    def __init__(self):
        self.pipeline = Pipeline(steps=[('empty', None)])
    
    def add_scaler(self):
        self.pipeline.steps.append(('scaler', StandardScaler()))
        return self
    
    def add_pca(self, n_components):
        self.pipeline.steps.append(('pca', PCA(n_components=n_components)))
        return self
    
    def add_feature_selection(self, k):
        self.pipeline.steps.append(('feature_selection', SelectKBest(k=k)))
        return self
    
    def add_model(self, model: BaseEstimator):
        self.pipeline.steps.append(('model', model))
        return self
    
    def build(self) -> Pipeline:
        return self.pipeline

if __name__ == '__main__':
    # Example usage:
    from sklearn.ensemble import RandomForestClassifier

    # Create pipeline builder
    pipeline_builder = MLPipelineBuilder()

    # Build pipeline
    pipeline = pipeline_builder.add_scaler()\
                                .add_pca(n_components=5)\
                                .add_feature_selection(k=10)\
                                .add_model(RandomForestClassifier())\
                                .build()

