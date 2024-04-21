# ML Design Patterns

This repo is designed to showcase how to apply certain software design patterns in a ML context.

## Creation Patterns

### Factory Pattern

The Factory pattern is a design pattern that creates objects without specifying the exact class of object that will be created. This is useful when you want to be able to change the specific class of object that is created without having to change the code that is using the factory.

A good example of how you might use the Factory pattern in an ML context is when you want to be able to create different types of models without having to change the code that is using the models. For example, if you want to be able to easily switch between a linear regression model and a decision tree model, you could use the Factory pattern to create instances of these models based on some parameter or configuration.

An example implementation can be seen [here](creation_patterns/factory\model_factory.py).

### Builder Pattern

The Builder pattern is a design pattern that allows you to create complex objects step by step. This is useful when you have a lot of optional components that make up a larger object and you want to be able to create the object in a modular way.

A good example of how you might use the Builder pattern in an ML context is when you want to create an ML pipeline. For example, you might have a pipeline that consists of scaling, transformations, feature selection, and then finally training a model. Each of these steps could be a separate class or a simple function that is responsible for doing one aspect of the pipeline. Using the Builder pattern, you can create the pipeline by calling methods on the Builder class that add each step to the pipeline in the correct order.

An example implementation of an ML pipeline builder can be seen [here](creation_patterns/builder/ml_pipeline_builder.py).

## Structural Patterns

### Adapter Pattern

The Adapter pattern is a design pattern that allows you to use an existing class (the "adaptee") in a new context without modifying the existing class. This is useful when you have an existing class that has some of the methods or interfaces that you need, but which needs to be adapted to fit into a new context.

A good example of how you might use the Adapter pattern in an ML context is when you want to use an existing library or framework for hyperparameter search, but which has a different interface. For example, you might want to use Optuna, but have it also be compatible with scikit-learn's `GridSearchCV` class to tune hyperparameters. In this case, you could write an adapter class that implements the required steps for optuna and make it available using the same interface as `GridSearchCV`.

An example implementation can be seen [here](structural_patterns/adapter/hyperopt_adapter.py)

### Bridge Pattern

The Bridge pattern is a design pattern that allows you to decouple an abstraction from its implementation so that the two can vary independently. This is useful when you have a lot of different implementations of an abstraction and you want to be able to change the implementation without affecting the code that is using the abstraction.

A good example of this is sklearn's BaseEstimator, which separates thee model details from its usage.

## Behavioral Patterns

### Strategy Pattern

The Strategy pattern is a design pattern that allows you to switch between different algorithms or "strategies" at runtime. This is useful when you have different approaches to solving the same problem and you want to be able to switch between them easily.

A good example of how you might use the Strategy pattern in a ML context is in hyperparameter tuning, where you might try different frameworks or search strategies, such as GridSearch or RandomSearch.

An example implementation can be seen [here](behavioral_patterns/strategy/hyperparameter_tuning_strategy.py).
