from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load your data
# Assuming you have loaded your data into X and y

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grids for Perceptron and MLPClassifier
perceptron_param_grid = {
    'max_iter': [100, 200, 300, 400, 500],
    'tol': [1e-3, 1e-4, 1e-5]
}

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [200, 300, 400],
    'tol': [1e-3, 1e-4, 1e-5]
}

try:
    # Hyperparameter tuning for Perceptron
    perceptron = Perceptron()
    perceptron_random_search = RandomizedSearchCV(perceptron, perceptron_param_grid, n_iter=10, cv=2, random_state=42)
    perceptron_random_search.fit(X_train, y_train)

    # Hyperparameter tuning for MLPClassifier
    mlp = MLPClassifier()
    mlp_random_search = RandomizedSearchCV(mlp, mlp_param_grid, n_iter=10, cv=2, random_state=42)
    mlp_random_search.fit(X_train, y_train)

    # Print results
    print("Perceptron Best Parameters:", perceptron_random_search.best_params_)
    print("Perceptron Accuracy:", perceptron_random_search.best_score_)
    print("MLP Best Parameters:", mlp_random_search.best_params_)
    print("MLP Accuracy:", mlp_random_search.best_score_)

except Exception as e:
    print("An error occurred:", e)
