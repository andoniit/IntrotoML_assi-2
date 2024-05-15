import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
###Implementation of my_perceptron
def preprocess_data(data):
    processed_data = np.copy(data)
    processed_data[:, -1] = np.where(processed_data[:, -1] == 'Positive', 1, 0) #Perceptron algorithm matrix
    return processed_data.astype(float)
def my_perceptron(X, y):

    # Add bias term to input data
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # Initialize weights randomly
    weights = np.random.rand(X_with_bias.shape[1])

    #Threshold for 99% accuracy to terminate once the missclassification rate is less than 1%
    threshold = 0.99 * len(y)

    misclassification_rate = len(y) #MissClassification tracking
    iterations = 0

    while misclassification_rate > 0.01 * len(y):
        misclassified_count = 0

        for i, x in enumerate(X_with_bias):
            prediction = np.dot(weights, x)
            if (prediction >= 0) != (y[i] == 1):
                misclassified_count += 1
                if y[i] == 1:
                    weights += x
                else:
                    weights -= x

        misclassification_rate = misclassified_count
        iterations += 1

        # Prevent infinite loops in case data is not linearly separable
        if iterations > 1000:
            print("Failed to converge.")
            return weights

    return weights #final weights based on the perceptron function calculation

#3D plot to showcase the dataset in 3D space
def plot_3d_dataset_with_separator(data, weights):
    
    # Separate features and labels
    X = data[:, :-1]
    y = data[:, -1]

    # Create a meshgrid for decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Compute corresponding z for decision boundary
    zz = (-weights[0] - weights[1] * xx - weights[2] * yy) / weights[3]

    # Plot dataset points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], c='green', label='Negative')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], c='orange', label='Positive')

    # Plot decision boundary
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Dataset with Linear Separator') #Linear Seperator
    ax.legend()
    plt.show()

# Read input data from CSV
data = pd.read_csv("lab02_dataset_1.csv")

# Preprocess data
data['Class'] = data['Class'].map({'Positive': 1, 'Negative': 0})
processed_data = data.values

# Train perceptron model
weights = my_perceptron(processed_data[:, :-1], processed_data[:, -1])

# Final Weights
print(weights)

# Plot 3D dataset with linear separator
plot_3d_dataset_with_separator(processed_data, weights)