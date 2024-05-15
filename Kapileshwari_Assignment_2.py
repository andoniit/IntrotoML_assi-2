# ASSIGMENT 2
# Name-Anirudha Kapileshwari
# Spring 2024 Introduction to Machine Learning (CS-484-01)
# prof Shouvik Roy


# Problem 1: Perceptron Learning (15 marks)
#The dataset lab02_dataset_1.csv has a 3-dimensional input space and a class label of Positive and Negative. For this task, you are not allowed to use any functionalities of the sklearn module.
#1. Write a function my_perceptron( ) which applies perceptron algorithm on the dataset to create a linear separator. my_perceptron( ) should return a 3-dimensional weight vector which can be used to create the linear separator. Use a classification threshold of 99% i.e., my_perceptron( ) will terminate once the misclassification rate is less than 1%. (10 marks)
#2. Create a 3D plot which showcases the dataset in a 3D-space alongwith the linear separator you obtained from my_perceptron( ). Use two different colors to represent the data points belonging in the two classes for ease of viewing. (5 marks)


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



#####################################################################################################################################################################################################################################

#Problem 2: Naïve Bayes Learning (25 marks)
#The dataset lab02_dataset_2.xlsx contains 10,302 observations on various vehicles. You will use the observations in this dataset to train models that predict the usage of a vehicle. Your models will use the following variables:
#Output Label:
#• CAR_USE. Vehicle Usage. It has two categories, namely, Commercial and Private.
#Input Features:
#• CAR_TYPE. Vehicle Type. It has six categories, namely, Minivan, Panel Truck, Pickup, SUV, Sports Car, and Van.
#• OCCUPATION. Occupation of Vehicle Owner. It has nine categories, namely, Clerical, Home Maker, Doctor, Lawyer, Manager, Professional, Blue Collar, Student, and Unknown.
#• EDUCATION. Highest Education Level of Vehicle Owner. It has five categories namely Below High Sc, High School, Bachelors, Masters, PhD.
#You will use only observations where there are no missing values in all the above four variables. After dropping the missing values, you will use all the 100% complete observations for training your Naïve Bayes models using sklearn. For each observation, you will calculate the predicted probabilities for CAR_USE = Commercial and CAR_USE = Private. You will classify the observation in the CAR_USE category that has the highest predicted probability. In case of ties, choose Private category as the output.
#1. You will train a Naïve Bayes model with a Laplace smoothing of 0.01. (5 marks)
#2. Output the Class counts and Probabilities P(Yj). Also display the probability of the
#input variables, given each output label P(Xi|Yj) alongwith their counts. (5 marks)
#3. Let us study a couple of fictitious persons (test cases). One person works in a Blue Collar occupation, has an education level of PhD, and owns an SUV. Another person works in a Managert occupation, has a Below High Sc level of education, and owns a Sports Car. What are the Car Usage probabilities of both these people? (5 marks)
#4. Generate a histogram of the predicted probabilities of CAR_USE = Private. The
#bin width is 0.05. The vertical axis is the proportion of observations. (5 marks)
#5. Finally, what is the misclassification rate of the Naïve Bayes model? (5 marks)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_excel("lab02_dataset_2.xlsx")

# Drop rows with missing values in the specified columns
dataset.dropna(subset=['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE'], inplace=True)

# Encode categorical variables into numerical format
categorical_encoders = {}
for column in ['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']:
    encoder = LabelEncoder()
    dataset[column] = encoder.fit_transform(dataset[column])
    categorical_encoders[column] = {label: idx for idx, label in enumerate(encoder.classes_)}

# Split dataset into features and target
features = dataset[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
target = dataset['CAR_USE']

# Split data into training and test sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_classifier = CategoricalNB(alpha=0.01)
nb_classifier.fit(features_train, target_train)

# Calculate class counts and probabilities P(Yj)
target_counts = dataset['CAR_USE'].value_counts()
total_count = target_counts.sum()
target_probabilities = target_counts / total_count

print("Class Counts=>")
print(target_counts)
print("\nProbabilities P(Yj)=>")
print(target_probabilities)

# Calculate the probability of the input variables given each output label P(Xi|Yj)
grouped_data = dataset.groupby(['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']).size().reset_index(name='occurrences')

# Calculate conditional probabilities P(Xi|Yj)
for idx, row in grouped_data.iterrows():
    conditional_prob = row['occurrences'] / target_counts[row['CAR_USE']]
    print(f"\nP({row['CAR_TYPE']}, {row['OCCUPATION']}, {row['EDUCATION']} | CAR_USE={row['CAR_USE']}) = {conditional_prob:.6f}, occurrences = {row['occurrences']}")

# Plot histogram of predicted probabilities for CAR_USE = Private
probabilities_private_use = nb_classifier.predict_proba(features_test)[:, 1]
plt.hist(probabilities_private_use, bins=np.arange(0, 1.05, 0.05), alpha=0.7)
plt.title('Histogram of Predicted Probabilities for CAR_USE = Private')
plt.xlabel('Predicted Probability of Private Use')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Calculate and print the misclassification rate
predictions = nb_classifier.predict(features_test)
model_accuracy = accuracy_score(target_test, predictions)
misclassification_rate = 1 - model_accuracy

print("\nMisclassification Rate:", misclassification_rate)