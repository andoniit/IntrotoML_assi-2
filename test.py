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