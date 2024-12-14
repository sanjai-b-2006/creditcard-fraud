import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Separate the data into fraud and valid transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

# Calculate the fraction of fraud and valid transactions
Fraction = len(fraud) / float(len(valid))
print('Fraction of fraud and valid transactions:', Fraction)
print('Fraud Cases:', len(fraud))
print('Valid Transactions:', len(valid))

# Generate the correlation matrix
corrmat = data.corr()

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show(block=True)

# Prepare the data for training and testing
X = data.drop(['Class'], axis=1)
Y = data["Class"]
xData = X.values
yData = Y.values

# Split the data into training and testing sets (80% training, 20% testing)
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=25)

# Initialize the Random Forest Classifier
rfc = RandomForestClassifier()

# Fit the model on the training data
rfc.fit(xTrain, yTrain)

# Predict the labels on the test data
yPred = rfc.predict(xTest)

# Calculate the number of fraud cases
n_outliers = len(fraud)

# Calculate the number of errors in the predictions
n_errors = (yPred != yTest).sum()

# Calculate and print the evaluation metrics
acc = accuracy_score(yTest, yPred)
print("The accuracy is:", acc)
prec = precision_score(yTest, yPred)
print("The precision is:", prec)
rec = recall_score(yTest, yPred)
print("The recall is:", rec)
f1 = f1_score(yTest, yPred)
print("The F1-Score is:", f1)
MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is:", MCC)

# Generate the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)

# Plot the confusion matrix heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show(block=True)
