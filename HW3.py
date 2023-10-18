# Import necessary libraries
import pandas as pd  # Importing the Pandas library for data manipulation
from sklearn.model_selection import train_test_split  #  For splitting the data into training and validation sets
from sklearn.ensemble import RandomForestClassifier  # Importing the Random Forest classifier
from sklearn.tree import DecisionTreeClassifier  # Importing the Decision Tree classifier
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from sklearn.model_selection import cross_val_score  # Importing cross-validation for model evaluation
from sklearn import tree  # Importing tree module for plotting decision trees
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing data

train_data = pd.read_csv("train.csv")  # Loading the training data from a CSV file.

imputer = SimpleImputer(strategy='mean')  # Create an imputer to fill missing age values with the mean values
train_data['Age'] = imputer.fit_transform(train_data[['Age']])  # Filling missing 'Age' values with the mean
train_data['Sex'] = train_data['Sex'].map({'female': 0, 'male': 1})  # Just converting Sex values into Numeric.

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True) # Filling the "NAN" values with most frequent data.

train_data = pd.get_dummies(train_data, columns=['Embarked']) # Doing one-hot encoding, converting into numeric values.

# Setting the attributes as inputs and outputs.
X = train_data[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']

# Splitting the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessed Data")
print(X_train.head())  # Displaying the preprocessed data

clf = DecisionTreeClassifier(max_depth=3)  # Creating a Decision Tree classifier with a maximum depth of 3
clf.fit(X_train, y_train)  # Training the Decision Tree model
print(" Decision Tree Model Built")

scores_dt = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')  # Performing 5 fold cross-validation
average_accuracy_decision_tree = scores_dt.mean()  # Calculating the average accuracy of Decision Tree
print("5 fold Cross-validation for Decision Tree is done")
print("Average Accuracy for Decision Tree:", average_accuracy_decision_tree)  # Displaying the average accuracy

# Plot the Decision Tree
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()  # Displaying the Decision Tree plot

# Build and evaluate a Random Forest model
rf = RandomForestClassifier()  # Create a Random Forest classifier
rf.fit(X_train, y_train)  # Train the Random Forest model
print(" Random Forest Model Built")

# Performing 5-fold cross-validation for Random Forest
scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')  
average_accuracy_random_forest = scores_rf.mean()  # Calculating the average accuracy

print("5 fold Cross-validation for Random Forest is done")
print("Average Accuracy for Random Forest:", average_accuracy_random_forest)  # Display the average accuracy

# Step 7: Compare the models and provide conclusions
if average_accuracy_decision_tree > average_accuracy_random_forest:
    print("Random Forest is better.")  # If Random Forest has higher accuracy, declare it as better
else:
    print("Decision Tree is better.")  # If Decision Tree has higher accuracy, declare it as better
