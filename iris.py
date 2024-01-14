# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# Loading the iris dataset
iris_dataset = load_iris()

# Printing information about the dataset
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))

# Printing the shape of data and information about the target
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Printing shapes of training and testing sets
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# Creating and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a new data point and make predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)

# Printing prediction results
print("X_new.shape: {}".format(X_new.shape))
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# Making predictions on the test set and evaluate the model
y_pred = knn.predict(X_test)
print("Test set prediction:\n {}".format(y_pred))
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, y_test)))

# Visualizing the results
# Scatter plot of the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
# Scatter plot of the testing data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Testing Data')
# Highlight the new data point
plt.scatter(X_new[0, 0], X_new[0, 1], c='red', marker='o', label='New Data Point')

# Set plot labels
plt.xlabel(iris_dataset['feature_names'][0])
plt.ylabel(iris_dataset['feature_names'][1])
plt.title('Iris Dataset - k-NN Classification')

# Showing legend
plt.legend()

# Showing the plot
plt.show()
