
# Import required libraries
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

#load the dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.
print(X.shape);print(y.shape)


#use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#create a model and train it
mlp = MLPClassifier(hidden_layer_sizes=(50,),
                    activation='relu',
                    solver='adam',
                    max_iter=10,
                    verbose=True)
mlp.fit(X_train, y_train)

#print the prediction result
print('accuracy for train set:', mlp.score(X_train, y_train))
print('accuracy for test set:', mlp.score(X_test, y_test))