
# Import required libraries
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

# load the data
df = pd.read_csv('diabetes.csv')
print(df.shape)
print(df.describe().transpose())

# separate predictor and target then normalize the data
target_column = ['Outcome']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
print(X_train.shape); print(X_test.shape)

# create a model
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500) # 3 hidden layer with 8 hidden unit each
mlp.fit(X_train,y_train)


predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))


