
# Import required libraries
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# load the data
df = pd.read_csv('diabetes.csv')
print(df.shape)
print(df.describe().transpose())

# separate predictor and target then normalize the data
target_column = ['Outcome']
predictors = list(set(list(df.columns))-set(target_column))

df[predictors] = df[predictors]/df[predictors].max()
X = df[predictors].values

y = df[target_column].values
y_ohe = tf.keras.utils.to_categorical(y, num_classes = 2)

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.3, random_state=40)
print(X_train.shape); print(X_test.shape)

# create a model and train it
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu')) # input layer 8, 1st later 8 with relu
model.add(Dense(8, activation='relu'))              # 2nd layer 8 with relu
model.add(Dense(8, activation='relu'))              # 3rd layer 8 with relu
model.add(Dense(2, activation='softmax'))           # output layer 2 with softmax
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            min_delta=1e-4,
                                            patience=10)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, batch_size=200,
          validation_data=(X_test, y_test), callbacks=[callback])

# evaluate the model
predict_train = model.predict(X_train)
print('confusion matrix for train set')
print(confusion_matrix(y_train.argmax(axis=1),predict_train.argmax(axis=1)))
print(classification_report(y_train.argmax(axis=1),predict_train.argmax(axis=1)))

predict_test = model.predict(X_test)
print('confusion matrix for test set')
print(confusion_matrix(y_test.argmax(axis=1),predict_test.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1),predict_test.argmax(axis=1)))
