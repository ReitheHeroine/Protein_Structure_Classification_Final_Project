import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def row_validation(row):
    for cell in row:
        # Check for NaN, None, or blank strings and filter them out
        if cell is None or cell == '' or (isinstance(cell, float) and np.isnan(cell)):
            return False
    return True

def classification(imported_data, model):
    x0 = imported_data['macromoleculeType']
    x1 = imported_data['residueCount']
    x2 = imported_data['structureMolecularWeight']
    x3 = imported_data['densityMatthews']
    x4 = imported_data['densityPercentSol']
    x5 = imported_data['phValue']

    sample_class = imported_data['classification']

    dataset = np.array([x0,x1,x2,x3,x4,x5,sample_class], dtype=object)
    dataset = dataset.T

    dataset = np.array([row for row in dataset if row_validation(row)], dtype=object)

    encoder = LabelEncoder()
    for i in range(dataset.shape[1]):
        if dataset[:, i].dtype == object:  # Check if column has strings
            dataset[:, i] = encoder.fit_transform(dataset[:, i])

    dataset = dataset.astype(np.float32)

    features = dataset[:,:-1]
    labels = dataset[:,-1]

    if model == 'rf':
        random_forest(features, labels)
    elif model == 'dt':
        decision_tree(features, labels)
    elif model == 'nn':
        neural_network(features,labels)

def random_forest(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, max_depth=25)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    my_evaluate(y_test,y_pred)

def decision_tree(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    my_evaluate(y_test, y_pred)

def neural_network(data, labels):
    data = np.expand_dims(data, axis=-1)
    num_classes = len(np.unique(labels))

    #Establishes training/testing data
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.25)

    #Defines the cnn model to be used and the layers that will be used in the process
    model = Sequential([
        Flatten(input_shape=(data.shape[1],data.shape[2])),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    #Defines the optimizer, the learning rate, and compiles the model using given parameters
    opt = Adam(learning_rate=1e-3)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    #Displays live model data processing and fits the training data
    model.summary()
    model.fit(X_train, y_train, epochs=15)
    print("Finish training")

    #Displays test label and compares it to label the cnn predicted
    y_pred_prob = model.predict(X_test)
    print("y_test:", y_test)
    print("y_pred_prob:", y_pred_prob)

    y_pred, truth = [], []
    for prob in y_pred_prob:
        y_pred.append(np.argmax(prob))
    for prob in y_test:
        truth.append(np.argmax(prob))

    #Displays accuracy of the model given the performance across all samples
    my_evaluate(truth, y_pred)


def my_evaluate(truth, y_pred):
    print("Accuracy:", accuracy_score(truth, y_pred) * 100, "%")