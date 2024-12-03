import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorboard.compat.tensorflow_stub import string
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def row_validation(row):
    for cell in row:
        # Check for NaN, None, or blank strings
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

    dataset = np.array([x1,x2,x3,x4,x5,sample_class], dtype=object)
    dataset = dataset.T

    dataset = np.array([row for row in dataset if row_validation(row)], dtype=object)

    n = len(dataset[0])
    features = dataset[:,:(n-1)]
    labels = dataset[:,-1]

    # labels = []
    # for label in clean_dataset[:,4]:
    #     if label == 'DNA/RNA Hybrid':
    #         labels.append(0)
    #     elif label == 'DNA':
    #         labels.append(1)
    #     elif label == 'Protein':
    #         labels.append(2)
    #     elif label == 'Protein#DNA':
    #         labels.append(3)
    #     elif label == 'DNA#RNA':
    #         labels.append(4)
    #     elif label == 'RNA':
    #         labels.append(5)
    #     elif label == 'DNA#DNA/RNA Hybrid':
    #         labels.append(6)
    #     elif label == 'Protein#RNA':
    #         labels.append(7)
    #     elif label == 'RNA#DNA/RNA Hybrid':
    #         labels.append(8)
    #     elif label == 'Protein#DNA/RNA Hybrid':
    #         labels.append(9)
    #     elif label == 'Protein#DNA#RNA':
    #         labels.append(10)
    #     elif label == 'Protein#DNA#DNA/RNA Hybrid':
    #         labels.append(11)
    #     elif label == 'Protein#RNA#DNA/RNA Hybrid':
    #         labels.append(12)
    #
    # labels = np.array(labels)
    # labels = labels[np.newaxis,:].T
    #
    # data = clean_dataset[:, 0:3]
    # print('og data =', dataset[:3])
    # print("data =", data.shape)
    # print(data[:3])
    # print("label =",labels.shape)
    # print(labels[:3])

    if model == 'rf':
        random_forest(features, labels)
    elif model == 'dt':
        decision_tree(features, labels)
    # elif model == 'CNN':
    #     cnn(dataset,labels)

def random_forest(features,labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=25, max_depth=50)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    my_evaluate(y_test,y_pred)

def decision_tree(features,labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    my_evaluate(y_test, y_pred)

# def cnn(data, labels, rs=42, epochs=10, batch_size=16):
#     #Sets value N to define the length of the dataset
#     N = range(len(data))
#
#     #Establishes training/testing data
#     X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.25,random_state=rs)
#
#     X_train = np.array(X_train)
#     X_test = np.array(X_test)
#     y_train = np.array(y_train)
#     y_test = np.array(y_test)
#
#     #Defines the cnn model to be used and the layers that will be used in the process
#     model = Sequential([
#         # This extracts features like edges, textures, and patterns
#         Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(N, 3)),
#         # This reduces the length/width of data while keeping important info.
#         MaxPooling1D(pool_size=2),
#         Conv1D(filters=32, kernel_size=3, activation='relu'),
#         MaxPooling1D(pool_size=2),
#         # "Flattens" all outputs into a one dimensional shape
#         Flatten(),
#         # Combines features into class scores/probabilities, then uses
#         Dense(32, activation='relu'),
#         Dense(3, activation='softmax')
#     ])
#
#     #Defines the optimizer, the learning rate, and compiles the model using given parameters
#     opt = Adam(learning_rate=1e-3)
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#
#     #Displays live model data processing and fits the training data
#     model.summary()
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
#     print("Finish training")
#
#     #Displays test label and compares it to label the cnn predicted
#     y_pred_prob = model.predict(X_test)
#     print("y_test:", y_test)
#     print("y_pred_prob:", y_pred_prob)
#
#     y_pred, truth = [], []
#     for prob in y_pred_prob:
#         y_pred.append(np.argmax(prob))
#     for prob in y_test:
#         truth.append(np.argmax(prob))
#
#     #Displays accuracy of the model given the performance across all samples
#     my_evaluate(truth, y_pred)


def my_evaluate(truth, y_pred):
    print("Accuracy:", accuracy_score(truth, y_pred) * 100, "%")